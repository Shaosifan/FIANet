import datetime
import os
import time
import torch
import torch.utils.data
import wandb
import random
import transforms as T
import utils
import numpy as np
import gc
import operator
from functools import reduce
from bert.modeling_bert import BertModel
from lib import segmentation
from loss.loss import Loss

os.environ["WANDB_API_KEY"] = '1ae5903bce9def26f040e6a15cc95aba3a99cc91'
os.environ["WANDB_MODE"] = "offline"

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def seed_everything(seed=2401):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_dataset(image_set, transform, args):
    if args.dataset == "rrsisd":
        from data.rrsisd_refer_bert import ReferDataset
    else:
        from data.refsegrs_refer_bert import ReferDataset
    ds = ReferDataset(args,
                      split=image_set,
                      image_transforms=transform,
                      target_transforms=None
                      )
    num_classes = 2

    return ds, num_classes


def IoU(pred, gt):
    pred = pred.argmax(1)

    intersection = torch.sum(torch.mul(pred, gt))
    union = torch.sum(torch.add(pred, gt)) - intersection

    if intersection == 0 or union == 0:
        iou = 0
    else:
        iou = float(intersection) / float(union)
    return iou, intersection, union


def get_transform(args):
    transforms = [
                  T.Resize(args.img_size, args.img_size),
                  T.ToTensor(),
                  T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                  ]
    return T.Compose(transforms)


def criterion(input, target, weight=0.1):
    return Loss(weight=weight)(input, target)


def evaluate(model, data_loader, bert_model, epoch):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test: "
    total_its = 0
    acc_ious = 0

    # evaluation variables
    cum_I, cum_U = 0, 0
    eval_seg_iou_list = [.5, .6, .7, .8, .9]
    seg_correct = np.zeros(len(eval_seg_iou_list), dtype=np.int32)
    seg_total = 0
    mean_IoU = []
    total_loss = 0

    with torch.no_grad():
        for data in metric_logger.log_every(data_loader, 100, header):
            total_its += 1
            image, target, sentences, attentions, target_masks, position_masks, _ = data
            image = image.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            sentences = sentences.cuda(non_blocking=True)
            attentions = attentions.cuda(non_blocking=True)
            target_masks = target_masks.cuda(non_blocking=True)
            position_masks = position_masks.cuda(non_blocking=True)

            sentences = sentences.squeeze(1)
            attentions = attentions.squeeze(1)
            target_masks = target_masks.squeeze(1)
            position_masks = position_masks.squeeze(1)

            if bert_model is not None:
                last_hidden_states = bert_model(sentences, attention_mask=attentions)[0]
                embedding = last_hidden_states.permute(0, 2, 1)  # (B, 768, N_l) to make Conv1d happy
                attentions = attentions.unsqueeze(dim=-1)  # (B, N_l, 1)
                output = model(image, embedding, l_mask=attentions)
            else:
                output = model(image, sentences, attentions, target_masks, position_masks)

            iou, I, U = IoU(output, target)
            loss = criterion(output, target)
            total_loss += loss.item()
            acc_ious += iou
            mean_IoU.append(iou)
            cum_I += I
            cum_U += U
            for n_eval_iou in range(len(eval_seg_iou_list)):
                eval_seg_iou = eval_seg_iou_list[n_eval_iou]
                seg_correct[n_eval_iou] += (iou >= eval_seg_iou)
            seg_total += 1
        iou = acc_ious / total_its

    mean_IoU = np.array(mean_IoU)
    mIoU = np.mean(mean_IoU)
    print('Final results:')
    print('Mean IoU is %.2f\n' % (mIoU * 100.))
    results_str = ''
    for n_eval_iou in range(len(eval_seg_iou_list)):
        results_str += '    precision@%s = %.2f\n' % \
                       (str(eval_seg_iou_list[n_eval_iou]), seg_correct[n_eval_iou] * 100. / seg_total)
    results_str += '    overall IoU = %.2f\n' % (cum_I * 100. / cum_U)
    print(results_str)

    if args.local_rank == 0:
        wandb.log({
            "val mIoU": mIoU,
            "val oiou": cum_I * 100. / cum_U,
            "val Loss": total_loss / total_its})

    return 100 * iou, 100 * cum_I / cum_U


def train_one_epoch(model, criterion, optimizer, data_loader, lr_scheduler, epoch, print_freq,
                    iterations, bert_model):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value}'))
    header = 'Epoch: [{}]'.format(epoch)
    train_loss = 0
    total_its = 0

    # for data in data_loader:
    for i, data in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        total_its += 1
        image, target, sentences, attentions, target_masks, position_masks, _ = data
        image          = image.cuda(non_blocking=True)
        target         = target.cuda(non_blocking=True)
        sentences      = sentences.cuda(non_blocking=True)
        attentions     = attentions.cuda(non_blocking=True)
        target_masks   = target_masks.cuda(non_blocking=True)
        position_masks = position_masks.cuda(non_blocking=True)

        sentences      = sentences.squeeze(1)
        attentions     = attentions.squeeze(1)
        target_masks   = target_masks.squeeze(1)
        position_masks = position_masks.squeeze(1)

        if bert_model is not None:
            last_hidden_states = bert_model(sentences, attention_mask=attentions)[0]  # (6, 10, 768)
            embedding = last_hidden_states.permute(0, 2, 1)  # (B, 768, N_l) to make Conv1d happy
            attentions = attentions.unsqueeze(dim=-1)  # (batch, N_l, 1)
            output = model(image, embedding, attentions)#, sentences_hidden_state)# [4,2,120,120]
        else:
            output = model(image, sentences, attentions, target_masks, position_masks)
            # output = model(image, sentences, attentions)
        optimizer.zero_grad()
        loss = criterion(output, target)

        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        torch.cuda.synchronize()
        train_loss += loss.item()
        iterations += 1
        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])

        del image, target, sentences, attentions, loss, output, data
        if bert_model is not None:
            del last_hidden_states, embedding

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    if args.local_rank == 0:
        wandb.log({
            "Train Loss": train_loss / total_its,})


def main(args):

    # make folders
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    # set datasets
    print("\n[***] Set Datasets")
    dataset, num_classes = get_dataset("train",
                                       get_transform(args=args),
                                       args=args)

    dataset_test, _ = get_dataset("test",
                                  get_transform(args=args),
                                  args=args)

    # Debug = False
    # if Debug:
    #     dataset = torch.utils.data.Subset(dataset, list(range(200)))
    #     dataset_test = torch.utils.data.Subset(dataset_test, list(range(20)))

    # build batch sampler
    print(f"local rank {args.local_rank} / global rank {utils.get_rank()} successfully built train dataset.")
    train_sampler = torch.utils.data.RandomSampler(dataset)
    test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    # build data loader
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size,
        sampler=train_sampler, num_workers=args.workers, pin_memory=args.pin_mem, drop_last=True)
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, sampler=test_sampler, num_workers=args.workers)

    # model initialization
    print("\n[***] Build Model")
    model = segmentation.__dict__[args.model](pretrained=args.pretrained_swin_weights,
                                              args=args)
    model.cuda()
    # print(args.model)

    if args.model != 'lavt_one':
        # need to load bert outside the model
        model_class = BertModel
        bert_model = model_class.from_pretrained(args.ck_bert)
        bert_model.pooler = None  # a work-around for a bug in Transformers = 3.0.2 that appears for DistributedDataParallel
        bert_model.cuda()
    else:
        bert_model = None
        single_bert_model = None

    # resume training
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'], strict=False)
        if args.model != 'lavt_one':
            bert_model.load_state_dict(checkpoint['bert_model'])

    # parameters to optimize
    backbone_no_decay = list()
    backbone_decay = list()
    for name, m in model.backbone.named_parameters():
        if 'norm' in name or 'absolute_pos_embed' in name or 'relative_position_bias_table' in name:
            backbone_no_decay.append(m)
        else:
            backbone_decay.append(m)

    if args.model != 'lavt_one':
        params_to_optimize = [
            {'params': backbone_no_decay, 'weight_decay': 0.0},
            {'params': backbone_decay},
            {"params": [p for p in model.classifier.parameters() if p.requires_grad]},
            # the following are the parameters of bert
            {"params": reduce(operator.concat,
                              [[p for p in bert_model.module.encoder.layer[i].parameters()
                                if p.requires_grad] for i in range(10)])},
        ]
    else:
        params_to_optimize = [
            {'params': backbone_no_decay, 'weight_decay': 0.0},
            {'params': backbone_decay},
            {"params": [p for p in model.classifier.parameters() if p.requires_grad]},
            # the following are the parameters of bert
            {"params": reduce(operator.concat,
                              [[p for p in model.text_encoder.encoder.layer[i].parameters()
                                if p.requires_grad] for i in range(10)])},
        ]

    # optimizer
    optimizer = torch.optim.AdamW(params_to_optimize,
                                  lr=args.lr,
                                  weight_decay=args.weight_decay,
                                  amsgrad=args.amsgrad
                                  )

    # learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                     lambda x: (1 - x / (len(data_loader) * args.epochs)) ** 0.9)

    # housekeeping
    start_time = time.time()
    iterations = 0
    best_oIoU = -0.1

    # resume training (optimizer, lr scheduler, and the epoch)
    if args.resume:
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        resume_epoch = checkpoint['epoch']
        best_oIoU = checkpoint['best_oIoU']
        print("Resume training from Epoch {}".format(resume_epoch+1))
        print("Initail best IoU is {}".format(best_oIoU))

    else:
        resume_epoch = -999

    # training loops
    if args.local_rank == 0:
        wandb.watch(model, log="all")

    for epoch in range(max(0, resume_epoch+1), args.epochs):
        # data_loader.sampler.set_epoch(epoch)
        train_one_epoch(model, criterion, optimizer, data_loader, lr_scheduler, epoch, args.print_freq,
                        iterations, bert_model)
        iou, overallIoU = evaluate(model, data_loader_test, bert_model, epoch)
        print('Average object IoU {}'.format(iou))
        print('Overall IoU {}'.format(overallIoU))
        best = (best_oIoU < overallIoU)
        if bert_model is not None:
            dict_to_save = {'model': model.state_dict(), 'bert_model': bert_model.state_dict(),
                            'optimizer': optimizer.state_dict(), 'epoch': epoch, 'args': args,
                            'lr_scheduler': lr_scheduler.state_dict(),
                            'best_oIoU': best_oIoU}
        else:
            dict_to_save = {'model': model.state_dict(),
                            'optimizer': optimizer.state_dict(), 'epoch': epoch, 'args': args,
                            'lr_scheduler': lr_scheduler.state_dict(),
                            'best_oIoU': best_oIoU}

        if best:
            best_oIoU = overallIoU
            print('Better epoch: {}\n'.format(epoch))
            dict_to_save['best_oIoU'] = best_oIoU
            utils.save_on_master(
                dict_to_save, os.path.join(args.output_dir, 'model_best_{}.pth'.format(args.model_id)))

        if epoch % 10 == 0:
            utils.save_on_master(dict_to_save, os.path.join(args.output_dir,
                                                        'model_last_{}_{}.pth'.format(args.model_id, epoch)))
        utils.save_on_master(dict_to_save, os.path.join(args.output_dir,
                                                        'model_last_{}.pth'.format(args.model_id)))
        if args.local_rank == 0:
            wandb.save('model.h5')

    # summarize
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == "__main__":
    from args import get_parser
    seed_everything()
    parser = get_parser()
    args = parser.parse_args()
    if args.local_rank == 0:
        wandb.init(project="fianet_2080")
    print('Image size: {}'.format(str(args.img_size)))
    main(args)
