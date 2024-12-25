import cv2
import torch
import torch.utils.data
import utils
import numpy as np
import transforms as T
from torchvision.transforms import functional as F
import random
from bert.modeling_bert import BertModel
from lib import segmentation
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import time
import datetime

def get_dataset(image_set, transform, args):

    if args.dataset == "rrsisd":
        from data.rrsisd_refer_bert import ReferDataset
    else:
        from data.refsegrs_refer_bert import ReferDataset
    ds = ReferDataset(args,
                      split=image_set,
                      image_transforms=transform,
                      target_transforms=None,
                      eval_mode=True
                      )
    num_classes = 2
    return ds, num_classes


def evaluate(model, data_loader, bert_model, device):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")

    # evaluation variables
    cum_I, cum_U = 0, 0
    eval_seg_iou_list = [.5, .6, .7, .8, .9]
    seg_correct = np.zeros(len(eval_seg_iou_list), dtype=np.int32)
    seg_total = 0
    mean_IoU = []
    header = 'Test:'
    save_dir = "experiments/test_vis"

    start_time = time.time()

    with torch.no_grad():

        for data in metric_logger.log_every(data_loader, 100, header):

            image, target, sentences, attentions, target_masks, position_masks, save_prefix = data
            image = image.to(device)
            target = target.to(device)
            sentences = sentences.to(device)
            attentions = attentions.to(device)
            target = target.to(device)

            target_masks = target_masks.to(device)
            position_masks = position_masks.to(device)

            sentences = sentences.squeeze(1)
            attentions = attentions.squeeze(1)
            target_masks = target_masks.squeeze(1)
            position_masks = position_masks.squeeze(1)

            target = target.cpu().data.numpy()
            for j in range(sentences.size(0)):
                if bert_model is not None:
                    last_hidden_states = bert_model(sentences[:, :, j], attention_mask=attentions[:, :, j])[0]
                    embedding = last_hidden_states.permute(0, 2, 1)
                    output = model(image, embedding, l_mask=attentions[:, :, j].unsqueeze(-1))
                else:
                    # output = model(image, sentences[:, :, j], l_mask=attentions[:, :, j])

                    # output = model(image, sentences[:, :, j], attentions[:, :, j], target_masks[:, :, j], position_masks[:, :, j])
                    output = model(image, sentences, attentions, target_masks, position_masks)

                output = output.cpu()

                output_mask = output.argmax(1).data.numpy()

                I, U = computeIoU(output_mask, target)

                # save pred results
                # save_path = os.path.join(save_dir, str(seg_total+1))
                # save_pred_targ_results(output_mask, target, image, save_path)

                if U == 0:
                    this_iou = 0.0
                else:
                    this_iou = I*1.0/U
                mean_IoU.append(this_iou)
                cum_I += I
                cum_U += U
                for n_eval_iou in range(len(eval_seg_iou_list)):
                    eval_seg_iou = eval_seg_iou_list[n_eval_iou]
                    seg_correct[n_eval_iou] += (this_iou >= eval_seg_iou)

                # mask = output_mask * 255
                # cv2.imwrite(os.path.join(save_dir, save_prefix[0] + "_" + str(this_iou) + "_pred.png"), mask[0, :, :])

                seg_total += 1

            del image, target, sentences, attentions, output,output_mask
            if bert_model is not None:
                del last_hidden_states, embedding

    mean_IoU = np.array(mean_IoU)
    mIoU = np.mean(mean_IoU)
    print('Final results:')
    print('Mean IoU is %.2f\n' % (mIoU*100.))
    results_str = ''
    for n_eval_iou in range(len(eval_seg_iou_list)):
        results_str += '    precision@%s = %.2f\n' % \
                       (str(eval_seg_iou_list[n_eval_iou]), seg_correct[n_eval_iou] * 100. / seg_total)
    results_str += '    overall IoU = %.2f\n' % (cum_I * 100. / cum_U)
    print(results_str)

    # summarize
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Total test time {}'.format(total_time_str))
    print('Test time for one image %.2f ' % (total_time / seg_total))


def get_transform(args):
    transforms = [T.Resize(args.img_size, args.img_size),
                  T.ToTensor(),
                  T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                  ]

    return T.Compose(transforms)


def computeIoU(pred_seg, gd_seg):
    I = np.sum(np.logical_and(pred_seg, gd_seg))
    U = np.sum(np.logical_or(pred_seg, gd_seg))

    return I, U

def save_pred_targ_results(output_mask, target, image, save_path):
    "leisen: show the predict and target outcomes"
    pred = output_mask[0, :, :]
    targ = target[0, :, :]
    # pred = np.uint8(pred * 255)
    # targ = np.uint8(targ * 255)
    # cv2.imwrite("pred.png", pred)
    # cv2.imwrite("targ.png", targ)

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    inv_mean = [-m / s for m, s in zip(mean, std)]
    inv_std = [1 / s for s in std]
    im = F.normalize(image, mean=inv_mean, std=inv_std)
    im = im[0, :, :, :].cpu().detach().numpy()
    im = im.transpose([1, 2, 0])
    im = np.uint8(im * 255)

    pred_red_mask = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
    pred_red_mask[:, :] = (0, 0, 255)
    pred_mask = np.repeat(pred[:, :, np.newaxis], 3, -1)
    pred_mask = pred_red_mask * pred_mask
    pred_mask = np.uint8(pred_mask)
    pred_img_write = cv2.addWeighted(im, 0.5, pred_mask, 0.5, 0)
    cv2.imwrite((save_path + "_pred.png"), pred_img_write)

    targ_red_mask = np.zeros((targ.shape[0], targ.shape[1], 3), dtype=np.uint8)
    targ_red_mask[:, :] = (0, 0, 255)
    targ_mask = np.repeat(targ[:, :, np.newaxis], 3, -1)
    targ_mask = targ_red_mask * targ_mask
    targ_mask = np.uint8(targ_mask)
    targ_img_write = cv2.addWeighted(im, 0.5, targ_mask, 0.5, 0)
    cv2.imwrite((save_path + "_targ.png"), targ_img_write)


def main(args):
    device = torch.device(args.device)
    dataset_test, _ = get_dataset(args.split, get_transform(args=args), args)

    test_sampler = torch.utils.data.SequentialSampler(dataset_test)
    data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1,
                                                   sampler=test_sampler, num_workers=args.workers)
    print(args.model)
    single_model = segmentation.__dict__[args.model](pretrained='',args=args)
    checkpoint = torch.load(args.resume, map_location='cpu')
    single_model.load_state_dict(checkpoint['model'], strict=False)
    model = single_model.to(device)

    if args.model != 'lavt_one':
        model_class = BertModel
        single_bert_model = model_class.from_pretrained(args.ck_bert)
        # work-around for a transformers bug; need to update to a newer version of transformers to remove these two lines
        if args.ddp_trained_weights:
            single_bert_model.pooler = None
        single_bert_model.load_state_dict(checkpoint['bert_model'])
        bert_model = single_bert_model.to(device)
    else:
        bert_model = None

    evaluate(model, data_loader_test, bert_model, device=device)


if __name__ == "__main__":
    from args import get_parser
    parser = get_parser()
    args = parser.parse_args()
    print('Image size: {}'.format(str(args.img_size)))
    main(args)
