import os
import torch.utils.data as data
import torch
import numpy as np
from PIL import Image
import random
from bert.tokenization_bert import BertTokenizer
from refer.refer import REFER

from args import get_parser

import re
import nltk
from nltk.tokenize import word_tokenize

# Dataset configuration initialization
parser = get_parser()
args = parser.parse_args()


def add_random_boxes(img, min_num=20, max_num=60, size=32):
    h,w = size, size
    img = np.asarray(img).copy()
    img_size = img.shape[1]
    boxes = []
    num = random.randint(min_num, max_num)
    for k in range(num):
        y, x = random.randint(0, img_size-w), random.randint(0, img_size-h)
        img[y:y+h, x: x+w] = 0
        boxes. append((x,y,h,w) )
    img = Image.fromarray(img.astype('uint8'), 'RGB')
    return img


class ReferDataset(data.Dataset):

    def __init__(self,
                 args,
                 image_transforms=None,
                 target_transforms=None,
                 split='train',
                 eval_mode=False):

        self.classes = []
        self.image_transforms = image_transforms
        self.target_transform = target_transforms
        self.split = split
        self.refer = REFER(args.refer_data_root, args.dataset, args.splitBy)

        self.max_tokens = 22

        ref_ids = self.refer.getRefIds(split=self.split)
        img_ids = self.refer.getImgIds(ref_ids)

        num_images_to_mask = int(len(ref_ids) * 0.2)
        self.images_to_mask = random.sample(ref_ids, num_images_to_mask)

        all_imgs = self.refer.Imgs
        self.imgs = list(all_imgs[i] for i in img_ids)
        self.ref_ids = ref_ids

        self.input_ids = []
        self.attention_masks = []
        self.tokenizer = BertTokenizer.from_pretrained(args.bert_tokenizer)

        # for ground target and spatial position
        self.target_masks = []
        self.position_masks = []

        self.sentense_raw = []
        self.pp_phrase = []

        # debug
        self.max_len = 0

        # for RRSIS-D dataset
        self.target_cls = {"airplane", "airport", "golf field", "expressway service area", "baseball field","stadium",
                      "ground track field", "storage tank", "basketball court", "chimney", "tennis court", "overpass",
                      "train station", "ship", "expressway toll station", "dam", "harbor", "bridge", "vehicle",
                      "windmill"}

        self.eval_mode = eval_mode
        # if we are testing on a dataset, test all sentences of an object;
        # o/w, we are validating during training, randomly sample one sentence for efficiency
        for r in ref_ids:
            ref = self.refer.Refs[r]

            sentences_for_ref = []
            attentions_for_ref = []

            target_for_ref = []
            position_for_ref = []

            for i, (el, sent_id) in enumerate(zip(ref['sentences'], ref['sent_ids'])):
                sentence_raw = el['raw']
                attention_mask = [0] * self.max_tokens
                padded_input_ids = [0] * self.max_tokens

                target_masks = [0] * self.max_tokens
                position_masks = [0] * self.max_tokens

                input_ids = self.tokenizer.encode(text=sentence_raw, add_special_tokens=True)

                # truncation of tokens
                input_ids = input_ids[:self.max_tokens]

                padded_input_ids[:len(input_ids)] = input_ids
                attention_mask[:len(input_ids)] = [1]*len(input_ids)

                sentences_for_ref.append(torch.tensor(padded_input_ids).unsqueeze(0))
                attentions_for_ref.append(torch.tensor(attention_mask).unsqueeze(0))

                # extract the ground object
                # print(sentence_raw)
                self.sentense_raw.append(sentence_raw)
                tokenized_sentence = word_tokenize(sentence_raw)

                for cls in self.target_cls:
                    if re.findall(cls, sentence_raw):

                        tokenized_cls = word_tokenize(cls)
                        nums_cls = len(tokenized_cls)
                        index = 0
                        for i, token in enumerate(tokenized_sentence):
                            if re.findall(tokenized_cls[0], token):
                                index = i
                                break
                        target_masks[index + 1: index + nums_cls +1] = [1] * nums_cls
                # print(target_masks)

                target_for_ref.append(torch.tensor(target_masks).unsqueeze(0))

                # extract the spatial position
                grammar = r"""
                PP: {<IN><DT>?<JJ.*>?<NN>}
                    {<IN><DT>?<JJ.*>?<JJ>}
                    {<IN><DT>?<JJ.*><VBD>}
                """
                chunkr = nltk.RegexpParser(grammar)
                # grammar parsing
                tree = chunkr.parse(nltk.pos_tag(tokenized_sentence))
                pp_phrases = []
                for subtree in tree.subtrees():
                    if subtree.label() == 'PP':
                        pp_phrases.append(' '.join(word for word, pos in subtree.leaves()))

                new_pp_phrase = []
                for phrase in pp_phrases:
                    if not re.findall("of", phrase):
                        new_pp_phrase.append(phrase)

                if len(new_pp_phrase) > 0:
                    tokenized_sentence = word_tokenize(sentence_raw)
                    for pp in new_pp_phrase:
                        tokenized_pos = word_tokenize(pp)
                        nums_pos = len(tokenized_pos)
                        index = 0
                        for i, token in enumerate(tokenized_sentence):
                            if tokenized_pos[0] == token:
                                index = i
                                break
                        position_masks[index + 1: index + nums_pos +1] = [1] * nums_pos

                self.pp_phrase.append(new_pp_phrase)
                position_for_ref.append(torch.tensor(position_masks).unsqueeze(0))
                # if there are no pp, the position_for_ref equals attentions_for_ref
                if torch.sum(position_for_ref[0]) == 0:
                    position_for_ref = attentions_for_ref

            self.input_ids.append(sentences_for_ref)
            self.attention_masks.append(attentions_for_ref)
            self.target_masks.append(target_for_ref)
            self.position_masks.append(position_for_ref)


    def get_classes(self):
        return self.classes

    def __len__(self):
        return len(self.ref_ids)

    def __getitem__(self, index):
        this_ref_id = self.ref_ids[index]
        this_img_id = self.refer.getImgIds(this_ref_id)
        this_img = self.refer.Imgs[this_img_id[0]]

        img = Image.open(os.path.join(self.refer.IMAGE_DIR, this_img['file_name']))
        if self.split == 'train' and this_ref_id in self.images_to_mask:
            img = add_random_boxes(img)

        ref = self.refer.loadRefs(this_ref_id)

        ref_mask = np.array(self.refer.getMask(ref[0])['mask'])
        annot = np.zeros(ref_mask.shape)
        annot[ref_mask == 1] = 1

        annot = Image.fromarray(annot.astype(np.uint8), mode="P")

        sentence = ref[0]['sentences'][0]['raw']
        save_prefix = str(ref[0]['image_id']) + "_" + sentence

        if self.image_transforms is not None:
            # resize, from PIL to tensor, and mean and std normalization
            # Leisen Debug: write the input images and labels
            SHOW_INPUT = False
            if SHOW_INPUT:
                import cv2
                save_dir = "experiments/input_vis"

                # write in the type of image and label
                img.save(os.path.join(save_dir, save_prefix + "_image.png"))
                mask = ref_mask * 255
                cv2.imwrite(os.path.join(save_dir, save_prefix + "_label.png"), mask)

            img, target = self.image_transforms(img, annot)

        choice_sent = np.random.choice(len(self.input_ids[index]))
        tensor_embeddings = self.input_ids[index][choice_sent]
        attention_mask = self.attention_masks[index][choice_sent]
        target_mask = self.target_masks[index][choice_sent]
        position_mask = self.position_masks[index][choice_sent]

        # bebug
        # print(img.size(), target.size(), tensor_embeddings.size(), attention_mask.size(),
        #     target_masks.size(), position_masks.size())
        # print(self.sentense_raw[index])
        # print(self.pp_phrase[index])
        # print(self.position_masks[index])
        return img, target, tensor_embeddings, attention_mask, target_mask, position_mask, save_prefix


