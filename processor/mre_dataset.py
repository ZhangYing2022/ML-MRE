import random
import os
import torch
import json
import ast
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from torchvision import transforms
import logging

logger = logging.getLogger(__name__)


class MREProcessor(object):
    def __init__(self, data_path, re_path, bert_name, clip_processor=None, aux_processor=None, rcnn_processor=None):
        self.data_path = data_path
        self.re_path = re_path
        self.tokenizer = BertTokenizer.from_pretrained(bert_name, do_lower_case=True)
        self.tokenizer.add_special_tokens({'additional_special_tokens': ['<s>', '</s>', '<e>', '</e>']})
        self.clip_processor = clip_processor
        self.aux_processor = aux_processor
        self.rcnn_processor = rcnn_processor

    def load_from_file(self, mode="train"):
        load_file = self.data_path[mode]
        logger.info("Loading data from {}".format(load_file))
        with open(load_file, "r", encoding="utf-8") as f:
            #lines = f.readlines()
            lines = json.load(f)
            words, relations, heads, tails, imgids, dataid = [], [], [], [], [], []
            for i, line in enumerate(lines):
                #line = ast.literal_eval(line)  # str to dict
                words.append(line['token'])
                relations.append(line['relation'])
                heads.append(line['h'])  # {name, pos}
                tails.append(line['t'])
                #1.27
                # heads.append(line['label_list'][0][0]['beg_ent'])  # {name, pos}
                # tails.append(line['label_list'][0][0]['sec_ent'])
                imgids.append(line['img_id'])
                dataid.append(i)

        assert len(words) == len(relations) == len(heads) == len(tails) == (len(imgids))

        aux_imgs = None
        rcnn_imgs = None
        # aux_path = self.data_path[mode + "_auximgs"]
        # aux_imgs = torch.load(aux_path)
        # rcnn_imgs = torch.load(self.data_path[mode + '_img2crop'])

        return {'words': words, 'relations': relations, 'heads': heads, 'tails': tails, 'imgids': imgids,
                'dataid': dataid, 'aux_imgs': aux_imgs, "rcnn_imgs": rcnn_imgs}

    
    def get_relation_dict(self, mode="train"):
        """
        遍历指定mode的数据集文件，自动统计所有标签，返回{标签: id}字典
        """
        load_file = self.data_path["train"]
        all_labels = set()
        with open(load_file, "r", encoding="utf-8") as f:
            lines = json.load(f)
            for line in lines:
                relation = line['relation']
                if isinstance(relation, list):
                    all_labels.update(relation)
                else:
                    all_labels.add(relation)
        re_dict = {label: idx for idx, label in enumerate(sorted(all_labels))}
        print(re_dict)
        return re_dict
   

   
    def get_rel2id(self, train_path):
        with open(self.re_path, 'r', encoding="utf-8") as f:
            # line = f.readlines()[0]
            # re_dict = json.loads(line)
            re_dict = json.load(f)
        re2id = {key: [] for key in re_dict.keys()}
        with open(train_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                line = ast.literal_eval(line)  # str to dict
                assert line['relation'] in re2id
                re2id[line['relation']].append(i)
        return re2id


class MREDataset(Dataset):
    def __init__(self, processor, img_path=None, aux_img_path=None, max_seq=40, aux_size=128, rcnn_size=64,
                 mode="train", write_path=None, do_test=False) -> None:
        self.processor = processor
        self.max_seq = max_seq
        self.img_path = img_path[mode] if img_path is not None else img_path
        self.aux_img_path = aux_img_path[mode] if aux_img_path is not None else aux_img_path
        self.rcnn_img_path = None
        self.mode = mode
        self.data_dict = self.processor.load_from_file(mode)
        self.tokenizer = self.processor.tokenizer
        self.clip_processor = self.processor.clip_processor
        self.aux_processor = self.processor.aux_processor
        self.rcnn_processor = self.processor.rcnn_processor
        self.aux_size = aux_size
        self.rcnn_size = rcnn_size
        self.write_path = write_path
        self.do_test = do_test

        # 自动构建标签字典
        all_labels = set()
        for rel in self.processor.load_from_file("train")['relations']:
            if isinstance(rel, list):
                all_labels.update(rel)
            else:
                all_labels.add(rel)
        self.re_dict = {label: idx for idx, label in enumerate(sorted(all_labels))}

        # #9.12
        self.all_ids = self.data_dict['dataid']
        self.unlabeled_ids = self.all_ids.copy()
        self.labeled_ids = []
        self.sample_ids = self.all_ids.copy()  # default for val and test datapool
       

    def __len__(self):
        #return len(self.data_dict['words'])
        return len(self.sample_ids)

    def __getitem__(self, idx):
        real_idx = int(self.sample_ids[idx])

        word_list, relation, head_d, tail_d, imgid = self.data_dict['words'][real_idx], self.data_dict['relations'][real_idx], \
                                                     self.data_dict['heads'][real_idx], self.data_dict['tails'][real_idx], \
                                                     self.data_dict['imgids'][real_idx]
        item_id = self.data_dict['dataid'][real_idx]
        # [CLS] ... <s> head </s> ... <o> tail <o/> .. [SEP]
        head_pos, tail_pos = head_d['pos'], tail_d['pos']
        # insert <s> <s/> <o> <o/>
        extend_word_list = []
        for i in range(len(word_list)):
            if i == head_pos[0]:
                extend_word_list.append('<s>')
            if i == head_pos[1]:
                extend_word_list.append('</s>')
            if i == tail_pos[0]:
                extend_word_list.append('<e>')
            if i == tail_pos[1]:
                extend_word_list.append('</e>')
            extend_word_list.append(word_list[i])
        extend_word_list = " ".join(extend_word_list)

        encode_dict = self.tokenizer.encode_plus(text=extend_word_list, max_length=self.max_seq, truncation=True,
                                                 padding='max_length')
        input_ids, token_type_ids, attention_mask = encode_dict['input_ids'], encode_dict['token_type_ids'], \
                                                    encode_dict['attention_mask']
        input_ids, token_type_ids, attention_mask = torch.tensor(input_ids), torch.tensor(token_type_ids), torch.tensor(
            attention_mask)

        # relation 现在是标签列表
        if isinstance(relation, list):
            num_classes = len(self.re_dict)
            re_label = torch.zeros(num_classes)
            for rel in relation:
                re_label[self.re_dict[rel]] = 1
        else:
            re_label = torch.zeros(len(self.re_dict))
            re_label[self.re_dict[relation]] = 1

        # image process
        if self.img_path is not None:
            try:
                img_path = os.path.join(self.img_path, imgid)
                image = Image.open(img_path).convert('RGB')
                image = self.clip_processor(images=image, return_tensors='pt')['pixel_values'].squeeze()
            except:
                img_path = os.path.join(self.img_path, 'inf.png')
                image = Image.open(img_path).convert('RGB')
                image = self.clip_processor(images=image, return_tensors='pt')['pixel_values'].squeeze()
            if self.aux_img_path is not None:
                # detected object img
                aux_imgs = []
                aux_img_paths = []
                imgid = imgid.split(".")[0]
                if item_id in self.data_dict['aux_imgs']:
                    aux_img_paths = self.data_dict['aux_imgs'][item_id]
                    aux_img_paths = [os.path.join(self.aux_img_path, path) for path in aux_img_paths]
                # select 3 img
                for i in range(min(3, len(aux_img_paths))):
                    aux_img = Image.open(aux_img_paths[i]).convert('RGB')
                    aux_img = self.aux_processor(images=aux_img, return_tensors='pt')['pixel_values'].squeeze()
                    aux_imgs.append(aux_img)
                # padding
                for i in range(3 - len(aux_imgs)):
                    aux_imgs.append(torch.zeros((3, self.aux_size, self.aux_size)))
                aux_imgs = torch.stack(aux_imgs, dim=0)
                assert len(aux_imgs) == 3

                if self.rcnn_img_path is not None:
                    rcnn_imgs = []
                    rcnn_img_paths = []
                    if imgid in self.data_dict['rcnn_imgs']:
                        rcnn_img_paths = self.data_dict['rcnn_imgs'][imgid]
                        rcnn_img_paths = [os.path.join(self.rcnn_img_path, path) for path in rcnn_img_paths]
                    # select 3 img
                    for i in range(min(3, len(rcnn_img_paths))):
                        rcnn_img = Image.open(rcnn_img_paths[i]).convert('RGB')
                        rcnn_img = self.rcnn_processor(images=rcnn_img, return_tensors='pt')['pixel_values'].squeeze()
                        rcnn_imgs.append(rcnn_img)
                    # padding
                    for i in range(3 - len(rcnn_imgs)):
                        rcnn_imgs.append(torch.zeros((3, self.rcnn_size, self.rcnn_size)))
                    rcnn_imgs = torch.stack(rcnn_imgs, dim=0)
                    assert len(rcnn_imgs) == 3
                    # if self.write_path is not None and self.mode == 'test' and self.do_test:
                    #     return input_ids, token_type_ids, attention_mask, torch.tensor(
                    #         re_label), image, aux_imgs, rcnn_imgs, extend_word_list, imgid
                    # else:
                    #     return input_ids, token_type_ids, attention_mask, torch.tensor(
                    #         re_label), image, aux_imgs, rcnn_imgs
                    if self.write_path is not None and self.mode == 'test' and self.do_test:
                        return input_ids, token_type_ids, attention_mask, torch.tensor(
                            re_label), image, aux_imgs, rcnn_imgs, extend_word_list, imgid
                    else:
                        return input_ids, token_type_ids, attention_mask, torch.tensor(
                            re_label), image, aux_imgs, rcnn_imgs

                return input_ids, token_type_ids, attention_mask, torch.tensor(re_label), image, aux_imgs


        return input_ids, token_type_ids, attention_mask, torch.tensor(re_label), image

    def initialize(self, query_budget: int):
        # query_budget is the number of labels been queried each round
        # random initialization for first batch of labels
        self.labeled_ids = self.unlabeled_ids[:query_budget]
        self.unlabeled_ids = [id for id in self.all_ids if id not in self.labeled_ids]

    def query_for_label(self, queried_ids: list):
        # queried_ids are generated from query strategy

        self.labeled_ids += queried_ids
        self.unlabeled_ids = [str(id) for id in self.all_ids if str(id) not in self.labeled_ids]
        assert len(self.labeled_ids) + len(self.unlabeled_ids) == len(self.all_ids)

    def query_for_label_temp(self, queried_ids: list):
        # queried_ids are generated from query strategy

        self.labeled_ids += queried_ids
        self.unlabeled_ids = [str(id) for id in self.all_ids if str(id) not in self.labeled_ids]
        assert len(self.labeled_ids) + len(self.unlabeled_ids) == len(self.all_ids)

    def query(self):
        # prepare unlabeled data index for label querying
        self.mode = "query"
        print("dataset for querying")
        self.sample_ids = self.unlabeled_ids

    def train(self):
        # prepare labeled queried data index for model training
        self.mode = "train"
        print("dataset for training")
        self.sample_ids = self.labeled_ids

