import re
import torch
from torch.utils.data import Dataset
import json
import os
from torch.utils.data import DataLoader
from bert_utils import *

class SimilarLawDataSet(Dataset):
    def __init__(self, candidates_path, query_path, label_path, tokenizer, max_len,data_type):
        super(SimilarLawDataSet, self).__init__()
        self.data_type = data_type
        self.candidates_path = candidates_path
        self.query_path = query_path
        self.label_path = label_path
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.querys = read_query(self.query_path,self.data_type)  # query id : query
        self.data_pair_list = read_pairs(self.candidates_path,self.label_path,self.data_type)

    def pad_seq(self, ids_list, types_list):
        batch_len = self.max_len  # max([len(ids) for ids in ids_list])
        new_ids_list, new_types_list, new_masks_list = [], [], []
        for ids, types in zip(ids_list, types_list):
            masks = [1] * len(ids) + [0] * (batch_len - len(ids))
            types += [0] * (batch_len - len(ids))
            ids += [0] * (batch_len - len(ids))
            new_ids_list.append(ids)
            new_types_list.append(types)
            new_masks_list.append(masks)
        return new_ids_list, new_types_list, new_masks_list

    def __len__(self):
        return len(self.data_pair_list)

    def __getitem__(self, item):
        q_idx, d_idx, label = self.data_pair_list[item]
        q = self.querys[q_idx]
        d = get_doc(self.candidates_path, q_idx, d_idx,self.data_type)
        q=q[:254]
        d=d[:254]

        ids, type_ids = [], []
        q_crime_tokens, d_crime_tokens = self.tokenizer.tokenize(q), self.tokenizer.tokenize(d)
        crime_tokens = ['[CLS]'] + q_crime_tokens + ['[SEP]'] + d_crime_tokens
        crime_ids = self.tokenizer.convert_tokens_to_ids(crime_tokens)
        crime_types = [0] * (len(q_crime_tokens) + 2) + [1] * (len(d_crime_tokens))
        ids.append(crime_ids)
        type_ids.append(crime_types)

        ids, type_ids, masks = self.pad_seq(ids, type_ids)

        label = torch.Tensor([label])

        return {'q_id':q_idx,
                'd_id':d_idx,
                'input_ids': torch.LongTensor(ids),
                'token_type_ids': torch.LongTensor(type_ids),
                'attention_mask': torch.LongTensor(masks),
                'label': label}

class SimilarLawTestDataSet(Dataset):
    def __init__(self, candidates_path, query_path,  tokenizer, max_len,data_type):
        super(SimilarLawTestDataSet, self).__init__()
        self.candidates_path = candidates_path
        self.query_path = query_path
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.data_type=data_type
        self.querys = read_query(self.query_path,self.data_type)  # query id : query
        self.test_data = self.read_test_data()
        self.data_pair_list = self.gen_data_pair()

    def read_test_data(self):
        if self.data_type=='elam':
            test_path='../elam_data/elam_test_top50.json'
        elif self.data_type=='Lecard':
            test_path='../LeCaRD-main/data/prediction/test_top100.json'
        else:
            print('data error')
            exit()
        with open(test_path, mode='r', encoding='utf-8')as f:
            js_dict = json.load(f)
            for k in js_dict.keys():
                js_dict[k] = [str(v) for v in js_dict[k]]
        return js_dict  # query id, can ids

    def gen_data_pair(self):
        data_pair_list = []
        for k in self.test_data.keys():
            query = self.querys[k]
            for v in self.test_data[k]:
                doc = get_doc(self.candidates_path, k, v,self.data_type)
                data_pair_list.append((k, v, query, doc))
        return data_pair_list  # query, doc

    def pad_seq(self, ids_list, types_list):
        batch_len = 512  # max([len(ids) for ids in ids_list])
        new_ids_list, new_types_list, new_masks_list = [], [], []
        for ids, types in zip(ids_list, types_list):
            masks = [1] * len(ids) + [0] * (batch_len - len(ids))
            types += [0] * (batch_len - len(ids))
            ids += [0] * (batch_len - len(ids))
            new_ids_list.append(ids)
            new_types_list.append(types)
            new_masks_list.append(masks)
        return new_ids_list, new_types_list, new_masks_list

    def __len__(self):
        return len(self.data_pair_list)

    def __getitem__(self, item):
        q_id, d_id, q, d = self.data_pair_list[item]

        q = q[:254]
        d = d[:254]

        ids, type_ids = [], []
        q_crime_tokens, d_crime_tokens = self.tokenizer.tokenize(q), self.tokenizer.tokenize(d)
        crime_tokens = ['[CLS]'] + q_crime_tokens + ['[SEP]'] + d_crime_tokens
        crime_ids = self.tokenizer.convert_tokens_to_ids(crime_tokens)
        crime_types = [0] * (len(q_crime_tokens) + 2) + [1] * (len(d_crime_tokens))
        ids.append(crime_ids)
        type_ids.append(crime_types)
        ids, type_ids, masks = self.pad_seq(ids, type_ids)

        return {'q_id':q_id,
                'd_id':d_id,
                'input_ids': torch.LongTensor(ids),
                'token_type_ids': torch.LongTensor(type_ids),
                'attention_mask': torch.LongTensor(masks)}


def get_doc(candidates_path, text1_idx, text2_idx,data_type):
    file_path = os.path.join(candidates_path, text1_idx, text2_idx+'.json')
    doc = ''
    with open(file_path, mode='r', encoding='utf-8')as f:
        js_dict = json.load(f)
        if data_type=='elam':
            doc += "".join(js_dict['doc'])
        elif data_type=='Lecard':
            doc += js_dict['ajjbqk']

    return doc


def read_query(query_path,data_type):
    querys = {}
    with open(query_path, mode='r', encoding='utf-8')as f:
        lines = f.readlines()
        for line in lines:
            data = json.loads(line)
            if data_type=='elam':
                ridx = data['id']
                query = "".join(data['q'])
            elif data_type=='Lecard':
                ridx = data['ridx']
                query = data['q']
            else:
                print("error")
                exit()

            querys[str(ridx)] = query
    return querys


def read_pairs(candidates_path,label_path,data_type):
    data_pair_list = []
    folders = os.listdir(candidates_path)
    neg_labels = {}
    with open(label_path, mode='r', encoding='utf-8')as f:
        line = f.readlines()[0]
        pos_labels = json.loads(line)
        for k in pos_labels.keys():
            for file in pos_labels[k]:
                if data_type=='elam':
                    data_pair_list.append((k, file, pos_labels[k][file] / 2))
                elif data_type=='Lecard':
                    data_pair_list.append((k, file, pos_labels[k][file] / 3))
                else:
                    print('error')
                    exit()
            pos_labels[k] = [str(i) for i in pos_labels[k]]

    for folder in folders:
        pos = pos_labels[folder]
        neg_labels[folder] = []
        folder_path = os.path.join(candidates_path, folder)
        files = os.listdir(folder_path)
        for filename in files:
            filename = filename.replace('.json', '')
            if filename not in pos:
                neg_labels[folder].append(filename)

    for key in neg_labels.keys():
        for filename in neg_labels[key]:
            data_pair_list.append((key, filename, 0))

    return data_pair_list



