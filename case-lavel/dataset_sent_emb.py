import re
import torch
from torch.utils.data import Dataset
import json
import os
import torch.nn as nn
import torch
import jieba
import numpy as np
from bert_utils import *
import argparse

class SimilarLawDataSet(Dataset):
    def __init__(self, candidates_path, query_path, label_path, data_type,query_embedding_path,embedding_path):
        super(SimilarLawDataSet, self).__init__()
        self.query_embedding_path=query_embedding_path
        self.embedding_path=embedding_path
        self.data_type = data_type
        self.candidates_path = candidates_path
        self.query_path = query_path
        self.label_path = label_path
        self.querys,self.query_embeddings = read_query(self.query_path,self.data_type,self.query_embedding_path)  # query id : query
        self.data_pair_list = read_pairs(self.candidates_path,self.label_path,self.data_type)

    def __len__(self):
        return len(self.data_pair_list)

    def __getitem__(self, item):
        q_idx, d_idx, label = self.data_pair_list[item]
        q = self.querys[q_idx]
        q_embeddings=self.query_embeddings[q_idx]
        doc_embeddings = get_doc(self.candidates_path, q_idx, d_idx,self.data_type,self.embedding_path)
        label = torch.Tensor([label])

        return {'q_id':q_idx,
                'd_id':d_idx,
                'q_outputs': q_embeddings,
                'd_outputs': doc_embeddings,
                'label': label}

def get_doc(candidates_path, text1_idx, text2_idx,data_type,embeddings_path):
    file_path = os.path.join(candidates_path, text1_idx, text2_idx+'.json')
    with open(file_path, mode='r', encoding='utf-8')as f:
        if data_type=='elam':
            data_x = np.load(embeddings_path + '/' + text2_idx + '.npy', allow_pickle=True)
            doc_sent_embedding = data_x
        elif data_type=='Lecard':
            data_x = np.load(embeddings_path + '/' + text2_idx+'.npy', allow_pickle=True)
            doc_sent_embedding=data_x

    return doc_sent_embedding


def read_query(query_path,data_type,query_embedding_path):
    querys = {}
    querys_embedding={}
    query_x = np.load(query_embedding_path, allow_pickle=True)
    query_t = query_x.item()
    with open(query_path, mode='r', encoding='utf-8')as f:
        lines = f.readlines()
        for line in lines:
            data = json.loads(line)
            if data_type=='elam':
                ridx = data['id']
                query=data['q']
                query_emb = query_t[str(ridx)]
            elif data_type=='Lecard':
                ridx = data['ridx']
                query = re.split('。|！|\!|？|\?', data['q'].strip())
                query = [x.strip() for x in query if x.strip() != '']#去掉空字符串
                query_emb=query_t[str(ridx)]

            else:
                print("error")
                exit()

            querys[str(ridx)] = query
            querys_embedding[str(ridx)]=query_emb
    return querys,querys_embedding


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



