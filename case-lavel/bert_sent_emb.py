from transformers import BertTokenizer
from transformers import AutoModel, AutoTokenizer,AutoConfig, AdamW, get_linear_schedule_with_warmup
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.optim import Adam
import os
from torch.nn import MSELoss
import random
import numpy as np
import math
from dataset_sent_emb import *
from bert_utils import *
import logging, os, random, json
import argparse
from argparse import Namespace
from numpy.linalg import norm
from sentence_transformers.util import cos_sim
from sentence_transformers import SentenceTransformer as SBert

parser = argparse.ArgumentParser(description="Help info:")
parser.add_argument('--label_path', type=str, default='../LeCaRD-main/data/label/test_label.json', help='Label file path.')
parser.add_argument('--query_path', type=str,  default='../LeCaRD-main/data/query/test_query.json', help='query_path')
parser.add_argument('--candidates_path', type=str,  default='../LeCaRD-main/data/candidates/test', help='candidates_path')
parser.add_argument('--save_path', type=str,  default='../LeCaRD-main/data/prediction/sent_emb1_L_res.json', help='save path')
parser.add_argument('--data_type', type=str, default="Lecard",choices= ['Lecard','elam'], help='dataset choice')
parser.add_argument('--query_embedding_path', type=str,  default='../LeCaRD-main/data/query/query_sent_emb.npy', help='query_path')
parser.add_argument('--embeddings_path', type=str,  default='../LeCaRD-main/data/candidates/all_doc_sent_emb', help='candidates_path')

args = parser.parse_args()

if args.data_type=='elam':
    args.label_path = '../elam_data/elam_test_label.json'
    args.query_path='../elam_data/elam_test_query.json'
    args.candidates_path= '../elam_data/elam_candidates/test'
    args.query_embedding_path = '../elam_data/elam_query_sent_emb.npy'
    args.embeddings_path = '../elam_data/elam_candidates/all_doc_sent_emb'
    args.save_path = '../LeCaRD-main/data/prediction/sent_emb1_E_res.json'

device = torch.device('cuda:'+'1') if torch.cuda.is_available() else torch.device('cpu')

# dataset
train_dataset = SimilarLawDataSet(args.candidates_path, args.query_path, args.label_path, args.data_type,args.query_embedding_path,args.embeddings_path)
train_data_loader = DataLoader(train_dataset, batch_size=1, collate_fn=select_collate_struct)

def get_sent_res():
    print('len_train:{}'.format(len(train_data_loader)))
    ans_dict = {}
    with torch.no_grad():
        for batch in train_data_loader:
            torch.cuda.empty_cache()
            query_id = batch['q_id']
            doc_id = batch['d_id']
            X_embedding=batch['q_outputs']
            Y_embedding=batch['d_outputs']

            cosine_X_Y = cos_sim(X_embedding, Y_embedding)
            # dim=1:按行，largest:按从大到小
            if len(cosine_X_Y[0]) > 1:
                cosine_X_Y, indice = torch.topk(cosine_X_Y, 1, dim=1, largest=True, sorted=True, out=None)
            cosine_X_Y = cosine_X_Y.numpy()
            cosine_X_Y = cosine_X_Y[cosine_X_Y > 0]

            # 先开根号再相乘
            cosine_X_Y=np.power(cosine_X_Y,1/np.size(cosine_X_Y))#先开根号
            score=np.prod(cosine_X_Y)

            if query_id not in ans_dict.keys():
                ans_dict[query_id] = {}
            ans_dict[query_id][doc_id] = score.item()
        for k in ans_dict.keys():
            ans_dict[k] = sorted(ans_dict[k].items(), key=lambda x: x[1], reverse=True)
        for k in ans_dict.keys():
            ans_dict[k] = [int(did) for did, _ in ans_dict[k]]

        with open(args.save_path, mode='w', encoding='utf-8') as f:
            f.write(json.dumps(ans_dict, ensure_ascii=False))
    print("finish")

if __name__ == '__main__':
    get_sent_res()