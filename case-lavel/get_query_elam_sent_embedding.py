import sys
sys.path.append("..")
import argparse
from tqdm import tqdm
from transformers import BertTokenizer, BertModel
# from utils.snippets import *
import torch.nn as nn
import torch
import os
import json
import re
import numpy as np
from sentence_transformers.util import cos_sim
from sentence_transformers import SentenceTransformer as SBert

parser = argparse.ArgumentParser()
parser.add_argument('--cuda_pos', type=str, default='1', help='which GPU to use')
parser.add_argument('--seed', type=int, default=42, help='max length of each case')
args = parser.parse_args()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

device = torch.device('cuda:'+args.cuda_pos) if torch.cuda.is_available() else torch.device('cpu')
print(device)

model = SBert('/pretrain_model/paraphrase-multilingual-MiniLM-L12-v2')
model.to(device)

def load_data(filename):
    """加载数据
    返回：[texts]
    """
    D={}
    # D = []
    with open(filename, mode='r', encoding='utf-8')as f:
        lines = f.readlines()
        for line in lines:
            data = json.loads(line)
            ridx = data['id']
            query = data['q']
            D[str(ridx)] = query

    return D

def convert(data):
    """转换所有样本
    """
    embeddings = {}
    model.to(device)
    for key, value in data.items():
        texts=value
        idx=key
        outputs_a = model.encode(texts)
        embeddings[idx]=outputs_a
    return embeddings

if __name__ == '__main__':

    query_path='../elam_data/elam_query.json'
    query_npy='../elam_data/elam_query_sent_emb.npy'

    data = load_data(query_path)
    print(len(data))
    embeddings = convert(data)
    print(len(embeddings))
    np.save(query_npy, embeddings)
    print(u'输出路径：%s' % query_npy)
