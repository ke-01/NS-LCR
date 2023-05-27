import argparse
from tqdm import tqdm
from transformers import BertTokenizer, BertModel
import torch.nn as nn
import torch
import os
import json
import re
import numpy as np
from sentence_transformers.util import cos_sim
from sentence_transformers import SentenceTransformer as SBert

parser = argparse.ArgumentParser()
parser.add_argument('--cuda_pos', type=str, default='0', help='which GPU to use')
parser.add_argument('--seed', type=int, default=42, help='max length of each case')
args = parser.parse_args()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
device = torch.device('cuda:'+args.cuda_pos) if torch.cuda.is_available() else torch.device('cpu')
print(device)

model = SBert('/pretrain_model/paraphrase-multilingual-MiniLM-L12-v2')
model.to(device)

if __name__ == '__main__':

    candidates_path='../LeCaRD-main/data/candidates/all_doc'
    embeddings_path = '../LeCaRD-main/data/candidates/all_doc_sent_emb'

    files = os.listdir(candidates_path)
    cnt=1
    for filename in files:
        filename_t = filename.replace('.json', '')
        with open(candidates_path + '/' + filename, mode='r', encoding='utf-8')as f:
            js_dict = json.load(f)
        data_extract_npy=embeddings_path+'/'+filename_t+'.npy'
        text = js_dict['ajjbqk'].strip()
        sentences = re.split('。|！|\!|？|\?', text)
        sentences = [x.strip() for x in sentences if x.strip() != '']  # 去掉空字符串
        embeddings = model.encode(sentences)
        np.save(data_extract_npy, embeddings)
        print(u'输出路径：%s' % data_extract_npy)

