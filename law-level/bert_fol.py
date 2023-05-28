from transformers import BertTokenizer
from transformers import AutoModel, AutoTokenizer,AutoConfig, AdamW, get_linear_schedule_with_warmup
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
from torch.optim import Adam
import os
from torch.nn import MSELoss
import random
import numpy as np
import math
from dataset_fol_bert import *
from bert_utils import *
from bert_model import *
import logging, os, random, json
import argparse
from argparse import Namespace

parser = argparse.ArgumentParser(description="Help info:")
parser.add_argument('--model_path', type=str, default="/pretrain_model/bert_legal_criminal", help='model for bert')
parser.add_argument('--label_path', type=str, default='../LeCaRD-main/data/label/test_label.json', help='Label file path.')
parser.add_argument('--query_path', type=str,  default='../LeCaRD-main/data/query/test_query.json', help='query_path')
parser.add_argument('--candidates_path', type=str,  default='../LeCaRD-main/data/candidates/test', help='candidates_path')
parser.add_argument('--logic_path', type=str,  default='./law_article_1_451.json', help='candidates_path')
parser.add_argument('--save_path', type=str,  default='../LeCaRD-main/data/prediction/Luka_fol_L_res.json', help='save path')
parser.add_argument('--max_len', type=int, default=512, help='maxlen for input')
parser.add_argument('--batch_size', type=int, default=1, help='batch_size for train')
parser.add_argument('--data_type', type=str, default="Lecard",choices= ['Lecard','elam'], help='dataset choice')
parser.add_argument('--best_model_path', type=str,  default='./predicate_best_model', help='best_model_path')
args = parser.parse_args()

if args.data_type=='elam':
    args.label_path = '../elam_data/elam_test_label.json'
    args.query_path='../elam_data/elam_test_query.json'
    args.candidates_path= '../elam_data/elam_candidates/test'
    args.save_path = '../LeCaRD-main/data/prediction/Luka_fol_E_res.json'
    args.all_ids='../elam_data/elam_ids_all.json'

device = torch.device('cuda:'+'1') if torch.cuda.is_available() else torch.device('cpu')
print(device)
def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.

seed_torch()

# model
tokenizer = BertTokenizer.from_pretrained(args.model_path)
model = torch.load(args.best_model_path)
model.to(device)
# symbolic model
symbolic_model = Logic()

# dataset
dataset = SimilarLawDataSet(args.candidates_path, args.logic_path, args.query_path,args.label_path, tokenizer, args.max_len, args.data_type)
print(len(dataset))
eval_data_loader = DataLoader(dataset, batch_size=args.batch_size,collate_fn=select_collate_fol)

with open('./law_article_1_451.json', 'r', encoding='utf-8') as f:
    file = json.load(f)
print('--data finish--')

def _move_to_device(batch, device):
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(device)
    return batch

def get_fol_res():
    model.eval()
    ans_dict = {}
    with torch.no_grad():
        batch_iterator = tqdm(eval_data_loader, desc='evaluating',ncols=100)
        for step, batch in enumerate(batch_iterator):
            query_id = batch['q_id']
            doc_id = batch['d_id']
            doc_id = doc_id.replace('.json', '')
            rules_t = []
            for i in range(len(batch['input_ids'])):
                # for each rule
                p_mapping = batch['mapping'][i]
                rules = batch['in_laws'][i]

                for j, key in zip(range(len(batch['input_ids'][i])), p_mapping.keys()):
                    # for each perdicate
                    sample = {}
                    sample['input_ids'], sample['attention_mask'], sample['token_type_ids'] = \
                        batch['input_ids'][i][j].to(device), \
                        batch['attention_mask'][i][j].to(device), \
                        batch['token_type_ids'][i][j].to(device)
                    score = model(sample)
                    p_mapping[key] = score.item()

                for rule in rules:
                    for key, val in p_mapping.items():
                        rule = rule.replace(key, str(val))
                    rules_t.append(rule)
            score_all = symbolic_model(rules_t)
            if query_id not in ans_dict.keys():
                ans_dict[query_id] = {}
            ans_dict[query_id][doc_id] = score_all

        for k in ans_dict.keys():
            ans_dict[k] = sorted(ans_dict[k].items(), key=lambda x: x[1], reverse=True)
        for k in ans_dict.keys():
            ans_dict[k] = [int(did) for did, _ in ans_dict[k]]

        with open(args.save_path, mode='w', encoding='utf-8') as f:
            f.write(json.dumps(ans_dict, ensure_ascii=False))
        return 0

if __name__ == '__main__':
    get_fol_res()