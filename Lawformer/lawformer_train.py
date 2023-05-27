from transformers import BertTokenizer
from transformers import AutoModel, AutoTokenizer, AutoConfig, AdamW, get_linear_schedule_with_warmup
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
from lawformer_dataset import *
from lawformer_utils import *
from lawformer_model import *
import logging, os, random, json
import argparse
from argparse import Namespace

parser = argparse.ArgumentParser(description="Help info:")
parser.add_argument('--model_path', type=str, default="/pretrain_model/lawformer", help='model for bert')
parser.add_argument('--label_path', type=str, default='../LeCaRD-main/data/label/train_label.json', help='Label file path.')
parser.add_argument('--query_path', type=str,  default='../LeCaRD-main/data/query/train_query.json', help='query_path')
parser.add_argument('--candidates_path', type=str,  default='../LeCaRD-main/data/candidates/train', help='candidates_path')
parser.add_argument('--eval_label_path', type=str, default='../LeCaRD-main/data/label/eval_label.json', help='Label file path.')
parser.add_argument('--eval_query_path', type=str,  default='../LeCaRD-main/data/query/eval_query.json', help='query_path')
parser.add_argument('--eval_candidates_path', type=str,  default='../LeCaRD-main/data/candidates/eval', help='candidates_path')
parser.add_argument('--all_ids', type=str,  default='../LeCaRD-main/data/prediction/combined_top100.json', help='all_ids')
parser.add_argument('--max_len', type=int, default=4096, help='maxlen for input')
parser.add_argument('--epoch', type=int, default=20, help='epoch for train')
parser.add_argument('--batch_size', type=int, default=2, help='batch_size for train')
parser.add_argument('--learning_rate', type=float, default=3e-6, help='learning_rate')
parser.add_argument('--gradient_accumulation_steps', type=int, default=10, help='gradient_accumulation_steps')
parser.add_argument('--data_type', type=str, default="Lecard",choices= ['Lecard','elam'], help='dataset choice')

args = parser.parse_args()

if args.data_type=='elam':
    args.label_path = '../elam_data/elam_train_label.json'
    args.query_path='../elam_data/elam_train_query.json'
    args.candidates_path= '../elam_data/elam_candidates/train'
    args.eval_label_path = '../elam_data/elam_eval_label.json'
    args.eval_query_path = '../elam_data/elam_eval_query.json'
    args.eval_candidates_path = '../elam_data/elam_candidates/eval'
    args.all_ids='../elam_data/elam_ids_all.json'
    args.max_len = 512
    args.batch_size=16
    args.learning_rate=2e-5

os.environ["CUDA_VISIBLE_DEVICES"] = '1'
device = torch.device('cuda:'+'0') if torch.cuda.is_available() else torch.device('cpu')

def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.

seed_torch()

print(args.data_type)
# model
tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")
model = FirstModel()
model.cuda()

optimizer = Adam(model.parameters(), lr=args.learning_rate)

# dataset
train_dataset = SimilarLawDataSet(args.candidates_path, args.query_path, args.label_path, tokenizer, args.max_len, args.data_type)
eval_dataset = SimilarLawDataSet(args.eval_candidates_path, args.eval_query_path, args.eval_label_path,tokenizer, args.max_len,args.data_type)

train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=select_collate)
eval_data_loader = DataLoader(eval_dataset, batch_size=1,collate_fn=select_collate_test)

# loss
criterion = MSELoss()

# for eval
with open(args.eval_label_path, 'r') as f:
    avglist = json.load(f)
with open(args.all_ids, 'r') as f:
    combdic = json.load(f)

def _move_to_device(batch, device):
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(device)
    return batch

def ndcg1(ranks, gt_ranks, K):
    dcg_value = 0.
    idcg_value = 0.

    sranks = sorted(gt_ranks, reverse=True)

    for i in range(0,K):
        logi = math.log(i+2,2)
        dcg_value += ranks[i] / logi
        idcg_value += sranks[i] / logi

    return dcg_value/idcg_value

def evaluate():
    model.eval()
    with torch.no_grad():
        batch_iterator = tqdm(eval_data_loader, desc='evaluating',ncols=100)
        ans_dict = {}
        for step, batch in enumerate(batch_iterator):
            batch = _move_to_device(batch, device)
            query_id = batch['q_id']
            doc_id = batch['d_id']

            score= model(batch)

            if query_id not in ans_dict.keys():
                ans_dict[query_id]={}
            ans_dict[query_id][doc_id]=score.item()

        for k in ans_dict.keys():
            ans_dict[k] = sorted(ans_dict[k].items(), key=lambda x: x[1], reverse=True)
        for k in ans_dict.keys():
            ans_dict[k] = [int(did) for did, _ in ans_dict[k]]
        dics = ans_dict
        sndcg = 0.0
        for key in ans_dict.keys():
            rawranks = []
            for i in dics[key]:
                if str(i) in list(avglist[key])[:30]:
                    rawranks.append(avglist[key][str(i)])
                else:
                    rawranks.append(0)
            ranks = rawranks + [0] * (30 - len(rawranks))
            if sum(ranks) != 0:
                sndcg += ndcg1(ranks, list(avglist[key].values()), 30)
        ndcg_30 = sndcg / len(ans_dict)

        print("ndcg_30:%f " % (ndcg_30))

        torch.cuda.empty_cache()
        model.train()
    return ndcg_30

def train():
    best_acc = 0.0
    for e in range(args.epoch):
        print(f"epoch:{e}")
        step_loss = 0.0
        epoch_loss = 0.0
        batch_iterator = tqdm(train_data_loader, desc='training',ncols=100)
        total_step = len(train_data_loader)
        model.train()
        model.zero_grad()
        for step, batch in enumerate(batch_iterator):
            batch = _move_to_device(batch, device)
            label = batch['label'].to(device)

            score= model(batch)
            loss = criterion(score, label)
            loss.backward()
            optimizer.step()

            model.zero_grad()

            step_loss += loss.item()
            epoch_loss += loss.item()

            if (step+1) % args.gradient_accumulation_steps == 0:
                print("avg loss:", step_loss/args.gradient_accumulation_steps)
                step_loss = 0.0

        epoch_loss /= total_step
        Loss_list.append(epoch_loss)
        print("epoch %d loss: %f" % (e, epoch_loss))
        ndcg_30 = evaluate()
        if ndcg_30 > best_acc:
            best_acc = ndcg_30
            print("best_ndcg_30: %f"%(best_acc))
            torch.save(model, args.data_type+'_lawformer_best_model')
    print("train finish")
if __name__ == '__main__':
    train()