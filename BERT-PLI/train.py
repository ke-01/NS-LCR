from transformers import BertTokenizer
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from dataset import SimilarLawDataSet
from tqdm import tqdm
from torch.optim import Adam
import os
from bertpli import BertPli
import random
import numpy as np
import argparse
import json
import math
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'
device = torch.device('cuda:'+'0') if torch.cuda.is_available() else torch.device('cpu')

parser = argparse.ArgumentParser(description="Help info:")
parser.add_argument('--model_path', type=str, default="/pretrain_model/bert_legal_criminal", help='model for bert')
parser.add_argument('--label_path', type=str, default='../LeCaRD-main/data/label/train_label.json', help='Label file path.')
parser.add_argument('--query_path', type=str,  default='../LeCaRD-main/data/query/train_query.json', help='query_path')
parser.add_argument('--candidates_path', type=str,  default='../LeCaRD-main/data/candidates/train', help='candidates_path')
parser.add_argument('--eval_label_path', type=str, default='../LeCaRD-main/data/label/eval_label.json', help='Label file path.')
parser.add_argument('--eval_query_path', type=str,  default='../LeCaRD-main/data/query/eval_query.json', help='query_path')
parser.add_argument('--eval_candidates_path', type=str,  default='../LeCaRD-main/data/candidates/eval', help='candidates_path')
parser.add_argument('--all_ids', type=str,  default='../LeCaRD-main/data/prediction/combined_top100.json', help='all_ids')
parser.add_argument('--max_len', type=int, default=512, help='maxlen for input')
parser.add_argument('--epoch', type=int, default=20, help='epoch for train')
parser.add_argument('--max_para_q', type=int, default=2, help='max_para_q')
parser.add_argument('--max_para_d', type=int, default=8, help='max_para_d')
parser.add_argument('--para_max_len', type=int, default=255, help='epoch for train')
parser.add_argument('--batch_size', type=int, default=2, help='batch_size for train')
parser.add_argument('--learning_rate', type=float, default=3e-6, help='learning_rate')
parser.add_argument('--gradient_accumulation_steps', type=int, default=10, help='gradient_accumulation_steps')
parser.add_argument('--data_type', type=str, default="Lecard",choices= ['Lecard','elam'], help='dataset choice')

args = parser.parse_args()

if args.data_type=='elam':
    args.label_path = '../elam_data/elam_train_label.json'
    args.query_path = '../elam_data/elam_train_query.json'
    args.candidates_path = '../elam_data/elam_candidates/train'
    args.eval_label_path = '../elam_data/elam_eval_label.json'
    args.eval_query_path = '../elam_data/elam_eval_query.json'
    args.eval_candidates_path = '../elam_data/elam_candidates/eval'
    args.all_ids = '../elam_data/elam_ids_all.json'
    args.max_para_d=4
    print('train elam')

def seed_torch(seed=42):
    print(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.

seed_torch()

tokenizer = BertTokenizer.from_pretrained(args.model_path)
criterion = nn.MSELoss()

model = BertPli(model_path=args.model_path, max_para_q=args.max_para_q, max_para_d=args.max_para_d, max_len=args.max_len, criterion=criterion,data_type=args.data_type)


if args.data_type=='elam':
    model.to(device)
else:   # Lecard
    model.cuda()
    if torch.cuda.device_count() > 1:
        print(f"GPU数：{torch.cuda.device_count()}")
        model = nn.DataParallel(model)

optimizer = Adam(model.parameters(), lr=args.learning_rate)
train_dataset = SimilarLawDataSet(args.candidates_path, args.query_path, args.label_path, tokenizer, args.max_para_q, args.max_para_d, args.para_max_len, args.max_len,args.data_type)
eval_dataset = SimilarLawDataSet(args.eval_candidates_path, args.eval_query_path, args.eval_label_path, tokenizer, args.max_para_q, args.max_para_d, args.para_max_len, args.max_len,args.data_type)

train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
eval_data_loader = DataLoader(eval_dataset, batch_size=1, shuffle=False)

# for eval
with open(args.eval_label_path, 'r') as f:
    avglist = json.load(f)
with open(args.all_ids, 'r') as f:
    combdic = json.load(f)

def ndcg(ranks,K):
    dcg_value = 0.
    idcg_value = 0.
    log_ki = []

    sranks = sorted(ranks, reverse=True)

    for i in range(0,K):
        logi = math.log(i+2,2)
        dcg_value += ranks[i] / logi
        idcg_value += sranks[i] / logi

    '''print log_ki'''
    # print ("DCG value is " + str(dcg_value))
    # print ("iDCG value is " + str(idcg_value))

    return dcg_value/idcg_value
def ndcg1(ranks, gt_ranks, K):
    dcg_value = 0.
    idcg_value = 0.
    # log_ki = []

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
        for step, (input_ids, token_type_ids, attention_mask, label,query_id, doc_id) in enumerate(batch_iterator):
            input_ids = input_ids.cuda()
            token_type_ids = token_type_ids.cuda()
            attention_mask = attention_mask.cuda()
            label = label.cuda()
            query_id=query_id[0]
            doc_id=doc_id[0]

            data = {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'token_type_ids': token_type_ids
            }
            score, loss = model(data=data, label=label, mode='eval')

            if query_id not in ans_dict.keys():
                ans_dict[query_id]={}
            ans_dict[query_id][doc_id]=score.item()

        for k in ans_dict.keys():
            ans_dict[k] = sorted(ans_dict[k].items(), key=lambda x: x[1], reverse=True)
        for k in ans_dict.keys():
            ans_dict[k] = [int(did) for did, _ in ans_dict[k]]
        dics=ans_dict
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
        ndcg_30=sndcg/len(ans_dict)

        print("ndcg_30:%f "%(ndcg_30))

        torch.cuda.empty_cache()
        model.train()
    print("evaluate finish")
    return ndcg_30


def train():
    best_val = 0.0
    for e in range(args.epoch):
        print(f"epoch:{e}")
        step_loss = 0.0
        epoch_loss = 0.0
        batch_iterator = tqdm(train_data_loader, desc='training',ncols=100)
        total_step = len(batch_iterator)
        print(total_step)
        model.train()
        model.zero_grad()
        for step, (input_ids, token_type_ids, attention_mask, label,q_id,d_id) in enumerate(batch_iterator):

            input_ids = input_ids.cuda()
            token_type_ids = token_type_ids.cuda()
            attention_mask = attention_mask.cuda()
            label = label.cuda()

            data = {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'token_type_ids': token_type_ids
            }

            loss = model(data=data, label=label, mode='train')
            loss = loss.mean()
            loss.backward()
            optimizer.step()

            model.zero_grad()

            step_loss += loss.item()
            epoch_loss += loss.item()


            if (step+1) % args.gradient_accumulation_steps == 0:
                print("avg loss:", step_loss/args.gradient_accumulation_steps)
                step_loss = 0.0

        epoch_loss /= total_step
        print("epoch %d loss: %f" % (e, epoch_loss))
        Loss_list.append(epoch_loss)
        ndcg30= evaluate()
        if ndcg30 > best_val:
            best_val = ndcg30
            torch.save(model, args.data_type+'_bertpli_best_model')


if __name__ == '__main__':
    train()