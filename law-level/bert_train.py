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
from bert_dataset import *
from bert_utils import *
from bert_model import *
import logging, os, random, json
import argparse
from argparse import Namespace

parser = argparse.ArgumentParser(description="Help info:")
parser.add_argument('--model_path', type=str, default="/pretrain_model/bert_legal_criminal", help='model for bert')
parser.add_argument('--candidates_path', type=str,  default='../LeCaRD-main/data/candidates_for_symbolic', help='candidates_path')
parser.add_argument('--logic_path', type=str,  default='./law_article_1_451.json', help='candidates_path')
parser.add_argument('--ids2group_path', type=str,  default='./ids2group.json', help='candidates_path')
parser.add_argument('--max_len', type=int, default=512, help='maxlen for input')
parser.add_argument('--epoch', type=int, default=20, help='epoch for train')
parser.add_argument('--batch_size', type=int, default=24, help='batch_size for train')
parser.add_argument('--learning_rate', type=float, default=2e-5, help='learning_rate')
parser.add_argument('--gradient_accumulation_steps', type=int, default=10, help='gradient_accumulation_steps')
parser.add_argument('--data_type', type=str, default="Lecard",choices= ['Lecard','elam'], help='dataset choice')

args = parser.parse_args()

device = torch.device('cuda:'+'0') if torch.cuda.is_available() else torch.device('cpu')

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
model = FirstModel()
model.to(device)
optimizer = Adam(model.parameters(), lr=args.learning_rate)

# dataset
dataset = SimilarLawDataSet(args.candidates_path, args.logic_path, args.ids2group_path, tokenizer, args.max_len, args.data_type)

train_size = int(0.8*len(dataset))
eval_size = len(dataset) - train_size
train_dataset, eval_dataset = torch.utils.data.random_split(dataset, [train_size, eval_size])

train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=select_collate)
eval_data_loader = DataLoader(eval_dataset, batch_size=args.batch_size,collate_fn=select_collate)
print('--data finish--')

# loss
criterion = MSELoss()

def _move_to_device(batch, device):
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(device)
    return batch

def evaluate():
    model.eval()
    with torch.no_grad():
        batch_iterator = tqdm(eval_data_loader, desc='evaluating',ncols=100)
        num = 0
        corr = 0
        for step, batch in enumerate(batch_iterator):
            batch = _move_to_device(batch, device)
            label = batch['label']

            score= model(batch)
            zero = torch.zeros_like(score)
            one = torch.ones_like(score)
            score = torch.where(score >= 0.5, one, score)
            score = torch.where(score < 0.5, zero, score)
            batch_corr = (label ==score).sum()
            corr += batch_corr.item()
            num += len(score)

        print("acc: %f" % (corr / num))
        torch.cuda.empty_cache()
        model.train()
        return corr/num

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
        acc = evaluate()
        if acc > best_acc:
            best_acc = acc
            print("best_ndcg_30: %f"%(best_acc))
            torch.save(model, 'predicate_best_model')
    print("train finish")


if __name__ == '__main__':
    train()