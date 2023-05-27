import torch
import json
import os
from transformers import AutoModel, AutoTokenizer,AutoConfig, AdamW, get_linear_schedule_with_warmup
from pytorch_pretrained_bert import BertModel
from transformers import BertTokenizer
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from bert_utils import *
from bert_model import *
from bert_dataset import SimilarLawTestDataSet
import argparse
from argparse import Namespace

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
device = torch.device('cuda:'+'1') if torch.cuda.is_available() else torch.device('cpu')
parser = argparse.ArgumentParser(description="Help info:")
parser.add_argument('--model_path', type=str, default="/pretrain_model/bert_legal_criminal", help='model for bert')
parser.add_argument('--query_path', type=str,  default='../LeCaRD-main/data/query/test_query.json', help='query_path')
parser.add_argument('--candidates_path', type=str,  default='../LeCaRD-main/data/candidates/test', help='candidates_path')
parser.add_argument('--save_path', type=str,  default='../LeCaRD-main/data/prediction/shaobert_L_res.json', help='save path')
parser.add_argument('--best_model_path', type=str,  default='Lecard_shaobert_best_model', help='best_model_path')
parser.add_argument('--max_len', type=int, default=512, help='maxlen for input')
parser.add_argument('--batch_size', type=int, default=1, help='batch_size for train')
parser.add_argument('--data_type', type=str, default="Lecard",choices= ['Lecard','elam'], help='dataset choice')

args = parser.parse_args()

# elam test
if args.data_type == 'elam':
    args.candidates_path = '../elam_data/elam_candidates/test'
    args.query_path = '../elam_data/elam_test_query.json'
    args.save_path = '../LeCaRD-main/data/prediction/shaobert_E_res.json'
    args.best_model_path= 'elam_shaobert_best_model'
    print('test elam')

print(args.query_path)
model = torch.load(args.best_model_path)
model.to(device)
tokenizer = BertTokenizer.from_pretrained(args.model_path)

dataset = SimilarLawTestDataSet(args.candidates_path, args.query_path, tokenizer, args.max_len,args.data_type)
test_data_loader = DataLoader(dataset, batch_size=args.batch_size,collate_fn=select_collate_test)

def _move_to_device(batch, device):
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(device)
    return batch

def test():
    model.eval()
    with torch.no_grad():
        batch_iterator = tqdm(test_data_loader, desc='testing...',ncols=100)
        ans_dict = {}
        for step, batch in enumerate(batch_iterator):
            batch = _move_to_device(batch, device)
            query_id = batch['q_id']
            doc_id = batch['d_id']

            score = model(batch)
            if query_id not in ans_dict.keys():
                ans_dict[query_id] = {}
            ans_dict[query_id][doc_id] = score.item()

        for k in ans_dict.keys():
            ans_dict[k] = sorted(ans_dict[k].items(), key=lambda x: x[1], reverse=True)
        for k in ans_dict.keys():
            ans_dict[k] = [int(did) for did, _ in ans_dict[k]]

        with open(args.save_path, mode='w', encoding='utf-8') as f:
            f.write(json.dumps(ans_dict, ensure_ascii=False))
    print("test finish")

if __name__ == '__main__':
    test()