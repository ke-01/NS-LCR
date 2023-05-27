import torch
import json
import os
from dataset import SimilarLawTestDataSet
from transformers import BertTokenizer
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import json
import math

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

parser = argparse.ArgumentParser(description="Help info:")
parser.add_argument('--model_path', type=str, default="/pretrain_model/bert_legal_criminal", help='model for bert')
parser.add_argument('--query_path', type=str,  default='../LeCaRD-main/data/query/test_query.json', help='query_path')
parser.add_argument('--candidates_path', type=str,  default='../LeCaRD-main/data/candidates/test_laws', help='candidates_path')
parser.add_argument('--save_path', type=str,  default='../LeCaRD-main/data/prediction/bertpli_L_res.json', help='save path')
parser.add_argument('--best_model_path', type=str,  default='Lecard_bertpli_best_model', help='best_model_path')
parser.add_argument('--max_len', type=int, default=512, help='maxlen for input')
parser.add_argument('--batch_size', type=int, default=1, help='batch_size for train')
parser.add_argument('--max_para_q', type=int, default=2, help='max_para_q')
parser.add_argument('--max_para_d', type=int, default=8, help='max_para_d')
parser.add_argument('--para_max_len', type=int, default=255, help='epoch for train')
parser.add_argument('--gradient_accumulation_steps', type=int, default=10, help='gradient_accumulation_steps')
parser.add_argument('--data_type', type=str, default="elam",choices= ['Lecard','elam'], help='dataset choice')

args = parser.parse_args()
device = torch.device('cuda:'+'0') if torch.cuda.is_available() else torch.device('cpu')
# elam test
if args.data_type == 'elam':
    args.candidates_path = '../elam_data/elam_candidates/test'
    args.query_path = '../elam_data/elam_test_query.json'
    args.save_path = '../LeCaRD-main/data/prediction/bertpli_E_res.json'
    args.best_model_path= 'elam_bertpli_best_model'
    args.max_para_d=4


model = torch.load(args.best_model_path)

if args.data_type=='elam':
    model.to(device)
else:
    model.cuda()
    if torch.cuda.device_count() > 1:
        print(f"GPU数：{torch.cuda.device_count()}")
        model = nn.DataParallel(model)

tokenizer = BertTokenizer.from_pretrained(args.model_path)

dataset = SimilarLawTestDataSet(args.candidates_path, args.query_path, tokenizer,
                                args.max_para_q, args.max_para_d, args.para_max_len, args.max_len,args.data_type)
test_data_loader = DataLoader(dataset, batch_size=args.batch_size)


def test():
    model.eval()
    with torch.no_grad():
        batch_iterator = tqdm(test_data_loader, desc='testing...',ncols=100)
        ans_dict = {}
        for step, (query_id, doc_id, input_ids, token_type_ids, attention_mask) in enumerate(batch_iterator):
            input_ids = input_ids.to(device)
            token_type_ids = token_type_ids.to(device)
            attention_mask = attention_mask.to(device)
            label = torch.ones([1], dtype=torch.int64).to(device)
            query_id = query_id[0]
            doc_id = doc_id[0]
            data = {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'token_type_ids': token_type_ids
            }

            score, loss = model(data, label, 'test')
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