import argparse
import pandas as pd
from metrics_Lecard import *
import json
import math
from argparse import Namespace

parser = argparse.ArgumentParser(description="Help info:")
parser.add_argument('--save_path', type=str,  default='../LeCaRD-main/data/prediction/fusion_res.json', help='save path')
parser.add_argument('--data_type', type=str,  default='Lecard', help='dataset')
parser.add_argument('--method', type=str,  default='bert', help='dataset')
args = parser.parse_args()

def reciprocal_rank_fusion_bert(results_lists, k=60,flag=0,K=1000):
    doc_ranks = {}  # inside there is dic with keys doc (string) and ranks (list of ints)
    for result in results_lists:

        for index, row in result.iterrows():
            # index starts at 0
            if row["_id"] not in doc_ranks:  # first insert list -> BERT
                doc_ranks[row["_id"]] = {}
                doc_ranks[row["_id"]]["ranks"] = []

                if math.cos(index*math.pi/K) < 0:
                    flag=0
                if flag == 1:
                    doc_ranks[row["_id"]]["rank_score"] = 5*math.cos(index*math.pi/K)+1
            doc_ranks[row["_id"]]["ranks"].append(index + 1)

    rrf_rank = []
    for doc_id, doc_rank in doc_ranks.items():
        rrf_score = 0
        is_bert=1
        for rank in doc_rank["ranks"]:
            # 对BERT中排名高的文档增加权重
            if "rank_score" in doc_ranks[doc_id].keys() and is_bert == 1:
                mutiples=doc_ranks[doc_id]["rank_score"]
                rrf_score += mutiples*1 / (k + rank)
                is_bert=0
            else:
                rrf_score += 1 / (k + rank)

        rrf_rank.append({"_id": doc_id, "_score": rrf_score})

    rrf_result = pd.DataFrame(data=rrf_rank)
    rrf_result = rrf_result.sort_values(by=['_score'], ascending=False)

    return rrf_result


def mm():
    bert_path=''
    fol_path=''
    semi_path=''
    if args.data_type=='Lecard':
        fol_path = '../LeCaRD-main/data/prediction/Luka_fol_L_res.json'
        semi_path = '../LeCaRD-main/data/prediction/sent_emb1_L_res.json'
        if args.method=='bert':
            bert_path='../LeCaRD-main/data/prediction/bert_L_res.json'
            flag= 0
            K=1000
        elif args.method=='shaobert':
            bert_path='../LeCaRD-main/data/prediction/shaobert_L_res.json'
            flag = 0
            K = 1000
        elif args.method=='bertpli':
            bert_path='../LeCaRD-main/data/prediction/bertpli_L_res.json'
            flag = 1
            K = 100.0
        elif args.method=='lawformer':
            bert_path='../LeCaRD-main/data/prediction/lawformer_L_res.json'
            flag = 1
            K = 4.0
    elif args.data_type=='ELAM':
        fol_path = '../LeCaRD-main/data/prediction/Luka_fol_E_res.json'
        semi_path = '../LeCaRD-main/data/prediction/sent_emb1_E_res.json'
        if args.method == 'bert':
            bert_path = '../LeCaRD-main/data/prediction/bert_E_res.json'
            flag = 0
            K = 2.0
        elif args.method == 'shaobert':
            bert_path = '../LeCaRD-main/data/prediction/shaobert_E_res.json'
            flag = 0
            K = 1000
        elif args.method == 'bertpli':
            bert_path = '../LeCaRD-main/data/prediction/bertpli_E_res.json'
            flag = 0
            K = 1000
        elif args.method == 'lawformer':
            bert_path = '../LeCaRD-main/data/prediction/lawformer_E_res.json'
            flag = 0
            K = 1000

    with open(bert_path, mode='r', encoding='utf-8')as f:
        ans_dict_bert = json.load(f)

    with open(fol_path, mode='r', encoding='utf-8')as f:
        ans_dict_fol = json.load(f)

    with open(semi_path, mode='r', encoding='utf-8')as f:
        ans_dict_semi = json.load(f)

    ans_dict = {}
    # all_result
    for query_id in ans_dict_bert:
        bert_result = ans_dict_bert[query_id]
        df_bert = pd.DataFrame({'_id': bert_result})
        semi_result = ans_dict_semi[query_id]
        df_semi = pd.DataFrame({'_id': semi_result})
        fol_result = ans_dict_fol[query_id]
        df_fol = pd.DataFrame({'_id': fol_result})
        result_lists = []
        result_lists.append(df_bert)
        result_lists.append(df_semi)
        result_lists.append(df_fol)
        rrf = reciprocal_rank_fusion_bert(result_lists,60,flag,K)
        ans_dict[query_id] = {}
        for index, row in rrf.iterrows():
            doc_id = int(row['_id'])
            score = row['_score']
            ans_dict[query_id][doc_id] = score
    for k in ans_dict.keys():
        ans_dict[k] = sorted(ans_dict[k].items(), key=lambda x: x[1], reverse=True)
    for k in ans_dict.keys():
        ans_dict[k] = [int(did) for did, _ in ans_dict[k]]

    if args.data_type == 'Lecard':
        labels = '../LeCaRD-main/data/label/test_label.json'
        coms = '../LeCaRD-main/data/prediction/combined_top100.json'
        labels, coms = load_file(labels, coms)
        keys = [i for i in list(coms.keys())[:100] if list(coms.keys())[:100].index(i) % 5 == 0]
        cal_metric([ans_dict], keys, labels, coms, args.data_type)
        print('Lecard test finish!')
    elif args.data_type == 'ELAM':
        labels = '../elam_data/elam_test_label.json'
        coms = '../elam_data/elam_test_top50.json'
        labels, coms = load_file(labels, coms)
        keys = list(coms.keys())
        cal_metric([ans_dict], keys, labels, coms, args.data_type)
        print('ELAM test finish!')

    # with open(args.save_path, mode='w', encoding='utf-8') as f:
    #     f.write(json.dumps(ans_dict, ensure_ascii=False))
    print("test finish")

if __name__ == '__main__':
    mm()