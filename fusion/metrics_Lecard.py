# -*- encoding: utf-8 -*-

import os
import numpy as np
import json
import math
import functools
import argparse
from tqdm import tqdm

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

def load_file(labels,coms):
    with open(labels, 'r') as f:
        avglist = json.load(f)

    with open(coms, 'r') as f:
        combdic = json.load(f)

    # with open(os.path.join(args.pred, 'cat_bertpli_L_0422_seed42_final.json'), 'r') as f:
    #     lawformer_dic = json.load(f)
    
    return avglist, combdic
    # return avglist, combdic,lawformer_dic

def cal_metric(dics,keys,avglist,combdic,data_type):
    print('cal metrics of {}:'.format(data_type))
    # NDCG
    topK_list = [10, 20, 30]
    ndcg_list = []
    for topK in topK_list:
        temK_list = []
        for dic in dics:
            sndcg = 0.0
            for key in keys:
                rawranks = []
                for i in dic[key]:
                    if str(i) in list(avglist[key])[:30]:
                        rawranks.append(avglist[key][str(i)])
                    else:
                        rawranks.append(0)
                # rawranks = [avglist[key][str(i)] for i in dic[key] if i in list(combdic[key][:30])]
                ranks = rawranks + [0] * (30 - len(rawranks))
                if sum(ranks) != 0:
                    sndcg += ndcg1(ranks, list(avglist[key].values()), topK)
                    # sndcg += ndcg(ranks, topK)
            temK_list.append(sndcg / len(keys))
        ndcg_list.append(temK_list)
    print('NDCG@10,NDCG@20,NDCG@30:{}'.format(ndcg_list))
    # print(ndcg_list)

    # P
    topK_list = [5, 10]
    sp_list = []

    for topK in topK_list:
        temK_list = []
        for rdic in dics:
            sp = 0.0
            for key in keys:
                ranks = [i for i in rdic[key] if i in list(combdic[key][:30])]
                if data_type=='Lecard':
                    sp += float(len([j for j in ranks[:topK] if avglist[key][str(j)] == 3]) / topK)
                elif data_type=='ELAM':
                    sp += float(len([j for j in ranks[:topK] if avglist[key][str(j)] == 2]) / topK)
            temK_list.append(sp / len(keys))
        sp_list.append(temK_list)
    print('P@5,P@10:{}'.format(sp_list))
    # print(sp_list)

    # MAP
    map_list = []
    for rdic in dics:
        smap = 0.0
        for key in keys:
            ranks = [i for i in rdic[key] if i in list(combdic[key][:30])]
            if data_type=='Lecard':
                rels = [ranks.index(i) for i in ranks if avglist[key][str(i)] == 3]
                tem_map = 0.0
                for rel_rank in rels:
                    tem_map += float(len([j for j in ranks[:rel_rank + 1] if avglist[key][str(j)] == 3]) / (rel_rank + 1))
            elif data_type=='ELAM':
                rels = [ranks.index(i) for i in ranks if avglist[key][str(i)] == 2]
                tem_map = 0.0
                for rel_rank in rels:
                    tem_map += float(len([j for j in ranks[:rel_rank + 1] if avglist[key][str(j)] == 2]) / (rel_rank + 1))
            if len(rels) > 0:
                smap += tem_map / len(rels)
        map_list.append(smap / len(keys))
    print('MAP:{}'.format(map_list))
    # print(map_list)
    return ndcg_list[2][0]

if __name__ == "__main__":
    print('scores:')





