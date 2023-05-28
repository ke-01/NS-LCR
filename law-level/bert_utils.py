import torch
import math
import pandas as pd


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



def select_collate(batch):
    inputs_ids, inputs_masks, types_ids,q_id,d_id,label= None, None, None,None, None,None
    for i, s in enumerate(batch):
        if i == 0:
            inputs_ids, inputs_masks, types_ids,q_id,d_id,label = s['input_ids'], s['attention_mask'], s['token_type_ids'],s['q_id'],s['d_id'],s['label']
        else:
            inputs_ids = torch.cat([inputs_ids, s['input_ids']], dim=0)
            inputs_masks = torch.cat([inputs_masks, s['attention_mask']], dim=0)
            types_ids = torch.cat([types_ids, s['token_type_ids']], dim=0)
            label=torch.cat([label, s['label']], dim=0)

    return {'input_ids': inputs_ids,
            'attention_mask': inputs_masks,
            'token_type_ids': types_ids,
            'q_id':q_id,
            'd_id':d_id,
            'label':label}

def select_collate_test(batch):
    inputs_ids, inputs_masks, types_ids,q_id,d_id= None, None, None,None, None
    for i, s in enumerate(batch):
        if i == 0:
            inputs_ids, inputs_masks, types_ids,q_id,d_id = s['input_ids'], s['attention_mask'], s['token_type_ids'],s['q_id'],s['d_id']

    return {'input_ids': inputs_ids,
            'attention_mask': inputs_masks,
            'token_type_ids': types_ids,
            'q_id':q_id,
            'd_id':d_id}

def select_collate_fol(batch):
    inputs_ids, inputs_masks, types_ids,mapping,in_laws,label,q_id,d_id= None, None, None,None,None,None,None,None
    for i, s in enumerate(batch):
        if i == 0:
            inputs_ids, inputs_masks, types_ids,mapping,in_laws,label,q_id,d_id = s['input_ids'], s['attention_mask'], s['token_type_ids'],s['mapping'],s['in_laws'],s['label'],s['q_id'],s['d_id']
        else:
            inputs_ids = torch.cat([inputs_ids, s['input_ids']], dim=0)
            inputs_masks = torch.cat([inputs_masks, s['attention_mask']], dim=0)
            types_ids = torch.cat([types_ids, s['token_type_ids']], dim=0)
            mapping = torch.cat([mapping, s['mapping']], dim=0)
            in_laws = torch.cat([in_laws, s['in_laws']], dim=0)
            label=torch.cat([label, s['label']], dim=0)

    return {'input_ids': inputs_ids,
            'attention_mask': inputs_masks,
            'token_type_ids': types_ids,
            'in_laws':in_laws,
            'mapping':mapping,
            'label':label,
            'q_id':q_id,
            'd_id':d_id
            }

def select_collate_struct(batch):
    q_outputs, d_outputs,q_id,d_id,label= None, None, None,None, None
    for i, s in enumerate(batch):
        if i == 0:
            q_outputs, d_outputs, q_id,d_id,label = s['q_outputs'], s['d_outputs'], s['q_id'],s['d_id'],s['label']
        else:
            q_outputs = torch.cat([q_outputs, s['q_outputs']], dim=0)
            d_outputs = torch.cat([d_outputs, s['d_outputs']], dim=0)
            types_ids = torch.cat([types_ids, s['token_type_ids']], dim=0)
            label=torch.cat([label, s['label']], dim=0)

    return {'q_outputs': q_outputs,
            'd_outputs': d_outputs,
            'q_id':q_id,
            'd_id':d_id,
            'label':label}

group_ids=[['第一条','第二条','第三条','第四条','第五条','第六条','第七条','第八条','第九条','第十条','第十一条','第十二条'],
           ['第十三条','第十四条','第十五条','第十六条','第十七条','第十七条之一','第十八条','第十九条','第二十条','第二十一条'],
           ['第二十二条','第二十三条','第二十四条'],
           ['第二十五条','第二十六条','第二十七条','第二十八条','第二十九条'],
           ['第三十条','第三十一条'],
           ['第三十二条','第三十三条','第三十四条','第三十五条','第三十六条','第三十七条','第三十七条之一'],
           ['第三十八条','第三十九条','第四十条','第四十一条'],
           ['第四十二条','第四十三条','第四十四条'],
           ['第四十五条','第四十六条','第四十七条'],
           ['第四十八条','第四十九条','第五十条','第五十一条'],
           ['第五十二条','第五十三条'],
           ['第五十四条','第五十五条','第五十六条','第五十七条','第五十八条'],
           ['第五十九条','第六十条'],
           ['第六十一条','第六十二条','第六十三条','第六十四条'],
           ['第六十五条','第六十六条'],
           ['第六十七条','第六十八条'],
           ['第六十九条','第七十条','第七十一条'],
           ['第七十二条','第七十三条','第七十四条','第七十五条','第七十六条','第七十七条'],
           ['第七十八条','第七十九条','第八十条'],
           ['第八十一条','第八十二条','第八十三条','第八十四条','第八十五条','第八十六条'],
           ['第八十七条','第八十八条','第八十九条'],
           ['第九十条','第九十一条','第九十二条','第九十三条','第九十四条','第九十五条',
            '第九十六条','第九十七条','第九十八条','第九十九条','第一百条','第一百零一条'],
           ['第一百零二条','第一百零三条','第一百零四条','第一百零五条','第一百零六条','第一百零七条',
            '第一百零八条','第一百零九条','第一百一十条','第一百一十一条','第一百一十二条','第一百一十三条'],
           ['第一百一十四条','第一百一十五条','第一百一十六条','第一百一十七条','第一百一十八条',
            '第一百一十九条','第一百二十条','第一百二十条之一','第一百二十条之二',
            '第一百二十条之三','第一百二十条之四','第一百二十条之五','第一百二十条之六','第一百二十一条',
            '第一百二十二条','第一百二十三条','第一百二十四条','第一百二十五条',
            '第一百二十六条','第一百二十七条','第一百二十八条','第一百二十九条','第一百三十条','第一百三十一条',
            '第一百三十二条','第一百三十三条','第一百三十三条之一',
            '第一百三十三条之二','第一百三十四条','第一百三十四条之一','第一百三十五条','第一百三十五条之一',
            '第一百三十六条','第一百三十七条','第一百三十八条','第一百三十九条','第一百三十九条之一'],
           ['第一百四十条','第一百四十一条','第一百四十二条','第一百四十二条之一','第一百四十三条','第一百四十四条',
            '第一百四十五条','第一百四十六条','第一百四十七条','第一百四十八条','第一百四十九条','第一百五十条'],
           ['第一百五十一条','第一百五十二条','第一百五十三条','第一百五十四条','第一百五十五条','第一百五十六条','第一百五十七条'],
           ['第一百五十八条','第一百五十九条','第一百六十条','第一百六十一条','第一百六十二条','第一百六十二条之一',
            '第一百六十二条之二','第一百六十三条','第一百六十四条','第一百六十五条','第一百六十六条','第一百六十七条',
            '第一百六十八条','第一百六十九条','第一百六十九条之一'],
           ['第一百七十条','第一百七十一条','第一百七十二条','第一百七十三条','第一百七十四条','第一百七十五条',
            '第一百七十五条之一','第一百七十六条','第一百七十七条','第一百七十七条之一','第一百七十八条',
            '第一百七十九条','第一百八十条','第一百八十一条','第一百八十二条','第一百八十三条',
           '第一百八十四条','第一百八十五条','第一百八十五条之一','第一百八十六条','第一百八十七条','第一百八十八条',
            '第一百八十九条','第一百九十条','第一百九十一条'],
           ['第一百九十二条','第一百九十三条','第一百九十四条','第一百九十五条','第一百九十六条','第一百九十七条',
            '第一百九十八条','第二百条'],
           ['第二百零一条','第二百零二条','第二百零三条','第二百零四条','第二百零五条','第二百零五条之一',
            '第二百零六条','第二百零七条','第二百零八条','第二百零九条','第二百一十条','第二百一十条之一',
           '第二百一十一条','第二百一十二条'],
           ['第二百一十三条','第二百一十四条','第二百一十五条','第二百一十六条','第二百一十七条','第二百一十八条',
            '第二百一十九条','第二百一十九条之一','第二百二十条'],
           ['第二百二十一条','第二百二十二条','第二百二十三条','第二百二十四条','第二百二十四条之一',
            '第二百二十五条','第二百二十六条','第二百二十七条','第二百二十八条','第二百二十九条','第二百三十条','第二百三十一条'],
           ['第二百三十二条','第二百三十三条','第二百三十四条','第二百三十四条之一','第二百三十五条',
            '第二百三十六条','第二百三十六条之一','第二百三十七条','第二百三十八条','第二百三十九条',
           '第二百四十条','第二百四十一条','第二百四十二条','第二百四十三条','第二百四十四条',
            '第二百四十四条之一','第二百四十五条','第二百四十六条','第二百四十七条','第二百四十八条',
           '第二百四十九条','第二百五十条','第二百五十一条','第二百五十二条','第二百五十三条',
            '第二百五十三条之一','第二百五十四条','第二百五十五条','第二百五十六条','第二百五十七条',
           '第二百五十八条','第二百五十九条','第二百六十条','第二百六十条之一','第二百六十一条',
            '第二百六十二条','第二百六十二条之一','第二百六十二条之二'],
           ['第二百六十三条','第二百六十四条','第二百六十五条','第二百六十六条','第二百六十七条',
            '第二百六十八条','第二百六十九条','第二百七十条','第二百七十一条','第二百七十二条',
           '第二百七十三条','第二百七十四条','第二百七十五条','第二百七十六条','第二百七十六条之一'],
           ['第二百七十七条','第二百七十八条','第二百七十九条','第二百八十条','第二百八十条之一',
            '第二百八十条之二','第二百八十一条','第二百八十二条','第二百八十三条','第二百八十四条',
           '第二百八十四条之一','第二百八十五条','第二百八十六条','第二百八十六条之一','第二百八十七条',
            '第二百八十七条之一','第二百八十七条之二','第二百八十八条','第二百八十九条','第二百九十条',
           '第二百九十一条','第二百九十一条之一','第二百九十一条之二','第二百九十二条','第二百九十三条',
            '第二百九十三条之一','第二百九十四条','第二百九十五条','第二百九十六条','第二百九十七条',
           '第二百九十八条','第二百九十九条','第二百九十九条之一','第三百条','第三百零一条','第三百零二条','第三百零三条','第三百零四条'],
           ['第三百零五条','第三百零六条','第三百零七条','第三百零七条之一','第三百零八条','第三百零八条之一',
            '第三百零九条','第三百一十条',
           '第三百一十一条','第三百一十二条','第三百一十三条','第三百一十四条','第三百一十五条','第三百一十六条','第三百一十七条'],
           ['第三百一十八条','第三百一十八条','第三百二十条','第三百二十一条','第三百二十二条','第三百二十三条'],
           ['第三百二十四条','第三百二十五条','第三百二十六条','第三百二十七条','第三百二十八条','第三百二十九条'],
           ['第三百三十条','第三百三十一条','第三百三十二条','第三百三十三条','第三百三十四条','第三百三十四条之一',
           '第三百三十五条','第三百三十六条','第三百三十六条之一','第三百三十七条'],
           ['第三百三十八条','第三百三十九条','第三百四十条','第三百四十一条','第三百四十二条','第三百四十二条之一',
           '第三百四十三条','第三百四十四条','第三百四十四条之一','第三百四十五条','第三百四十六条'],
           ['第三百四十七条','第三百四十八条','第三百四十九条','第三百五十条','第三百五十一条','第三百五十二条',
           '第三百五十三条','第三百五十四条','第三百五十五条','第三百五十五条之一','第三百五十六条','第三百五十七条'],
           ['第三百五十八条','第三百五十九条','第三百六十条','第三百六十一条','第三百六十二条'],
           ['第三百六十三条','第三百六十四条','第三百六十五条','第三百六十六条','第三百六十七条'],
           ['第三百六十八条','第三百六十九条','第三百七十条','第三百七十一条','第三百七十二条','第三百七十三条',
           '第三百七十四条','第三百七十五条','第三百七十六条','第三百七十七条','第三百七十八条','第三百七十九条',
           '第三百八十条','第三百八十一条'],
           ['第三百八十二条','第三百八十三条','第三百八十四条','第三百八十五条','第三百八十六条','第三百八十七条',
           '第三百八十八条','第三百八十八条之一','第三百八十九条','第三百九十条','第三百九十条之一','第三百九十一条',
           '第三百九十二条','第三百九十三条','第三百九十四条','第三百九十五条','第三百九十六条'],
           ['第三百九十七条','第三百九十八条','第三百九十九条','第三百九十九条之一','第四百条','第四百零一条','第四百零二条',
           '第四百零三条','第四百零四条','第四百零五条','第四百零六条','第四百零七条','第四百零八条','第四百零八条之一',
           '第四百零九条','第四百一十条','第四百一十一条','第四百一十二条','第四百一十三条','第四百一十四条','第四百一十五条',
           '第四百一十六条','第四百一十七条','第四百一十八条','第四百一十九条'],
           ['第四百二十条','第四百二十一条','第四百二十二条','第四百二十三条','第四百二十四条','第四百二十五条','第四百二十六条',
           '第四百二十七条','第四百二十八条','第四百二十九条','第四百三十条','第四百三十一条','第四百三十二条','第四百三十三条',
           '第四百三十四条','第四百三十五条','第四百三十六条','第四百三十七条','第四百三十八条','第四百三十九条','第四百四十条',
           '第四百四十一条','第四百四十二条','第四百四十三条','第四百四十四条','第四百四十五条','第四百四十六条',
           '第四百四十七条','第四百四十八条','第四百四十九条','第四百五十条','第四百五十一条']]