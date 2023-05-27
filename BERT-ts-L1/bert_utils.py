import torch
import math

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