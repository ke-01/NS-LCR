import re
import torch
from torch.utils.data import Dataset
import json
import os
from torch.utils.data import DataLoader
from bert_utils import *
from random import choice
import random

class SimilarLawDataSet(Dataset):
    def __init__(self, candidates_path, logic_path, query_path, label_path,tokenizer, max_len,data_type):
        super(SimilarLawDataSet, self).__init__()
        self.data_type = data_type
        self.candidates_path = candidates_path
        self.logic_path = logic_path
        self.query_path= query_path
        self.label_path = label_path
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.querys = read_query(self.query_path,self.data_type)  # query id : query
        self.data_pair_list= read_pos_pairs(self.candidates_path,self.logic_path,self.querys,self.label_path,self.data_type)

    def pad_seq(self, ids_list, types_list):
        batch_len = 512  # max([len(ids) for ids in ids_list])
        new_ids_list, new_types_list, new_masks_list = [], [], []
        for ids, types in zip(ids_list, types_list):
            masks = [1] * len(ids) + [0] * (batch_len - len(ids))
            types += [0] * (batch_len - len(ids))
            ids += [0] * (batch_len - len(ids))
            new_ids_list.append(ids)
            new_types_list.append(types)
            new_masks_list.append(masks)
        return new_ids_list, new_types_list, new_masks_list

    def __len__(self):
        return len(self.data_pair_list)

    def __getitem__(self, item):
        tx, p,s_p_mapping,p_num, in_laws,label,q_id,d_id = self.data_pair_list[item]
        x_list=[tx]*p_num
        x_num=0
        s_idss=[]
        s_type_idss=[]
        s_mask_ss=[]
        for ft in p:
            idss = []
            type_idss = []
            mask_ss = []
            for tp in ft:
                x=x_list[x_num]
                x_num+=1
                if len(x)+len(tp)>509:
                    if len(tp) > 29:
                        tp=tp[:29]
                        x=x[:480]
                    else:
                        t=len(tp)
                        x=x[:509-t]

                ids, type_ids = [], []
                x_tokens, p_tokens = self.tokenizer.tokenize(x), self.tokenizer.tokenize(tp)
                crime_tokens = ['[CLS]'] + x_tokens + ['[SEP]'] + p_tokens + ['[SEP]']
                crime_ids = self.tokenizer.convert_tokens_to_ids(crime_tokens)
                crime_types = [0] * (len(x_tokens) + 2) + [1] * (len(p_tokens)+1)
                ids.append(crime_ids)
                type_ids.append(crime_types)
                ids, type_ids, masks = self.pad_seq(ids, type_ids)
                ids, type_ids, masks = torch.LongTensor(ids),torch.LongTensor(type_ids),torch.LongTensor(masks)
                idss.append(ids)
                type_idss.append(type_ids)
                mask_ss.append(masks)
            s_idss.append(idss)
            s_type_idss.append(type_idss)
            s_mask_ss.append(mask_ss)

        return {'input_ids': s_idss,
                'token_type_ids': s_type_idss,
                'attention_mask': s_mask_ss,
                'mapping':s_p_mapping,
                'in_laws':in_laws,
                'label':label,
                'q_id':q_id,
                'd_id':d_id}


def read_query(query_path,data_type):
    querys = {}
    with open(query_path, mode='r', encoding='utf-8')as f:
        lines = f.readlines()
        for line in lines:
            data = json.loads(line)
            if data_type=='elam':
                ridx = data['id']
                query = "".join(data['q'])
            elif data_type=='Lecard':
                ridx = data['ridx']
                query = data['q']
            else:
                print("error")
                exit()

            querys[str(ridx)] = query
    return querys

to_num={'一':1,'二':2,'三':3,'四':4,'五':5,'六':6,'七':7,'八':8,'九':9}

def read_pos_pairs(candidates_path, logic_path,querys, label_path,data_type):

    with open(logic_path, 'r', encoding='utf-8') as f:
        file = json.load(f)
    folders = os.listdir(candidates_path)
    data_pair_list = []

    t_label={}#存对应的标签
    with open(label_path, mode='r', encoding='utf-8')as f:
        line = f.readlines()[0]
        pos_labels = json.loads(line)
        for k in pos_labels.keys():
            for file_t in pos_labels[k]:
                if data_type=='elam':
                    t_label[k,file_t]=pos_labels[k][file_t] / 2
                elif data_type=='Lecard':
                    t_label[k, file_t] = pos_labels[k][file_t] / 3
                else:
                    print('error')
                    exit()
            pos_labels[k] = [str(i) for i in pos_labels[k]]
    neg_labels = {}
    for folder in folders:
        pos = pos_labels[folder]
        neg_labels[folder] = []
        folder_path = os.path.join(candidates_path, folder)
        files = os.listdir(folder_path)
        for filename in files:
            filename = filename.replace('.json', '')
            if filename not in pos:  # 负样本
                neg_labels[folder].append(filename)

    for key in neg_labels.keys():
        for filename in neg_labels[key]:
            t_label[key, filename] = 0.0

    for folder in folders:
        query_id=folder
        query = querys[folder]  # query和candidates配对
        folder_path = os.path.join(candidates_path, folder)
        files = os.listdir(folder_path)
        for filename in files:
            doc_id=filename
            filename_t = filename.replace('.json', '')
            label_t=t_label[folder,filename_t]
            with open(folder_path+'/'+filename, mode='r', encoding='utf-8')as f:
                js_dict = json.load(f)
            all_law = js_dict['laws']# 得到所有法条
            law_idx = []
            for i in all_law:
                p = re.compile('第(?:十|百|零|一|二|三|四|五|六|七|八|九){1,10}条(?:之(?:一|二|三|四|五|六|七|八|九))?(?:第(?:十|百|零|一|二|三|四|五|六|七|八|九)款)?')
                m = re.findall(pattern=p, string=i)
                law_idx.append(m[0])
            s_predic = []  # 得到对应的谓词存入列表
            p_sum = 0  # 统计该json文件所有法条对应的谓词一共多少个
            s_p_mapping = []  # 统计该json文件每个法条中谓词名称
            in_laws=[]  # 统计真实存在在logic文件中的laws,存当前候选案例的所有法条，每个法条应该有不同谓词，所以应该append
            for i in law_idx:
                # 对每个法条
                predic=[] #得到当前法条对应的谓词
                p_mapping = {} #可能加空的mapping进去
                if i in file:  # 没有区分第几款，直接可以在法条里面索引出来的，加入全部谓词
                    law_one=[]
                    for ll in file[i]['Rules']:
                        law_one.extend(ll)
                    in_laws.append(law_one)
                    # law_one.extend(file[i]['Rules'])
                    for key, value in file[i]['Predicate'].items():
                        if 'P' in key:
                            predic.append(value)
                            p_sum+=1
                            p_mapping[key] = value
                elif "之一" in i and "款" in i:  # 有之一和款，用之一分割得到第几款
                    t = i.split('之一')
                    idx = t[0] + '之一'
                    idx_k = to_num[t[1][1]] - 1
                    if idx in file:
                        if idx_k >= len(file[idx]['Rules']):
                            law_one = []
                            for ll in file[idx]['Rules']:
                                law_one.extend(ll)
                            in_laws.append(law_one)
                            for key, value in file[idx]['Predicate'].items():
                                if 'P' in key:
                                    predic.append(value)
                                    p_sum += 1
                                    p_mapping[key] = value
                        else:
                            rule_t = file[idx]['Rules'][idx_k]  # 找到对应的那款rule,存在法条或者案例中款不对应
                            in_laws.append(rule_t) #这样加入的是[]，可以直接访问
                            P_list = []  # 得到对应rule里面的谓词
                            for item in rule_t:
                                P_list.extend(re.findall("[P]\d*", item))#只用P,不用Y
                            for j in P_list:
                                p_t = file[idx]['Predicate'][j]
                                predic.append(p_t)
                                p_sum += 1
                                p_mapping[j] = p_t


                elif "款" in i:  # 没有之一，只有款，用条分割得到第几款
                    t = i.split('条')
                    idx = t[0] + '条'
                    idx_k = to_num[t[1][1]] - 1
                    if idx in file:
                        if idx_k >= len(file[idx]['Rules']):
                            law_one = []
                            for ll in file[idx]['Rules']:
                                law_one.extend(ll)
                            in_laws.append(law_one)
                            for key, value in file[idx]['Predicate'].items():
                                if 'P' in key:
                                    predic.append(value)
                                    p_sum += 1
                                    p_mapping[key] = value
                        else:
                            rule_t = file[idx]['Rules'][idx_k]  # 找到对应的那款rule
                            in_laws.append(rule_t)
                            P_list = []  # 得到对应rule里面的谓词
                            for item in rule_t:
                                P_list.extend(re.findall("[P]\d*", item))
                            for j in P_list:
                                p_t = file[idx]['Predicate'][j]
                                predic.append(p_t)
                                p_sum += 1
                                p_mapping[j] = p_t
                else:
                    continue
                if len(predic) > 0:
                    s_predic.append(predic)
                    s_p_mapping.append(p_mapping)
            data_pair_list.append((query,s_predic,s_p_mapping,p_sum,in_laws,label_t,query_id,doc_id))

    return data_pair_list
