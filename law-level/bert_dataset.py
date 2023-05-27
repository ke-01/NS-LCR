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
    def __init__(self, candidates_path, logic_path, ids2group_path,tokenizer, max_len,data_type):
        super(SimilarLawDataSet, self).__init__()
        self.data_type = data_type
        self.candidates_path = candidates_path
        self.logic_path = logic_path
        self.ids2group_path= ids2group_path
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.data_pair_list = read_pos_pairs(self.candidates_path,self.logic_path,self.ids2group_path)

    def pad_seq(self, ids_list, types_list):
        batch_len = 512  # max([len(ids) for ids in ids_list])
        # batch_len = 4096  # lawformer
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
        x, p, label = self.data_pair_list[item]
        # for lawformer
        # if len(x)+len(p)>4093:
        #     if len(p) > 29:
        #         p=p[:29]
        #         x=x[:4060]
        #     else:
        #         t=len(p)
        #         x=x[:4093-t]

        # for bert
        if len(x)+len(p)>509:
            if len(p) > 29:
                p=p[:29]
                x=x[:480]
            else:
                t=len(p)
                x=x[:509-t]


        ids, type_ids = [], []
        x_tokens, p_tokens = self.tokenizer.tokenize(x), self.tokenizer.tokenize(p)
        crime_tokens = ['[CLS]'] + x_tokens + ['[SEP]'] + p_tokens + ['[SEP]']
        crime_ids = self.tokenizer.convert_tokens_to_ids(crime_tokens)
        crime_types = [0] * (len(x_tokens) + 2) + [1] * (len(p_tokens)+1)
        ids.append(crime_ids)
        type_ids.append(crime_types)

        ids, type_ids, masks = self.pad_seq(ids, type_ids)

        label = torch.Tensor([label])

        return {'input_ids': torch.LongTensor(ids),
                'token_type_ids': torch.LongTensor(type_ids),
                'attention_mask': torch.LongTensor(masks),
                'label': label}

def get_doc(candidates_path, text1_idx, text2_idx,data_type):
    file_path = os.path.join(candidates_path, text1_idx, text2_idx+'.json')
    doc = ''
    with open(file_path, mode='r', encoding='utf-8')as f:
        js_dict = json.load(f)
        if data_type=='elam':
            doc += "".join(js_dict['doc'])
        elif data_type=='Lecard':
            doc += js_dict['ajjbqk']
            laws=js_dict['laws']
            law_idx = []
            for i in laws:
                p = re.compile('第(?:十|百|零|一|二|三|四|五|六|七|八|九){1,10}条(?:之(?:一|二|三|四|五|六|七|八|九))?(?:第(?:十|百|零|一|二|三|四|五|六|七|八|九)款)?')
                m = re.findall(pattern=p, string=i)
                law_idx.append(m[0])

    return doc

to_num={'一':1,'二':2,'三':3,'四':4,'五':5,'六':6,'七':7,'八':8,'九':9}


def read_pos_pairs(candidates_path,logic_path,ids2group_path):
    with open(logic_path, 'r', encoding='utf-8') as f:
        file = json.load(f)
    with open(ids2group_path, 'r', encoding='utf-8') as f:
        ids2group = json.load(f)
    folders = os.listdir(candidates_path)
    data_pair_list = []
    for filename in folders:
        #对每个json文件
        len_pos=0
        with open(candidates_path+'/'+filename, mode='r', encoding='utf-8')as f:
            js_dict = json.load(f)
        doc = js_dict['ajjbqk']
        all_law = js_dict['laws']#得到法条
        law_idx = []
        for i in all_law:
            p = re.compile(
                '第(?:十|百|零|一|二|三|四|五|六|七|八|九){1,10}条(?:之(?:一|二|三|四|五|六|七|八|九))?(?:第(?:十|百|零|一|二|三|四|五|六|七|八|九)款)?')
            m = re.findall(pattern=p, string=i)
            law_idx.append(m[0])
        # law_idx 即去除前面的《》
        predic = [] #得到对应的谓词存入列表
        pos_idx=[]
        for i in law_idx:
            if i in file:  # 没有区分第几款，直接可以在法条里面索引出来的，加入全部谓词
                pos_idx.append(i)
                for key, value in file[i]['Predicate'].items():
                    if 'P' in key:
                        predic.append(value)
            elif "之一" in i and "款" in i:  # 有之一和款，用之一分割得到第几款
                t = i.split('之一')
                idx = t[0] + '之一'
                idx_k = to_num[t[1][1]] - 1
                if idx in file:
                    pos_idx.append(idx)
                    if idx_k >= len(file[idx]['Rules']):
                        for key, value in file[idx]['Predicate'].items():
                            if 'P' in key:
                                predic.append(value)
                    else:
                        rule_t = file[idx]['Rules'][idx_k]  # 找到对应的那款rule
                        P_list = []  # 得到对应rule里面的谓词
                        for item in rule_t:
                            P_list.extend(re.findall("[P]\d*", item))#只用P,不用Y
                        for j in P_list:
                            p_t = file[idx]['Predicate'][j]
                            predic.append(p_t)
            elif "款" in i:  # 没有之一，只有款，用条分割得到第几款
                t = i.split('条')
                idx = t[0] + '条'
                idx_k = to_num[t[1][1]] - 1
                if idx in file:
                    pos_idx.append(idx)
                    if idx_k >= len(file[idx]['Rules']):
                        for key, value in file[idx]['Predicate'].items():
                            if 'P' in key:
                                predic.append(value)
                    else:
                        rule_t = file[idx]['Rules'][idx_k]  # 找到对应的那款rule
                        P_list = []  # 得到对应rule里面的谓词
                        for item in rule_t:  # 只用P
                            P_list.extend(re.findall("[P]\d*", item))
                        for j in P_list:
                            p_t = file[idx]['Predicate'][j]
                            predic.append(p_t)
            else:
                continue
        for pr in predic:
            data_pair_list.append((doc, pr, 1))#正例
            len_pos += 1  # 统计正例谓词数量

        neg_cnt=0
        end_flag=0
        used_ids=[]
        used_ids.extend(pos_idx)
        pos_group_id=[]  # 得到所有正例所在的group，区分hard和soft
        for pos_id in pos_idx:
            group_id=ids2group[pos_id]  # 找到这一个法条对应法条的同一节/章
            pos_group_id.append(group_id)
        for pos_id in pos_idx:  # 遍历每一个法条
            # 对每一个法条 添加一次硬负例和一次软负例
            group_id=ids2group[pos_id]  # 找到这一个法条对应法条的同一节/章
            group_hard=group_ids[group_id]  # 获得该组所有法条名称

            hard_id=choice(group_hard)  # 得到一个hard_id
            cnt=1
            while hard_id in used_ids or hard_id not in file:
                hard_id = choice(group_hard)  # 得到一个hard_id
                cnt+=1
                if cnt >= len(group_hard)*2:break  # 该group里面的id已经全被用过了
            if hard_id not in used_ids and hard_id in file:
                used_ids.append(hard_id)
                for j in file[hard_id]['Predicate'].values():  # 添加硬负例的谓词
                    data_pair_list.append((doc, j, 0))
                    neg_cnt = neg_cnt+1
                    if neg_cnt >= len_pos:  # 负例数量等于正例数量
                        end_flag = 1
                        break
            if end_flag == 1:
                break

            soft_group_id=random.randint(0,len(group_ids)-1)  # 在别的章节选择软负例
            while soft_group_id in pos_group_id:
                soft_group_id = random.randint(0, len(group_ids) - 1)
            soft_group=group_ids[soft_group_id]  # 软负例组号
            soft_id= choice(soft_group)  # 在组内选一个法条
            while soft_id in used_ids or soft_id not in file:
                soft_id = choice(soft_group)  # 得到一个soft_id
                cnt+=1
                if cnt >= len(soft_group)*2:break  # 该group里面的id已经全被用过了
            if soft_id not in used_ids and soft_id in file:  # 最后还是落到
                used_ids.append(soft_id)
                for j in file[soft_id]['Predicate'].values():  # 添加软负例的谓词
                    data_pair_list.append((doc, j, 0))
                    neg_cnt = neg_cnt+1
                    if neg_cnt >= len_pos:
                        end_flag = 1
                        break
            if end_flag == 1:
                break

    return data_pair_list




