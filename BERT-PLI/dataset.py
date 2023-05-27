import re
import torch
from torch.utils.data import Dataset
import json
import os

class SimilarLawDataSet(Dataset):
    def __init__(self, candidates_path, query_path, golden_label_path, tokenizer, max_para_q, max_para_d, para_max_len, max_len, data_type):
        super(SimilarLawDataSet, self).__init__()
        self.data_type = data_type
        self.candidates_path = candidates_path
        self.query_path = query_path
        self.label_path = golden_label_path
        self.tokenizer = tokenizer
        self.para_max_len = para_max_len
        self.max_para_q = max_para_q
        self.max_para_d = max_para_d
        self.max_len = max_len
        self.querys = read_query(self.query_path, self.data_type)
        self.data_pair_list = read_pairs(self.candidates_path, self.label_path, self.data_type)

    def __len__(self):
        return len(self.data_pair_list)

    def __getitem__(self, item):
        q_idx, d_idx, label = self.data_pair_list[item]

        q = self.querys[q_idx]
        d = get_doc(self.candidates_path, q_idx, d_idx,self.data_type)
        q_para = segment_to_para(q, self.para_max_len)
        d_para = segment_to_para(d, self.para_max_len)

        input_ids_item = []
        attention_mask_item = []
        token_type_ids_item = []

        for m in range(min(self.max_para_q, len(q_para))):
            p1 = q_para[m]
            input_ids_row = []
            attention_mask_row = []
            token_type_ids_row = []
            for n in range(min(self.max_para_d, len(d_para))):
                p2 = d_para[n]
                res_dict = self.tokenizer.encode_plus(text=p1, text_pair=p2, max_length=self.max_len,
                                                      return_tensors='pt', add_special_tokens=True, padding='max_length')  # input_ids, token_type_ids, attention_mask
                input_ids_row.append(res_dict['input_ids'])
                attention_mask_row.append(res_dict['attention_mask'])
                token_type_ids_row.append(res_dict['token_type_ids'])

            if len(d_para) < self.max_para_d:
                for j in range(len(d_para), self.max_para_d):
                    input_ids_row.append(torch.zeros([1, self.max_len], dtype=torch.int64))
                    attention_mask_row.append(torch.zeros([1, self.max_len], dtype=torch.int64))
                    token_type_ids_row.append(torch.zeros([1, self.max_len], dtype=torch.int64))

            assert (len(input_ids_row) == self.max_para_d)
            assert (len(attention_mask_row) == self.max_para_d)
            assert (len(token_type_ids_row) == self.max_para_d)

            input_ids_row = torch.cat(input_ids_row, dim=0).unsqueeze(0)
            attention_mask_row = torch.cat(attention_mask_row, dim=0).unsqueeze(0)
            token_type_ids_row = torch.cat(token_type_ids_row, dim=0).unsqueeze(0)

            input_ids_item.append(input_ids_row)
            attention_mask_item.append(attention_mask_row)
            token_type_ids_item.append(token_type_ids_row)

        if len(q_para) < self.max_para_q:
            for i in range(len(q_para), self.max_para_q):
                input_ids_row = []  #
                attention_mask_row = []
                token_type_ids_row = []
                for j in range(self.max_para_d):
                    input_ids_row.append(torch.zeros([1, self.max_len], dtype=torch.int64))
                    attention_mask_row.append(torch.zeros([1, self.max_len], dtype=torch.int64))
                    token_type_ids_row.append(torch.zeros([1, self.max_len], dtype=torch.int64))

                input_ids_row = torch.cat(input_ids_row, dim=0).unsqueeze(0)
                attention_mask_row = torch.cat(attention_mask_row, dim=0).unsqueeze(0)
                token_type_ids_row = torch.cat(token_type_ids_row, dim=0).unsqueeze(0)

                input_ids_item.append(input_ids_row)
                attention_mask_item.append(attention_mask_row)
                token_type_ids_item.append(token_type_ids_row)

        assert (len(input_ids_item) == self.max_para_q)
        assert (len(attention_mask_item) == self.max_para_q)
        assert (len(token_type_ids_item) == self.max_para_q)
        label = torch.Tensor([label])

        input_ids_item = torch.cat(input_ids_item, dim=0)
        token_type_ids_item = torch.cat(token_type_ids_item, dim=0)
        attention_mask_item = torch.cat(attention_mask_item, dim=0)
        return input_ids_item, token_type_ids_item, attention_mask_item, label, q_idx, d_idx


class SimilarLawTestDataSet(Dataset):
    def __init__(self, candidates_path, query_path, tokenizer, max_para_q, max_para_d, para_max_len, max_len, data_type):
        super(SimilarLawTestDataSet, self).__init__()
        self.candidates_path = candidates_path
        self.query_path = query_path
        self.tokenizer = tokenizer
        self.para_max_len = para_max_len
        self.max_para_q = max_para_q
        self.max_para_d = max_para_d
        self.max_len = max_len
        self.data_type = data_type
        self.querys = read_query(self.query_path, self.data_type)
        self.test_data = self.read_test_data()

        self.data_pair_list = self.gen_data_pair()

    def read_test_data(self):
        if self.data_type == 'elam':
            test_path = '../elam_data/elam_test_top50.json'
        elif self.data_type == 'Lecard':
            test_path = '../LeCaRD-main/data/prediction/test_top100.json'
        else:
            print('data error')
            exit()
        with open(test_path, mode='r', encoding='utf-8')as f:
            js_dict = json.load(f)
            for k in js_dict.keys():
                js_dict[k] = [str(v) for v in js_dict[k]]
        return js_dict  # query id, can ids

    def gen_data_pair(self):
        data_pair_list = []
        for k in self.test_data.keys():
            query = self.querys[k]
            for v in self.test_data[k]:
                doc = get_doc(self.candidates_path, k, v,self.data_type)
                data_pair_list.append((k, v, query, doc))
        return data_pair_list  # query, doc

    def __len__(self):
        return len(self.data_pair_list)

    def __getitem__(self, item):
        q_id, d_id, q, d = self.data_pair_list[item]
        # 分段
        q_para = segment_to_para(q, self.para_max_len)
        d_para = segment_to_para(d, self.para_max_len)

        input_ids_item = []
        attention_mask_item = []
        token_type_ids_item = []

        for m in range(min(self.max_para_q, len(q_para))):
            p1 = q_para[m]
            input_ids_row = []
            attention_mask_row = []
            token_type_ids_row = []
            for n in range(min(self.max_para_d, len(d_para))):
                p2 = d_para[n]
                res_dict = self.tokenizer.encode_plus(text=p1, text_pair=p2, max_length=self.max_len,
                                                      return_tensors='pt', add_special_tokens=True, padding='max_length')  # input_ids, token_type_ids, attention_mask
                input_ids_row.append(res_dict['input_ids'])
                attention_mask_row.append(res_dict['attention_mask'])
                token_type_ids_row.append(res_dict['token_type_ids'])

            if len(d_para) < self.max_para_d:
                for j in range(len(d_para), self.max_para_d):
                    input_ids_row.append(torch.zeros([1, self.max_len], dtype=torch.int64))
                    attention_mask_row.append(torch.zeros([1, self.max_len], dtype=torch.int64))
                    token_type_ids_row.append(torch.zeros([1, self.max_len], dtype=torch.int64))

            assert (len(input_ids_row) == self.max_para_d)
            assert (len(attention_mask_row) == self.max_para_d)
            assert (len(token_type_ids_row) == self.max_para_d)

            input_ids_row = torch.cat(input_ids_row, dim=0).unsqueeze(0)
            attention_mask_row = torch.cat(attention_mask_row, dim=0).unsqueeze(0)
            token_type_ids_row = torch.cat(token_type_ids_row, dim=0).unsqueeze(0)

            input_ids_item.append(input_ids_row)
            attention_mask_item.append(attention_mask_row)
            token_type_ids_item.append(token_type_ids_row)

        if len(q_para) < self.max_para_q:
            for i in range(len(q_para), self.max_para_q):
                input_ids_row = []  #
                attention_mask_row = []
                token_type_ids_row = []
                for j in range(self.max_para_d):
                    input_ids_row.append(torch.zeros([1, self.max_len], dtype=torch.int64))
                    attention_mask_row.append(torch.zeros([1, self.max_len], dtype=torch.int64))
                    token_type_ids_row.append(torch.zeros([1, self.max_len], dtype=torch.int64))

                input_ids_row = torch.cat(input_ids_row, dim=0).unsqueeze(0)
                attention_mask_row = torch.cat(attention_mask_row, dim=0).unsqueeze(0)
                token_type_ids_row = torch.cat(token_type_ids_row, dim=0).unsqueeze(0)

                input_ids_item.append(input_ids_row)
                attention_mask_item.append(attention_mask_row)
                token_type_ids_item.append(token_type_ids_row)


        assert (len(input_ids_item) == self.max_para_q)
        assert (len(attention_mask_item) == self.max_para_q)
        assert (len(token_type_ids_item) == self.max_para_q)

        input_ids_item = torch.cat(input_ids_item, dim=0)
        token_type_ids_item = torch.cat(token_type_ids_item, dim=0)
        attention_mask_item = torch.cat(attention_mask_item, dim=0)

        return q_id, d_id, input_ids_item, token_type_ids_item, attention_mask_item

def segment_to_para(text, para_max_len): #255
    paras = []
    text = text.strip()
    sentences = re.split('(。|；|，|！|？|、)', text)
    para = ''
    for sen in sentences:
        if len(sen) == 0 or len(sen) > para_max_len:
            continue

        if len(para) + len(sen) >= para_max_len:
            paras.append(para)
            para = ''
        para += sen

    if len(para) > 0:
        paras.append(para)
    return paras


def get_doc(candidates_path, text1_idx, text2_idx,data_type):
    file_path = os.path.join(candidates_path, text1_idx, text2_idx+'.json')
    doc = ''
    with open(file_path, mode='r', encoding='utf-8')as f:
        js_dict = json.load(f)
        if data_type=='elam':
            doc += "".join(js_dict['doc'])
        elif data_type=='Lecard':
            doc += js_dict['ajjbqk']
    return doc


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

def read_pairs(candidates_path,label_path,data_type):
    data_pair_list = []
    folders = os.listdir(candidates_path)
    neg_labels = {}
    with open(label_path, mode='r', encoding='utf-8')as f:
        line = f.readlines()[0]
        pos_labels = json.loads(line)
        for k in pos_labels.keys():
            for file in pos_labels[k]:
                if data_type=='elam':
                    data_pair_list.append((k, file, pos_labels[k][file] / 2))
                elif data_type=='Lecard':
                    data_pair_list.append((k, file, pos_labels[k][file] / 3))
                else:
                    print('error')
                    exit()
            pos_labels[k] = [str(i) for i in pos_labels[k]]

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
            data_pair_list.append((key, filename, 0))

    return data_pair_list



