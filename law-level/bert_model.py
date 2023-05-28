import torch.nn as nn
import torch
from transformers import AutoModel, AutoTokenizer,AutoConfig, AdamW, get_linear_schedule_with_warmup

model_path = "/pretrain_model/bert_legal_criminal"

import torch.nn as nn

class Logic(nn.Module):
    def __init__(self):
        super(Logic, self).__init__()

    def Logic_OR(self, values):
        """
        values: tensor list
        :return:
        """
        output = min(values[0]+values[1],1.0)  # Łukasiewicz t-conorm
        # output = values[0] + values[1] - values[0] * values[1]  # product t-conorm
        # output = (values[0] + values[1] - 2 * values[0] * values[1]) / (1 - values[0] * values[1])  # Hamacher t-norm
        # output = (values[0] + values[1]) / (1 + values[0] * values[1])  # Einstein t-conorm

        return output

    def Logic_AND(self, values):
        """
        values: tensor list
        :return:
        """
        output = max(values[0] + values[1] - 1, 0)  # Łukasiewicz t-norm
        # output = values[0] * values[1]  # product t-conorm
        # output = (values[0] * values[1]) / (values[0] + values[1] - values[0] * values[1]) # Hamacher t-norm
        # output = (values[0] * values[1]) / (2 - (values[0] + values[1] - values[0] * values[1]))  # Einstein t-norm
        return output if output > 0 else 0

    def Logic_not(self, x):
        """
        :param x: feature
        :return:
        """
        return 1 - x

    def compare(self, op1, op2):
        """
        比较两个运算符的优先级,乘除运算优先级比加减高
        op1优先级比op2高返回True，否则返回False
        """
        return op1 in ["*"] and op2 in ["+"]

    def getvalue(self, num1, num2, operator):
        """
        根据运算符号operator计算结果并返回
        """
        if operator == "*":
            return self.Logic_AND([num1, num2])
        elif operator == "+":
            return self.Logic_OR([num1, num2])

    def process(self, data, opt):
        """
        opt出栈一个运算符，data出栈两个数值，进行一次计算，并将结果入栈data
        """
        operator = opt.pop()
        num2 = data.pop()
        num1 = data.pop()
        data.append(self.getvalue(num1, num2, operator))

    def calculate_logic(self, s):
        """
        计算字符串表达式的值,字符串中不包含空格
        """
        data = []
        opt = []
        i = 0
        symbo = [')', '(', '+', '*', '~']
        equl_index = s.find('=')
        while i < equl_index:
            if s[i] not in symbo:
                start = i
                while i + 1 < equl_index and s[i + 1] not in symbo:
                    i += 1
                data.append(float(s[start: i + 1]))
            elif s[i] == '~':
                start = i + 1
                while i + 1 < equl_index and s[i + 1] not in symbo:
                    i += 1
                num = 1 - float(s[start: i + 1])
                data.append(num)
            elif s[i] == ")":
                while opt[-1] != "(":
                    self.process(data, opt)
                opt.pop()
            elif not opt or opt[-1] == "(":
                opt.append(s[i])
            elif s[i] == "(" or self.compare(s[i], opt[-1]):
                opt.append(s[i])
            else:
                while opt and not self.compare(s[i], opt[-1]):
                    if opt[-1] == "(":
                        break
                    self.process(data, opt)
                opt.append(s[i])
            i += 1
        while opt:
            self.process(data, opt)
        return data.pop()

    def forward(self, rules):
        """
        :param ground: P(x)的值，list [ tensor, ... ]
        :param rules: 对应某个法条的rules, list [string, ...], 或者解析好的rule
        :return: score
        """
        """
        for rule in rules:
            先求not
            再求析取
            最后合取
        最后对rules中各个rule进行析取
        """
        scores = []

        for rule in rules:
            score = self.calculate_logic(rule)
            scores.append(score)
        if len(rules)==0:  #some case don't have rules
            return 0
        res=sum(scores)/len(scores)
        return res


class FirstModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_path)
        self.linear = nn.Linear(self.bert.config.hidden_size, 1)

    def forward(self, sample):
        inputs_ids, inputs_masks = sample['input_ids'], sample['attention_mask']
        types_ids = sample['token_type_ids']
        cls= self.bert(input_ids=inputs_ids,  attention_mask=inputs_masks,token_type_ids=types_ids).pooler_output
        score = self.linear(cls)
        score = torch.sigmoid(score).squeeze(-1)
        return score