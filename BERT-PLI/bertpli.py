import torch.nn as nn
import torch
from transformers import BertModel
from transformers import BertConfig
from rnn_attention import RNNAttention

class BertPli(nn.Module):
    def __init__(self, model_path, max_para_q, max_para_d, max_len, criterion,data_type):
        super(BertPli, self).__init__()
        self.max_para_q = max_para_q
        self.max_para_d = max_para_d
        self.max_len = max_len
        self.criterion = criterion
        self.data_type=data_type

        # stage 2
        self.config = BertConfig.from_pretrained(model_path, return_dict=False)
        self.bert = BertModel.from_pretrained(model_path, config=self.config)
        self.maxpool = nn.MaxPool2d(kernel_size=(1, self.max_para_d))

        # stage 3
        self.attn = RNNAttention(max_para_q=self.max_para_q,data_type=self.data_type)

    def forward(self, data, label, mode='train'):
        input_ids, attention_mask, token_type_ids = data['input_ids'], data['attention_mask'], data['token_type_ids']
        batch_size = input_ids.size()[0]

        last_hidden_state, cls = self.bert(input_ids=input_ids.view(-1, self.max_len),
                                            attention_mask=attention_mask.view(-1, self.max_len),
                                            token_type_ids=token_type_ids.view(-1, self.max_len))

        feature = cls

        feature = feature.view(self.max_para_q, self.max_para_d, -1)

        feature = feature.permute(2, 0, 1)

        feature = feature.unsqueeze(0)
        max_out = self.maxpool(feature)
        max_out = max_out.squeeze()
        max_out = max_out.transpose(0, 1)
        max_out = max_out.unsqueeze(0)
        max_out = max_out.view(batch_size, self.max_para_q, -1)

        score = self.attn(max_out)


        loss = self.criterion(score, label)

        if mode == 'eval' or mode == 'test':
            return score, loss

        return loss