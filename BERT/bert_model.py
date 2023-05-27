import torch.nn as nn
import torch
from transformers import AutoModel, AutoTokenizer,AutoConfig, AdamW, get_linear_schedule_with_warmup

model_path = "/pretrain_model/bert_legal_criminal"

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