import torch.nn as nn
import torch
from transformers import AutoModel, AutoTokenizer,AutoConfig, AdamW, get_linear_schedule_with_warmup
from pytorch_pretrained_bert import BertModel
model_path = "/pretrain_model/bert_legal_criminal"
shao_model_path='/pretrain_model/shao_bert/pytorch_model.pkl'

class FirstModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = BertModel.from_pretrained(model_path)
        shao_model=torch.load(shao_model_path)['model']
        self.model.load_state_dict(shao_model,False)
        self.linear = nn.Linear(768, 1)

    def forward(self, sample):
        inputs_ids, inputs_masks = sample['input_ids'], sample['attention_mask']
        types_ids = sample['token_type_ids']
        outputs=self.model(input_ids=inputs_ids,  attention_mask=inputs_masks,token_type_ids=types_ids)
        pooled_output = outputs[1]
        score = self.linear(pooled_output)
        score = torch.sigmoid(score).squeeze(-1)
        return score