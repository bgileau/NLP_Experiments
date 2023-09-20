import torch

torch.manual_seed(10)
import torch.nn as nn
from transformers import BertModel


class SequenceLabeling(nn.Module):
    def __init__(self, num_classes):
        super(SequenceLabeling, self).__init__()

        self.bertmodel = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.5)
        self.linear = nn.Linear(768, num_classes)

    def forward(self, inputs, mask, token_type_ids):
        bertmodel_output = self.bertmodel(inputs, attention_mask=mask, token_type_ids=token_type_ids)[0]
        bertmodel_output = self.dropout(bertmodel_output)
        
        output = self.linear(bertmodel_output)
        output = torch.relu(output)
        return output
