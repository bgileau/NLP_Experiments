import torch

torch.manual_seed(10)
import torch.nn as nn
from transformers import BertModel


class BERTClassifier(nn.Module):
    def __init__(self, num_classes):
        super(BERTClassifier, self).__init__()

        self.bertmodel = BertModel.from_pretrained('bert-base-uncased')

        self.dropout = nn.Dropout(0.5)
        
        self.linear = nn.Linear(768, num_classes)



    def forward(self, inputs, mask):
        bertmodel_outputs = self.bertmodel(inputs, attention_mask=mask)
        bertmodel_outputs = bertmodel_outputs[0][:, 0, :]
        bertmodel_outputs = self.dropout(bertmodel_outputs)
        
        output = self.linear(bertmodel_outputs)

        return output
