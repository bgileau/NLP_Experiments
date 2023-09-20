import torch

torch.manual_seed(10)
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class LSTM(nn.Module):
    def __init__(self, vocab, num_classes):
        super(LSTM, self).__init__()

        self.embed_len = 50
        self.embedding_layer = nn.Embedding(len(vocab), self.embed_len)

        self.hidden_dim = 100
        self.n_layers = 1 # why is a var needed then
        self.lstm = nn.LSTM(self.embed_len, self.hidden_dim, num_layers=self.n_layers, bidirectional=True, batch_first=True)

        self.dropout = nn.Dropout(0.5)

        self.linear = nn.Linear(self.hidden_dim*2, num_classes)



    def forward(self, inputs, inputs_len):
        # print(inputs)
        embedding = self.embedding_layer(inputs)
        embedding_packed = pack_padded_sequence(embedding, inputs_len, batch_first=True, enforce_sorted=False)

        output_packed, _ = self.lstm(embedding_packed)
        output_padded, _ = pad_packed_sequence(output_packed, batch_first=True)

        # print(output_packed)

        output_padded = self.dropout(output_padded)

        # print(output_padded)

        output_padded = output_padded[range(len(output_padded)), inputs_len - 1, :]

        output = self.linear(output_padded)

        return output
