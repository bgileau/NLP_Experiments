import torch
torch.manual_seed(10)
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class RNN(nn.Module):
    def __init__(self, vocab, num_classes):
        super(RNN, self).__init__()
        self.embed_len = 50  # embedding_dim default value for embedding layer
        self.hidden_dim = 75 # hidden_dim default value for rnn layer
        self.n_layers = 1    # num_layers default value for rnn
        #######
        vocab_size = len(vocab)
        self.embedding = nn.Embedding(vocab_size, self.embed_len)
        self.rnn = nn.RNN(self.embed_len, self.hidden_dim, self.n_layers, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(self.hidden_dim*2, num_classes)
        self.dropout = nn.Dropout(0.7) # decent value


    def forward(self, inputs, inputs_len):
        embeddings = self.embedding(inputs)
        packed_embeddings = pack_padded_sequence(embeddings, inputs_len, batch_first=True, enforce_sorted=False)
        packed_outputs, hidden = self.rnn(packed_embeddings)

        forward_hidden = hidden[0, :, :]
        reverse_hidden = hidden[1, :, :]
        concat_hiddens = torch.cat((forward_hidden, reverse_hidden), dim=1)

        concat_hiddens = self.dropout(concat_hiddens)
        output = self.linear(concat_hiddens)
        return output

