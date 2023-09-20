import torch
torch.manual_seed(10)
import torch.nn.functional as F
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, w2vmodel, num_classes, window_sizes=(1,2,3,5)):
        super(CNN, self).__init__()
        weights = w2vmodel.wv # use this to initialize the embedding layer
        EMBEDDING_SIZE = 500  # Use this to set the embedding_dim in embedding layer
        NUM_FILTERS = 10      # Number of filters in CNN
        ##############################
        weight_for_embedding = torch.FloatTensor(weights.vectors)
        self.embedding = nn.Embedding.from_pretrained(weight_for_embedding)

        self.convolutions = nn.ModuleList()
        for window in window_sizes:
            convolution_layer = nn.Conv2d(1, NUM_FILTERS, (window, EMBEDDING_SIZE), padding=(window - 1, 0))
            self.convolutions.append(convolution_layer)

        self.linear = nn.Linear(NUM_FILTERS * len(window_sizes), num_classes)


    def forward(self, x):
        ##############################
        # print(x.shape)
        embedded_x = self.embedding(x)
        # print(x.shape)
        embedded_x = embedded_x.unsqueeze(1)
        
        convolution_x = []
        for conv in self.convolutions:
            convolution_x.append(F.tanh(conv(embedded_x)).squeeze(3))
        
        max_pooled_x = []
        for i in convolution_x:
            max_pooled_x.append(F.max_pool1d(i, i.size(2)).squeeze(2))
        max_pooled_x = torch.cat(max_pooled_x, 1)

        logit_x = self.linear(max_pooled_x)
        prob_x = F.softmax(logit_x, dim=1)
        
        return prob_x
