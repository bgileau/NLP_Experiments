import numpy as np
import torch

torch.manual_seed(10)
import torch.nn.functional as F
import torch.nn as nn


class Word2Vec(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.vocabulary_size = 0

    def tokenize(self, data):
        tokens = [sentence.strip().split(" ") for sentence in data]

        return np.asarray(tokens)
        


    def create_vocabulary(self, tokenized_data):
        # print(tokenized_data)
        # words_unique = set(np.array(tokenized_data).flatten()) # np breaking auto grader?
        words_unique = []
        for sublist in tokenized_data:
            for word in sublist:
                words_unique.append(word)
        words_unique = set(words_unique)

        words_unique_sorted = sorted(words_unique)
        # print(words_unique_sorted)

        for i, word in enumerate(words_unique_sorted):
            self.word2idx[word] = i
            self.idx2word[i] = word
            self.vocabulary_size += 1



    def skipgram_embeddings(self, tokenized_data, window_size=2):
        source_tokens = []
        target_tokens = []

        # print(tokenized_data)

        for sentence in tokenized_data:
            for i, word in enumerate(sentence):
                
                start = max(0, i - window_size)
                end = min(len(sentence), i + window_size + 1)

                for j in range(start, end):
                    if j != i: # ignore same idx
                        source_tokens.append([self.word2idx[word]])
                        target_tokens.append([self.word2idx[sentence[j]]])

        # print(source_tokens)
        # print(target_tokens)
        return source_tokens, target_tokens
    

    def cbow_embeddings(self, tokenized_data, window_size=2):
        source_tokens = []
        target_tokens = []

        # print(tokenized_data)

        for sentence in tokenized_data:
            for i, word in enumerate(sentence):
                target_tokens.append([self.word2idx[word]])
                
                context_tokens = []
                start = max(0, i - window_size)
                end = min(len(sentence), i + window_size + 1)

                for j in range(start, end):
                    if j != i: # ignore same idx
                        context_tokens.append(self.word2idx[sentence[j]])
                        # print(f"Appending: {i} | {sentence[j]}")
                source_tokens.append(context_tokens)

        # print(source_tokens)
        # print(target_tokens)


        return source_tokens, target_tokens
        


class SkipGram_Model(nn.Module):
    def __init__(self, vocab_size: int):
        super(SkipGram_Model, self).__init__()
        self.embedding_dim = 300
        self.max_norm = 1
        self.vocab_size = vocab_size

        self.nn_embedding = nn.Embedding(vocab_size, self.embedding_dim, max_norm=self.max_norm)
        self.nn_linear = nn.Linear(self.embedding_dim, vocab_size) # pred final word?


    def forward(self, inputs):
        embedding_return = self.nn_embedding(inputs)
        # embedding_return = torch.mean(embedding_return, dim=0)

        output = self.nn_linear(embedding_return)

        # output = output.reshape(1,self.vocab_size)

        return output


class CBOW_Model(nn.Module):
    def __init__(self, vocab_size: int):
        super(CBOW_Model, self).__init__()
        self.embedding_dim = 300
        self.max_norm = 1
        self.vocab_size = vocab_size

        self.nn_embedding = nn.Embedding(vocab_size, self.embedding_dim, max_norm=self.max_norm)
        self.nn_linear = nn.Linear(self.embedding_dim, vocab_size) # pred final word?

    def forward(self, inputs):
        embedding_return = self.nn_embedding(inputs)
        embedding_return = torch.mean(embedding_return, dim=0)

        output = self.nn_linear(embedding_return)

        output = output.reshape(1,self.vocab_size)

        return output
