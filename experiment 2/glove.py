import numpy as np
import os
from scipy import spatial


class Glove(object):
    def __init__(self):
        pass

    def load_glove_model(self, glove_file=os.path.join("data/glove.6B.50d.txt")):
        print("Loading Glove Model")
        glove_model = {}
        with open(glove_file, "r", encoding="utf-8") as f:
            for line in f:
                split_line = line.split()
                word = split_line[0]
                embedding = np.array(split_line[1:], dtype=np.float64)
                glove_model[word] = embedding
        print(f"{len(glove_model)} words loaded!")
        return glove_model

    def find_similar_word(self, model, emmbeddings):
        nearest = sorted(
            model.keys(),
            key=lambda word: spatial.distance.euclidean(model[word], emmbeddings),
        )
        return nearest

    def transform(self, model, data, dimension=50):
        transformed_features = np.zeros((len(data), dimension))
        idx = 0
        for sentence in data:
            embedding_list = []
            sentence_arr = sentence.split()
            for word in sentence_arr:
                try:
                    embedding_list.append(model[word])
                except:
                    continue
            if len(embedding_list) > 0: # check if empty
                transformed_features[idx] = np.mean(embedding_list, axis=0)
            idx += 1
        return transformed_features
