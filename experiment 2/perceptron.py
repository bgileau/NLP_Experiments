import numpy as np
from sklearn.preprocessing import OneHotEncoder

from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

class Perceptron(object):
    def __init__(self):
        self.oh = OneHotEncoder()

    def onehot(self, Y):
        Y = np.array(Y)
        
        Y_2d = np.array(Y).reshape(-1, 1)
        encoded = self.oh.fit_transform(Y_2d).toarray()
        encoded[encoded == 0] = -1
        return encoded

    def perceptron(self, X, Y, epochs=10):
        alpha = 1
        print(X.shape, Y.shape)
        weight = np.zeros((X.shape[1], 1))
        for e in range(epochs):
            for i in range(X.shape[0]):
                if Y[i] * np.dot(X[i], weight) <= 0:
                    weight = weight + (alpha * Y[i] * X[i].reshape(-1, 1))

        return weight
                

    def fit(self, data, labels):

        bias_ones = np.ones((len(data), 1))
        X = np.hstack((data, bias_ones))
        Y = self.onehot(labels)

        self.classes = Y.shape[1]
        self.weights = np.zeros((X.shape[1], Y.shape[1]))

        for i in range(Y.shape[1]):
            W = self.perceptron(X, Y[:, i])
            self.weights[:, i] = W[:, 0]

    def predict(self, data):
        bias_ones = np.ones((len(data), 1)) # fit adds bias, so we add it too
        data = np.hstack((data, bias_ones))
        dot_prod = np.dot(data, self.weights)

        predictedLabels = np.argmax(dot_prod, axis=1)

        return predictedLabels
