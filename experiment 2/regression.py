import pandas as pd
import numpy as np
from scipy.special import softmax
from sklearn.preprocessing import OneHotEncoder

from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)


class Regression(object):
    def __init__(self):
        self.oh = OneHotEncoder()

    def onehot(self, labels):
        labels = np.array(labels)
        
        labels_2d = np.array(labels).reshape(-1, 1)
        # print(labels.shape, labels_2d.shape)
        encoded = self.oh.fit_transform(labels_2d)
        return encoded.toarray()

    def gradient(self, X, Y, W):
        N = X.shape[0]
        P = softmax(np.dot(X, W), axis=1)
        mu = 0.01  # Do not change momentum value.
        gradient = (-1/N) * np.dot(X.T, (Y - P)) + (2 * mu * W)
        return gradient

    def gradient_descent(self, X, Y, epochs=10, eta=0.1):
        W = np.zeros((X.shape[1], Y.shape[1]))

        for i in range(epochs):
            gradient_result_iterative = self.gradient(X, Y, W)
            W = W - eta * gradient_result_iterative

        return W # final result from gradient descent
        

    def fit(self, data, labels):
        X = np.asarray(data)
        Y_onehot = self.onehot(labels)
        self.W = self.gradient_descent(X, Y_onehot)

    def predict(self, data):
        step1_dot_prod = np.dot(data, self.W)
        P = softmax(step1_dot_prod, axis=1)

        predictedLabels = np.argmax(P, axis=1)

        return predictedLabels
        
