import numpy as np
from sklearn.naive_bayes import MultinomialNB, GaussianNB


class MNB(object):
	def __init__(self):
		self.clf = MultinomialNB()

	def fit(self, data, y):
		self.clf.fit(data, y)

	def predict(self, data):
		return self.clf.predict(data)


class GNB(object):
	def __init__(self):
		self.clf = GaussianNB()

	def fit(self, data, y):
		self.clf.fit(data, y)

	def predict(self, data):
		return self.clf.predict(data)
