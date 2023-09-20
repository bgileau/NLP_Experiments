import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

class BagOfWords(object):
	def __init__(self):
		self.vectorizer = CountVectorizer()
		

	def fit(self, data):
		self.vectorizer.fit(data)

	def transform(self, data):
		x = self.vectorizer.transform(data).toarray()
		return x


class TfIdf(object):
	def __init__(self):
		self.vectorizer = TfidfVectorizer()


	def fit(self, data):
		self.vectorizer.fit(data)


	def transform(self, data):
		x = self.vectorizer.transform(data).toarray()
		return x