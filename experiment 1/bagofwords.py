import numpy as np
from sklearn.preprocessing import OneHotEncoder
from concurrent.futures import ProcessPoolExecutor  # You may not need to use this
from functools import partial # You may not need to use this

class OHE_BOW(object): 
	def __init__(self):
		self.vocab_size = None
		self.oh = OneHotEncoder()

	def split_text(self, data):
		data_split = []
		for sentence in data:
			temp_list = []
			sentence_list = sentence.split(" ")
			for word in sentence_list:
				if len(word) > 0: # needed to pass local
					temp_list.append(word)
			data_split.append(temp_list)

		return data_split

	def flatten_list(self, data):
		data_split = []
		for list_of_words in data:
			for word in list_of_words:
				data_split.append(word)

		return np.array(data_split)

	def fit(self, data):
		flat_list = np.unique(self.flatten_list(self.split_text(data)))

		self.vocab_size = np.shape(flat_list)[0]

		final_X = np.reshape(flat_list, (self.vocab_size, 1))

		self.oh.fit(final_X)


	def onehot(self, words):
		words = np.array(words).reshape(-1, 1)
		onehotencoded = self.oh.transform(words).toarray()
		return onehotencoded

	def oneHotEncoding(self, row):
		try:
			return self.onehot(row).sum(axis=0)
		except:
			return np.zeros(self.vocab_size)    

	def transform(self, data):
		bow = []
		max_length = 0
		for row in data:
			row_list = row.split(" ")
			encoded = self.oneHotEncoding(row_list)
			bow.append(encoded)
			if len(encoded) > max_length:
				max_length = len(encoded)

		for i in range(len(bow)):
			padding = max_length - len(bow[i])
			bow[i] = np.pad(bow[i], (0, padding))

		return np.array(bow)