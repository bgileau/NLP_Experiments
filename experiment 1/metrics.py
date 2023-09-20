import numpy as np
from sklearn.metrics import recall_score, precision_score, f1_score, confusion_matrix, roc_auc_score

class Metrics(object):
	def __init__(self):
		pass

	def accuracy(self, y, y_hat):
		y = np.array(y)

		true_negative = np.sum((y_hat == 0) & (y == 0))
		false_negative = np.sum((y_hat == 0) & (y == 1))
		true_positive = np.sum((y_hat == 1) & (y == 1))
		false_positive = np.sum((y_hat == 1) & (y == 0))

		total = true_positive + true_negative + false_positive + false_negative

		return (true_positive + true_negative) / total

	def recall(self, y, y_hat, average='macro'):
		return recall_score(y, y_hat, average=average)

	def precision(self, y, y_hat, average='macro'):
		return precision_score(y, y_hat, average=average)

	def f1_score(self, y, y_hat, average='macro'):
		return f1_score(y, y_hat, average=average)

	def roc_auc_score(self, y, y_hat, average='macro'):
		return roc_auc_score(y, y_hat, average=average)
    
	def confusion_matrix(self, y, y_hat):
		return confusion_matrix(y, y_hat)
