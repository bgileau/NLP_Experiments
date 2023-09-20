import re
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('stopwords')
class Preprocess(object):
	def __init__(self):
		pass

	def clean_text(self, text):
		# Step 1
		cleaned_text = BeautifulSoup(text)
		cleaned_text = cleaned_text.get_text()

		# Step 2
		cleaned_text = re.sub('^\s+|\W+|[0-9]|\s+$',' ',cleaned_text).strip()

		# # Step 3 is above too?
		# cleaned_text = cleaned_text.strip()

		# Step 4
		cleaned_text = cleaned_text.lower()

		# Step 5
		# print(cleaned_text)
		cleaned_text_tokens = word_tokenize(cleaned_text, language="english")

		stop_words = set(stopwords.words('english'))

		cleanted_text = []
		for word in cleaned_text_tokens:
			if word not in stop_words:
				cleanted_text.append(word)

		# print(cleaned_text)
		# Step 6
		cleaned_text = " ".join(cleanted_text)


		return cleaned_text

		# raise NotImplementedError

	def clean_dataset(self, data):
		# cleaned_data = []
		# for strings in data:
		# 	cleaned_data.append(self.clean_text(strings))

		cleaned_data = [self.clean_text(strings) for strings in data]
		return cleaned_data
		# raise NotImplementedError


def clean_wos(x_train, x_test):
	preprocess = Preprocess()
	cleaned_text_wos = preprocess.clean_dataset(x_train)
	cleaned_text_wos_test = preprocess.clean_dataset(x_test)

	return cleaned_text_wos, cleaned_text_wos_test
	# raise NotImplementedError
