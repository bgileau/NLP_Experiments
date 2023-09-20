import gensim
from gensim.utils import simple_preprocess
from gensim.corpora import Dictionary
from gensim.models import LdaMulticore

class LDA:
    def __init__(self):
        pass

    def tokenize_words(self, inputs):
        output = []
        for sentence in inputs:
            preprocess_output = simple_preprocess(sentence)
            output.append(preprocess_output)

        return output

    def remove_stopwords(self, inputs, stop_words):
        output = []
        for doc in inputs:
            temp_list = []
            for word in doc:
                if word not in stop_words:
                    temp_list.append(word)
            
            output.append(temp_list)
                    
        return output

    def create_dictionary(self, inputs):
        print(inputs)

        id2word = Dictionary(inputs)
        corpus = []
        for text in inputs:
            corpus.append(id2word.doc2bow(text))
        
        return id2word, corpus

    def build_LDAModel(self, id2word, corpus, num_topics=10):
        lda_model = LdaMulticore(id2word=id2word,
                                 corpus=corpus,
                                 num_topics=num_topics)
        return lda_model
