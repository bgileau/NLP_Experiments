from sklearn.linear_model import SGDClassifier


class SVM(object):
    def __init__(self, random_seed=None):
        self.clf = SGDClassifier(loss="hinge", random_state=random_seed)

    def fit(self, data, labels):
        self.clf.fit(data, labels)

    def predict(self, data):
        predictedLabels = self.clf.predict(data)
        return predictedLabels 