from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
import joblib

class SVM_TF_IDF:
    def __init__(self, tfidf_params=None, svm_params=None):
        """
        Initializes the SVM + TF-IDF Pipeline.
        :param tfidf_params: Dict of parameters for TfidfVectorizer
        :param svm_params: Dict of parameters for LinearSVC
        """
        self.tfidf_params = tfidf_params if tfidf_params else {'stop_words': 'english'}
        self.svm_params = svm_params if svm_params else {'C': 1.0}

        self.model = Pipeline([
            ('tfidf', TfidfVectorizer(**self.tfidf_params)),
            ('svm', LinearSVC(**self.svm_params))
        ])

    def train(self, X, y):
        self.model.fit(X, y)
        print("Model training complete.")

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X_test, y_test):
        return self.model.score(X_test, y_test)

    def save_model(self, filename):
        joblib.dump(self.model, filename)
        print(f"Model saved to {filename}")