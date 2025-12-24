import torch
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import joblib
from sentence_transformers import SentenceTransformer



class SentenceEmbeddingTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, embedding_model_name="sentence-transformers/all-MiniLM-L6-v2", device=None):
        self.embedding_model_name = embedding_model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None

    def fit(self, X, y=None):
        self.model = SentenceTransformer(self.embedding_model_name, device=self.device)
        return self

    def transform(self, X):
        return self.model.encode(
            X,
            batch_size=32,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False
        )



class SVM_Embedding:
    def __init__(self, embedding_model_name="sentence-transformers/all-MiniLM-L6-v2", svm_params=None, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.svm_params = svm_params if svm_params else {"C": 1.0, "class_weight": "balanced"}

        # embedding transformer
        self.embedding_transformer = SentenceEmbeddingTransformer(embedding_model_name, device=self.device)

        # Pipeline
        self.model = Pipeline([
            ("embed", self.embedding_transformer),
            ("svm", LinearSVC(**self.svm_params))
        ])

    def train(self, X, y):
        self.model.fit(X, y)
        print("Model training complete.")

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        return accuracy_score(y_test, y_pred)

    def save_model(self, filename):
        joblib.dump(self.model, filename)
        print(f"Model saved to {filename}")

    def load_model(self, filename):
        self.model = joblib.load(filename)
        print(f"Model loaded from {filename}")
