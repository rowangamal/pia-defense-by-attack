import torch
import numpy as np
import joblib

from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.metrics import accuracy_score

from sentence_transformers import SentenceTransformer


class SVM_Embedding:
    def __init__(
        self,
        embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
        svm_params=None,
        device=None
    ):
        """
        Initializes SVM + Embedding pipeline.

        :param embedding_model_name: HuggingFace / SentenceTransformer model name
        :param svm_params: Dict of parameters for LinearSVC
        :param device: "cpu" or "cuda"
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.embedder = SentenceTransformer(embedding_model_name, device=self.device)

        self.svm_params = svm_params if svm_params else {
            "C": 1.0,
            "class_weight": "balanced"
        }

        # Wrap embedding function for sklearn pipeline
        self.embedding_transformer = FunctionTransformer(
            self._embed_texts,
            validate=False
        )

        self.model = Pipeline([
            ("embed", self.embedding_transformer),
            ("svm", LinearSVC(**self.svm_params))
        ])

    def _embed_texts(self, texts):
        """
        Converts list of texts to normalized embeddings.
        """
        embeddings = self.embedder.encode(
            texts,
            batch_size=32,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        return embeddings

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
