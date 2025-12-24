# train_classifiers.py
import random
import numpy as np
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

from svm_tf_idf import SVM_TF_IDF  
from svm_embedding import SVM_Embedding

# -------------------------
# Seed + Label mapping
# -------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

LABEL2ID = {
    "benign": 0,
    "naive": 1,
    "ignore": 2,
    "escape": 3,
    "combine": 4
}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

# -------------------------
# Dataset
# -------------------------
def build_dataset():
    ds = load_dataset("guychuk/open-prompt-injection", split="train")
    texts, labels = [], []
    for row in ds:
        instruction = row["instruction"]
        # benign
        texts.append(f"Instruction: {instruction}\nUser input: {row['normal_input']}")
        labels.append(LABEL2ID["benign"])
        # attack
        texts.append(f"Instruction: {instruction}\nUser input: {row['attack_input']}")
        labels.append(LABEL2ID[row["attack_type"]])
    return texts, labels

def split_data(X, y):
    X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.3, random_state=SEED, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=SEED, stratify=y_tmp)
    return X_train, X_val, X_test, y_train, y_val, y_test

def evaluate_model(model, X, y, name):
    y_pred = model.predict(X)
    print(f"\n=== {name} ===")
    print("Accuracy:", accuracy_score(y, y_pred))
    print(classification_report(
        y, y_pred, target_names=[ID2LABEL[i] for i in range(len(ID2LABEL))]
    ))

# -------------------------
# Main
# -------------------------
def main():
    print("Loading dataset...")
    X, y = build_dataset()
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    # ---- TF-IDF SVM ----
    # print("Training TF-IDF SVM...")
    # tfidf_model = SVM_TF_IDF()
    # tfidf_model.train(X_train, y_train)
    # print("Validation:")
    # print(tfidf_model.evaluate(X_val, y_val))
    # print("Test:")
    # print(tfidf_model.evaluate(X_test, y_test))
    # tfidf_model.save_model("svm_tfidf_5class.pkl")

    # # ---- Embedding SVM ----
    # print("Training Embedding SVM...")
    # emb_model = SVM_Embedding()
    # emb_model.train(X_train, y_train)
    # print("Validation:")
    # print(emb_model.evaluate(X_val, y_val))
    # print("Test:")
    # print(emb_model.evaluate(X_test, y_test))
    # emb_model.save_model("svm_embedding_5class.pkl")

    # ---- TF-IDF SVM ----
    tfidf_model = SVM_TF_IDF()
    tfidf_model.train(X_train, y_train)
    evaluate_model(tfidf_model, X_val, y_val, "TF-IDF SVM (Validation)")
    evaluate_model(tfidf_model, X_test, y_test, "TF-IDF SVM (Test)")
    tfidf_model.save_model("svm_tfidf_5class.pkl")

    # ---- Embedding SVM ----
    emb_model = SVM_Embedding()
    emb_model.train(X_train, y_train)
    evaluate_model(emb_model, X_val, y_val, "Embedding SVM (Validation)")
    evaluate_model(emb_model, X_test, y_test, "Embedding SVM (Test)")
    emb_model.save_model("svm_embedding_5class.pkl")

if __name__ == "__main__":
    main()
