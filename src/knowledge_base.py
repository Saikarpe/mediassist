"""
Knowledge base built from the MedQuAD (Medical Question Answering) dataset.

Two learned components:
  1. MedicalRetriever      - TF-IDF vector index over questions, returns the best
                             matching Q&A pairs for a user query (the "R" in RAG).
  2. QuestionTypeClassifier - TF-IDF + Logistic Regression model that predicts the
                             question type (symptoms, treatment, causes, ...).
"""
import os
import re

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import linear_kernel

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(ROOT, "data", "medquad.csv")
MODEL_DIR = os.path.join(ROOT, "models")


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"\?+", "?", text)
    text = re.sub(r"[^a-z0-9?\s\-']", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def load_medquad(path: str = DATA_PATH) -> pd.DataFrame:
    """Load MedQuAD, drop empty rows, duplicate questions and ultra-rare classes."""
    df = pd.read_csv(path)
    df = df.dropna(subset=["Question", "Answer"])
    df["Question"] = df["Question"].str.replace(r"\s*\?\s*\?$", "?", regex=True).str.strip()
    df = df.drop_duplicates(subset=["Question"]).reset_index(drop=True)
    counts = df["qtype"].value_counts()
    df = df[df["qtype"].isin(counts[counts >= 10].index)].reset_index(drop=True)
    return df


class MedicalRetriever:
    """Hybrid word + character n-gram TF-IDF retriever over MedQuAD questions."""

    def __init__(self, use_char_ngrams: bool = True):
        self.use_char_ngrams = use_char_ngrams
        self.word_vec = TfidfVectorizer(ngram_range=(1, 2), stop_words="english",
                                        sublinear_tf=True, min_df=1)
        self.char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                                        sublinear_tf=True, min_df=2)
        self.df = None
        self.matrix = None

    def _encode(self, texts, fit=False):
        texts = [clean_text(t) for t in texts]
        if fit:
            w = self.word_vec.fit_transform(texts)
            if not self.use_char_ngrams:
                return w
            c = self.char_vec.fit_transform(texts)
        else:
            w = self.word_vec.transform(texts)
            if not self.use_char_ngrams:
                return w
            c = self.char_vec.transform(texts)
        # equal weighting of both views, rows re-normalised for cosine similarity
        m = hstack([w * 0.6, c * 0.4]).tocsr()
        norms = np.sqrt(m.multiply(m).sum(axis=1)).A1
        norms[norms == 0] = 1.0
        return m.multiply(1 / norms[:, None]).tocsr()

    def fit(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True)
        self.matrix = self._encode(self.df["Question"].tolist(), fit=True)
        return self

    def search(self, query: str, top_k: int = 3):
        q = self._encode([query])
        scores = linear_kernel(q, self.matrix).ravel()
        idx = np.argsort(-scores)[:top_k]
        return [
            {
                "question": self.df.at[i, "Question"],
                "answer": self.df.at[i, "Answer"],
                "qtype": self.df.at[i, "qtype"],
                "score": round(float(scores[i]), 3),
                "index": int(i),
            }
            for i in idx
        ]


class QuestionTypeClassifier:
    """Predicts what kind of medical question is being asked."""

    def __init__(self):
        self.vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1)
        self.clf = LogisticRegression(max_iter=2000, C=5.0, class_weight="balanced")

    def fit(self, questions, labels):
        X = self.vec.fit_transform([clean_text(q) for q in questions])
        self.clf.fit(X, labels)
        return self

    def predict(self, questions):
        return self.clf.predict(self.vec.transform([clean_text(q) for q in questions]))

    def predict_one(self, question: str):
        X = self.vec.transform([clean_text(question)])
        proba = self.clf.predict_proba(X)[0]
        best = int(np.argmax(proba))
        return self.clf.classes_[best], float(proba[best])


def save_models(retriever, classifier):
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(retriever, os.path.join(MODEL_DIR, "retriever.joblib"))
    joblib.dump(classifier, os.path.join(MODEL_DIR, "qtype_classifier.joblib"))


def load_models():
    """Load trained models; train on the fly if they don't exist yet."""
    r_path = os.path.join(MODEL_DIR, "retriever.joblib")
    c_path = os.path.join(MODEL_DIR, "qtype_classifier.joblib")
    if os.path.exists(r_path) and os.path.exists(c_path):
        return joblib.load(r_path), joblib.load(c_path)
    df = load_medquad()
    retriever = MedicalRetriever().fit(df)
    classifier = QuestionTypeClassifier().fit(df["Question"], df["qtype"])
    save_models(retriever, classifier)
    return retriever, classifier
