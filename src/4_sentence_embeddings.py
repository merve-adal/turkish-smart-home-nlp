# -*- coding: utf-8 -*-
"""
Name Surname : Merve Adali
Student No#  : 211805049
Course       : CSE 431
Homework     : Term Project

Sentence embeddings:
1) Mean
2) TF-IDF weighted mean
(works for unigram & bigram models)
"""

import numpy as np
from pathlib import Path
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"

TRAIN_TXT = DATA_DIR / "train_commands.txt"
W2V_MODEL = MODEL_DIR / "finetuned.model"
W2V_BIGRAM_MODEL = MODEL_DIR / "finetuned_bigram.model"

sentences = []

with open(TRAIN_TXT, encoding="utf-8") as f:
    for line in f:
        _, cmd = line.strip().split("\t")
        sentences.append(cmd)

tokenized = [s.split() for s in sentences]

def generate_embeddings(model_path, prefix):
    model = Word2Vec.load(str(model_path))
    dim = model.vector_size

    def mean_emb(tokens):
        vecs = [model.wv[w] for w in tokens if w in model.wv]
        return np.mean(vecs, axis=0) if vecs else np.zeros(dim)

    mean_vectors = np.array([mean_emb(t) for t in tokenized])
    np.save(MODEL_DIR / f"{prefix}_mean.npy", mean_vectors)

    tfidf = TfidfVectorizer()
    tfidf.fit(sentences)
    idf = dict(zip(tfidf.get_feature_names_out(), tfidf.idf_))

    def tfidf_emb(tokens):
        vecs, weights = [], []
        for w in tokens:
            if w in model.wv and w in idf:
                vecs.append(model.wv[w])
                weights.append(idf[w])
        return np.average(vecs, axis=0, weights=weights) if vecs else np.zeros(dim)

    tfidf_vectors = np.array([tfidf_emb(t) for t in tokenized])
    np.save(MODEL_DIR / f"{prefix}_tfidf.npy", tfidf_vectors)

    print(f"Embeddings saved for {prefix}")

generate_embeddings(W2V_MODEL, "unigram")
generate_embeddings(W2V_BIGRAM_MODEL, "bigram")
