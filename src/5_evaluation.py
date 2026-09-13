# -*- coding: utf-8 -*-
"""
Name Surname : Merve Adali
Student No#  : 211805049
Course       : CSE 431 
Homework     : Term Project

Evaluation with Precision, Recall, F1
for unigram & bigram Word2Vec models
"""

import numpy as np
from pathlib import Path
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"

TRAIN_TXT = DATA_DIR / "train_commands.txt"

commands = []
with open(TRAIN_TXT, encoding="utf-8") as f:
    for line in f:
        _, cmd = line.strip().split("\t")
        commands.append(cmd)

test_commands = [
    ("lambayı aç", 1),
    ("ışığı yak", 1),
    ("televizyonu kapat", 1),
    ("klimayı aç", 1),
    ("odayı ısıt", 1),
    ("müziği durdur", 1),
    ("perdeyi kapat", 1),
    ("fanı aç", 1),
    ("alarmı kur", 1),
    ("günaydın", 1),
    ("arabayı çalıştır", 0),
    ("bahçeyi sula", 0),
    ("kapıyı kilitle", 0),
    ("interneti kapat", 0),
    ("yemeği pişir", 0),
    ("asansörü çağır", 0),
    ("kamerayı izle", 0),
    ("pencereyi aç", 0),
    ("ışıkları tamir et", 0),
    ("çamaşır yıka", 0),
]

def evaluate(model_type):
    """
    model_type: 'unigram' or 'bigram'
    """

    # Select correct model file
    if model_type == "unigram":
        model_path = MODEL_DIR / "finetuned.model"
    elif model_type == "bigram":
        model_path = MODEL_DIR / "finetuned_bigram.model"
    else:
        raise ValueError("model_type must be 'unigram' or 'bigram'")

    # Load Word2Vec model
    model = Word2Vec.load(str(model_path))

    # Load sentence vectors
    mean_vectors = np.load(MODEL_DIR / f"{model_type}_mean.npy")
    tfidf_vectors = np.load(MODEL_DIR / f"{model_type}_tfidf.npy")

    # TF-IDF for weighting
    tfidf = TfidfVectorizer()
    tfidf.fit(commands)
    idf = dict(zip(tfidf.get_feature_names_out(), tfidf.idf_))

    # Mean embedding
    def mean_embedding(sentence):
        vecs = [model.wv[w] for w in sentence.split() if w in model.wv]
        if not vecs:
            return np.zeros(model.vector_size)
        return np.mean(vecs, axis=0)

    # TF-IDF weighted embedding
    def tfidf_embedding(sentence):
        vecs, weights = [], []
        for w in sentence.split():
            if w in model.wv and w in idf:
                vecs.append(model.wv[w])
                weights.append(idf[w])
        if not vecs:
            return np.zeros(model.vector_size)
        return np.average(vecs, axis=0, weights=weights)

    results = {}

    for method_name, vectors, emb_func in [
        ("MEAN", mean_vectors, mean_embedding),
        ("TF-IDF", tfidf_vectors, tfidf_embedding),
    ]:
        y_true, y_pred = [], []

        for cmd, label in test_commands:
            input_vec = emb_func(cmd)
            sims = cosine_similarity([input_vec], vectors)[0]
            pred = 1 if np.max(sims) >= 0.75 else 0

            y_true.append(label)
            y_pred.append(pred)

        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)

        results[method_name] = (precision, recall, f1)

    return results

for model_type in ["unigram", "bigram"]:
    print(f"\n {model_type.upper()} RESULTS")
    results = evaluate(model_type)
    for method, scores in results.items():
        print(
            f"{method:<7} "
            f"Precision={scores[0]:.2f} "
            f"Recall={scores[1]:.2f} "
            f"F1={scores[2]:.2f}"
        )
