# -*- coding: utf-8 -*-
"""
Name Surname : Merve Adali
Student No#  : 211805049
Course       : CSE 431
Homework     : Term Project

Bigram (multiword) Word2Vec fine-tuning
"""

from pathlib import Path
from gensim.models import Word2Vec, KeyedVectors
from gensim.models.phrases import Phrases, Phraser

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"

TRAIN_FILE = DATA_DIR / "train_commands.txt"
PRETRAINED_MODEL = MODEL_DIR / "trmodel"
OUTPUT_MODEL = MODEL_DIR / "finetuned_bigram.model"

sentences = []
with open(TRAIN_FILE, encoding="utf-8") as f:
    for line in f:
        _, text = line.strip().split("\t")
        sentences.append(text.split())

phrases = Phrases(sentences, min_count=2, threshold=5)
bigram = Phraser(phrases)
sentences_bigram = [bigram[s] for s in sentences]

print("Example bigram sentence:", sentences_bigram[0])

base_vectors = KeyedVectors.load_word2vec_format(
    str(PRETRAINED_MODEL),
    binary=True
)

model = Word2Vec(
    vector_size=base_vectors.vector_size,
    window=5,
    min_count=1,
    workers=4
)

model.build_vocab(sentences_bigram)

common_words = set(model.wv.key_to_index).intersection(
    base_vectors.key_to_index
)

print("Common unigram words:", len(common_words))

for word in common_words:
    model.wv[word] = base_vectors[word]

model.train(
    sentences_bigram,
    total_examples=len(sentences_bigram),
    epochs=25
)

model.save(str(OUTPUT_MODEL))

print("Bigram Word2Vec model saved successfully.")
