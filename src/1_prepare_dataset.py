# -*- coding: utf-8 -*-
"""
Name Surname : Merve Adali
Student No#  : 211805049
Course       : CSE 431
Homework     : Term Project

General Comments:
Fine-tuning a pre-trained Turkish Word2Vec model
using smart home command sentences.
"""

import pandas as pd
import numpy as np
from pathlib import Path

from gensim.models import Word2Vec, KeyedVectors
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

excel_path = DATA_DIR / "TR_Commands.xlsx"
output_path = DATA_DIR / "train_commands.txt"

print("Excel path:", excel_path)

df = pd.read_excel(excel_path)
commands = df.iloc[:, 1].dropna().str.lower()

INTENTS = {
    "LIGHT_ON": ["aç", "yak"],
    "LIGHT_OFF": ["kapat", "söndür"],
    "LIGHT_UP": ["arttır"],
    "LIGHT_DOWN": ["kıs", "azalt", "düşür"],
    "COLOR": ["kırmızı", "mavi", "yeşil"],
    "PLUG": ["priz"],
    "CLIMATE": ["klima", "ısıt", "soğut", "sıcaklık", "kombi"],
    "STATUS": ["kaç derece"],
    "FAN": ["fanı"],
    "TV": ["tv", "televizyon"],
    "MEDIA": ["müzik", "multimedya"],
    "CURTAIN": ["perde", "panjur"],
    "ALARM": ["alarm"],
    "YESNO": ["evet", "hayır"],
    "SCENE": ["zamanı", "geldim", "çıkıyorum", "günaydın", "iyi geceler", "senaryo"]
}

def assign_intent(cmd):
    for intent, keywords in INTENTS.items():
        for k in keywords:
            if k in cmd:
                return intent
    return "UNKNOWN"

with open(output_path, "w", encoding="utf-8") as f:
    for cmd in commands:
        intent = assign_intent(cmd)
        f.write(f"{intent}\t{cmd}\n")

print("train_commands.txt has been successfully created")
print("Total number of samples:", len(commands))

