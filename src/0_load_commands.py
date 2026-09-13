"""
Name Surname : Merve Adali
Student No#  : 211805049
Course       : CSE 431
Homework     : Term Project

General Comments:
This project implements a Turkish smart home command
recognition system using Word2Vec embeddings, cosine
similarity, and TF-IDF weighted sentence vectors.
"""

import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "TR_Commands.xlsx")

print("Excel path:", DATA_PATH)

df = pd.read_excel(DATA_PATH)

commands = df.iloc[:, 1].dropna().str.lower().tolist()

print("Total command:", len(commands))
print(commands[:10])
