# Turkish Smart Home Command Recognition

This project implements a Turkish smart home command recognition system designed to understand user intent across varying linguistic expressions. It was developed as a term project for the CSE 431 Natural Language Processing with Machine Learning course.

## Project Overview
* The system allows users to control smart home devices, such as lighting, climate control, and multimedia, using natural language commands.
* It employs Word2Vec-based word embeddings to capture the semantic similarity between different command phrases.
* Command matching is performed using cosine similarity to evaluate user inputs against known commands.

## Dataset Creation
* Core commands were defined for common actions, including lighting control (on/off, brightness), television control, climate control, and alarms.
* AI-based language models (such as ChatGPT) were utilized to generate natural Turkish paraphrases for each core command to increase linguistic diversity.
* The final training corpus consists of 71 lowercased and tokenized command sentences.

## Methodology & Training
* **Base Model:** The project utilizes a pre-trained Turkish Word2Vec model by Akoksal et al.
* **Unigram Fine-tuning:** The model was fine-tuned on the custom smart home corpus for 25 epochs with a window size of 5.
* **Bigram Fine-tuning:** Frequently co-occurring word pairs (multiword expressions) were incorporated using Gensim's Phrases module to capture phrase-level semantics.

## Sentence Embeddings & Decision Making
Each command sentence was converted into a single fixed-length vector using two distinct approaches:
* **Mean Pooling:** Computing the arithmetic mean of the Word2Vec embeddings of all words in the sentence.
* **TF-IDF Weighted Mean:** Weighting word embeddings by their TF-IDF scores to prioritize informative words.

To recognize user intent, the system computes the cosine similarity between the input vector and all stored command vectors. A confidence threshold of 0.75 is applied; if the similarity exceeds this threshold, the command is accepted.

## Evaluation & Results
The models were evaluated using a balanced test dataset consisting of 10 valid smart home commands and 10 invalid/out-of-domain commands. 

* **Precision:** 0.82
* **Recall:** 0.90
* **F1-Score:** 0.86

These evaluation metrics remained consistent across both Unigram and Bigram Word2Vec models, as well as for both Mean and TF-IDF pooling methods.
