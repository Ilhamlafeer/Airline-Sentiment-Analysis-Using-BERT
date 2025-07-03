✈️ Airline Tweet Sentiment Classification with BERT
This repository contains a complete workflow for fine-tuning a pretrained BERT model (bert-base-uncased) for a multi-class text classification task on the Airline Tweet Sentiment dataset. The model predicts whether a tweet expresses a positive, neutral, or negative sentiment toward an airline.

📌 Features
Dataset preprocessing and label encoding using pandas and sklearn
Hugging Face Datasets for efficient data management and splits
BERT tokenizer with padding and truncation
Fine-tuning using BertForSequenceClassification and Trainer API
Evaluation metrics: loss, accuracy (can be extended to F1, confusion matrix)
Trained and tested on Google Colab with GPU acceleration
Model and tokenizer saving using Hugging Face's .save_pretrained()

🛠️ Tech Stack
Python
PyTorch
Hugging Face Transformers
Google Colab (with GPU)
Scikit-learn
Pandas

🔄 From Traditional NLP to Deep Learning with PyTorch
This project builds upon previous work where traditional NLP techniques such as TF-IDF vectorization combined with classical machine learning models (like Naive Bayes and Random Forest) were used for sentiment classification on the same dataset.
In this repository, I have fine-tuned a pretrained BERT model using PyTorch through the Hugging Face Transformers library.

📂 Dataset
Source: Kaggle - Airline Tweet Sentiment
Classes: positive, neutral, negative
Text Format: Short tweets mentioning airline services
