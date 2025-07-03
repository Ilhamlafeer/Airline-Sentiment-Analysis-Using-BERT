## 🤖 Airline Tweet Sentiment Classification Using BERT
This project fine-tunes a pretrained **BERT (bert-base-uncased)** model to classify airline-related tweets as **positive**, **neutral**, or **negative**. It improves upon traditional NLP techniques by applying **transformer-based transfer learning** for better accuracy and generalization.

### 📌 Features
* 💬 **Text Preprocessing**
  * Basic cleaning (e.g., lowercasing, URL removal)
  * Hugging Face’s `BertTokenizer` used for subword tokenization and input formatting

* 🔧 **Model Fine-tuning**
  * Used Hugging Face `Trainer` API with `BertForSequenceClassification`
  * Fine-tuned on cleaned tweet dataset with 3 sentiment classes
  * Configured training with:
    * Learning rate: `2e-5`
    * Batch size: `16`
    * Epochs: `3`
    * Weight decay: `0.01`

* 📊 **Training Monitoring**
  * Integrated **Weights & Biases (wandb)** for experiment tracking and hyperparameter tuning

* 📈 **Evaluation**
  * Accuracy: **\~85.96%**
  * Evaluation Loss: **\~0.47**
  * Evaluation Speed: **\~146 samples/sec**

### 🧠 Techniques Used
* **Transfer Learning** with pretrained BERT
* **Tokenization** with `BertTokenizer`
* **Fine-tuning** on a real-world dataset using Hugging Face Transformers
* **Evaluation Metrics**: Accuracy, Cross-Entropy Loss
* **Experiment Tracking** with wandb
* Optimized model for limited hardware (Google Colab + T4 GPU)

### 📁 Dataset
* **Source**: Kaggle - Airline Tweet Sentiment
* **Classes**: positive, neutral, negative
* **Text Format**: Short tweets mentioning airline services

### 🧪 Results
* ✅ Final accuracy: **85.96%**
* ✅ Loss: **0.47**
* ✅ Model saved using `model.save_pretrained()` for future use or deployment

### 📦 Requirements
* `transformers`
* `datasets`
* `pandas`
* `sklearn`
* `wandb`
* `nltk`
* `pytorch`
