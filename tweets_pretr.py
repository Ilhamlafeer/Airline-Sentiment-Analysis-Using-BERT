import pandas as pd
from sklearn.preprocessing import LabelEncoder
from datasets import Dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments

# 1. Load and prepare data
df = pd.read_csv("/content/drive/MyDrive/Colab Notebooks/NLP/Tweets.csv")
df = df[["text", "airline_sentiment"]]

le = LabelEncoder()
df['label'] = le.fit_transform(df['airline_sentiment'])

dataset = Dataset.from_pandas(df[['text','label']])
dataset = dataset.train_test_split(test_size=0.2)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def tokenize_function(tweets):
    return tokenizer(tweets['text'], padding='max_length', truncation=True, max_length=128)

tokenized_dataset = dataset.map(tokenize_function, batched=True)

#pretrained BERT model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased',num_labels=3)
training_args = TrainingArguments(
    output_dir='./results',
    #evaluation_strategy='epoch',
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir='./logs',
)

from sklearn.metrics import accuracy_score
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=1)
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset['train'],
    eval_dataset=tokenized_dataset['test'],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()
trainer.evaluate()
model.save_pretrained('./sentiment_bert')
tokenizer.save_pretrained('./sentiment_bert')

#{'eval_loss': 0.4731791317462921,
#'eval_accuracy': 0.8596311475409836,
# 'eval_runtime': 19.9593,
# 'eval_samples_per_second': 146.699,
# 'eval_steps_per_second': 9.169,
# 'epoch': 3.0}
