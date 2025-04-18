import os
import pandas as pd
import numpy as np
import re
from tqdm import tqdm
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

from datasets import Dataset
from transformers import BertTokenizer, BertForSequenceClassification, TrainingArguments, Trainer, set_seed
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import torch

from transformers import TrainingArguments
print("TrainingArguments source:", TrainingArguments.__module__)

set_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#  1. Datasett TA
file1 = 'data/Data_TA.csv'
file2 = 'data/Data_TA 2.csv'
df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)
df = pd.concat([df1, df2], ignore_index=True)
df = df[df['textDisplay'].notna()].reset_index(drop=True)

# 2. Normalisasi & Stemming 
factory = StemmerFactory()
stemmer = factory.create_stemmer()

def normalize_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return stemmer.stem(text)

tqdm.pandas()
df['normalized'] = df['textDisplay'].progress_apply(normalize_text)

# 3. Dummy Labeling (karena belum ada label)
np.random.seed(42)
df['label'] = np.random.choice([0, 1, 2], size=len(df))

# 4. Train/Test Split 
X_train, X_test, y_train, y_test = train_test_split(
    df['normalized'], df['label'], test_size=0.2, stratify=df['label'], random_state=42
)

train_data = Dataset.from_pandas(pd.DataFrame({'text': X_train, 'label': y_train}))
test_data = Dataset.from_pandas(pd.DataFrame({'text': X_test, 'label': y_test}))

#  5. Tokenizer & Model 
tokenizer = BertTokenizer.from_pretrained('indobenchmark/indobert-base-p1')
model = BertForSequenceClassification.from_pretrained('indobenchmark/indobert-base-p1', num_labels=3)
model.to(device)

# 6. Tokenisasi 
def tokenize_function(example):
    return tokenizer(example['text'], truncation=True, padding='max_length', max_length=128)

train_data = train_data.map(tokenize_function, batched=True)
test_data = test_data.map(tokenize_function, batched=True)

train_data = train_data.remove_columns(["text"])
test_data = test_data.remove_columns(["text"])
train_data.set_format("torch")
test_data.set_format("torch")

training_args = TrainingArguments(
    output_dir="./output",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    logging_dir="./logs",
    logging_steps=10,
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=test_data,
    tokenizer=tokenizer,
)

# 9. Train 
trainer.train()

# 10. Evaluasi 
predictions = trainer.predict(test_data)
y_pred = np.argmax(predictions.predictions, axis=1)
y_true = predictions.label_ids

# Simpan classification report ke file
report = classification_report(y_true, y_pred, digits=4)
print("=== Classification Report ===")
print(report)

# Simpan ke file .txt
with open("output/classification_report.txt", "w", encoding="utf-8") as f:
    f.write("=== Classification Report ===\n")
    f.write(report)

#  11. Prediksi Semua Komentar Dataset
def predict_sentiment(text):
    model.eval()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=128).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        prediction = torch.argmax(logits, dim=-1).item()
    return prediction

df['predicted_label'] = df['normalized'].progress_apply(predict_sentiment)
label_map = {0: 'Negatif', 1: 'Netral', 2: 'Positif'}
df['sentimen'] = df['predicted_label'].map(label_map)

print("\n=== Contoh Hasil Analisis ===")
print(df[['textDisplay', 'sentimen']].head(10))

#output
os.makedirs('output', exist_ok=True)
df.to_csv('output/hasil_analisis_sentimen.csv', index=False)
print("\nHasil disimpan di: output/hasil_analisis_sentimen.csv")