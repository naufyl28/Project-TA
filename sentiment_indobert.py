import os
import pandas as pd
import torch
import time
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, Trainer, TrainingArguments
from src.model_builder import IndoBERTClassifier
from src.train_classifier import train_model
from src.evaluate_model import evaluate_prediction_latency
from src.data_preprocessing import load_and_preprocess_data

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f"Using device: {device}")

# 2. Load 
df = load_and_preprocess_data('data/Data_TA_LABEL_cleaned.csv')
print(f"Jumlah data setelah filter komentar tidak bermakna: {len(df)}")


train_texts, val_texts, train_labels, val_labels = train_test_split(
    df['cleaned_comment'].tolist(),
    df['label'].tolist(),
    test_size=0.2,
    random_state=42,
    stratify=df['label']
)

# 4. Tokenisasi
pretrained_model_name = "indobenchmark/indobert-base-p1"
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)

def tokenize_function(texts):
    return tokenizer(texts, padding="max_length", truncation=True, max_length=128)

train_encodings = tokenize_function(train_texts)
val_encodings = tokenize_function(val_texts)

# 5. Konversi ke Dataset PyTorch
class SentimentDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = SentimentDataset(train_encodings, train_labels)
val_dataset = SentimentDataset(val_encodings, val_labels)

# Tanpa & Dengan Hidden Layer
for use_hidden in [True, False]:
    label_config = "with_hidden" if use_hidden else "no_hidden"
    print("=" * 60)
    print(f"Training {label_config.upper()}")
    print("=" * 60)

   #output
    output_dir = f"output/indobert_sentiment_{label_config}"
    os.makedirs(output_dir, exist_ok=True)

    model = IndoBERTClassifier(pretrained_model_name, use_hidden_layer=use_hidden).to(device)
    epoch_log = []

    #  Train Loop 
    for epoch in range(1):
        start_time = time.time()
        
        train_loss, train_accuracy = train_model(
            model_class=IndoBERTClassifier,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            output_dir=output_dir,
            train_labels=train_labels,
            pretrained_model_name=pretrained_model_name,
            use_hidden_layer=use_hidden,
            num_classes=3
        )

        duration = time.time() - start_time

        # Simpan hasil log per epoch
        epoch_log.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "epoch_duration_sec": duration
        })

        # Simpan CSV log setelah setiap epoch
        log_path = f"{output_dir}/epoch_log_{label_config}.csv"
        pd.DataFrame(epoch_log).to_csv(log_path, index=False)

    # Load model dan evaluasi latency
    model = IndoBERTClassifier(pretrained_model_name, use_hidden_layer=use_hidden)
    model_path = os.path.join(output_dir, "pytorch_model.bin")
    model.load_state_dict(torch.load(model_path))
    model.to(device)

    training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_eval_batch_size=1,
    evaluation_strategy="epoch",      
    save_strategy="epoch",            
    load_best_model_at_end=True,
    logging_steps=1,
    save_steps=500,
    eval_steps=100,
)
    trainer = Trainer(model=model, args=training_args)
    avg_latency = evaluate_prediction_latency(trainer, val_dataset)
    print(f"Latency {label_config.upper()}: {avg_latency:.6f} detik\n")

# 7. Cek NaN
print("Jumlah NaN label_manual:", df['label_manual'].isna().sum())
print("Label unik:", df['label_manual'].unique())
print("Training selesai.")