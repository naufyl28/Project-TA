import pandas as pd
import torch
from transformers import Trainer, TrainingArguments
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Fungsi untuk menghitung metrik evaluasi
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = torch.argmax(torch.tensor(logits), axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
    acc = accuracy_score(labels, predictions)
    return {'accuracy': acc, 'precision': precision, 'recall': recall, 'f1': f1}

# Fungsi untuk mendapatkan bobot kelas
def get_class_weights(train_labels, num_classes):
    classes = np.unique(train_labels)
    
    if len(classes) != num_classes:
        print(f"Peringatan: Ditemukan {len(classes)} kelas dalam data, tetapi jumlah kelas yang diharapkan adalah {num_classes}.")
    
    class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=train_labels)
    return torch.tensor(class_weights, dtype=torch.float)

# Fungsi untuk melatih model
def train_model(model_class, tokenizer, train_dataset, eval_dataset, output_dir, train_labels, pretrained_model_name, use_hidden_layer=True, num_classes=3):
    # Hitung bobot kelas
    class_weights = get_class_weights(train_labels, num_classes)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    class_weights = class_weights.to(device)

    # Bangun model dengan class weight
    model = model_class(
        pretrained_model_name=pretrained_model_name,
        use_hidden_layer=use_hidden_layer,
        num_classes=num_classes,
        class_weights=class_weights
    )

    # Pengaturan TrainingArguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,  # Sesuaikan dengan eksperimentasi Anda
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        logging_dir=f"{output_dir}/logs",
        logging_steps=50,  # Log every 50 steps
        evaluation_strategy="epoch",  # Evaluate at the end of each epoch
        save_strategy="epoch",  # Save model at the end of each epoch
        save_total_limit=2,  # Limit the number of saved models
        load_best_model_at_end=True,  # Load the best model based on evaluation
        weight_decay=0.01,  # Regularization to avoid overfitting
        logging_first_step=True,  # Log the first step
    )

    # Membuat objek Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # Melatih model
    trainer.train()

    # Ambil log history
    logs = pd.DataFrame(trainer.state.log_history)

    # Simpan log ke dalam file CSV
    logs.to_csv(f"{output_dir}/training_logs.csv", index=False)

    # Mengembalikan hanya train_loss dan train_accuracy dari log
    # Di sini, kita hanya mengembalikan dua metrik yang dibutuhkan
    last_epoch_log = logs.iloc[-1]  # Ambil log epoch terakhir
    train_loss = last_epoch_log.get('train_loss', None)
    train_accuracy = last_epoch_log.get('eval_accuracy', None)  # Mengambil akurasi dari evaluasi

    return train_loss, train_accuracy
