"""
Fine-tuning FinBERT for Financial Sentiment Analysis on CPU.
Optimized for 8GB RAM and Intel Core i5 CPU.
"""

import os
import json
import math
import time
import torch
import numpy as np
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)

# 1. Optimize PyTorch thread allocation for 4-core / 8-thread CPU
torch.set_num_threads(4)

MODEL_NAME = "ProsusAI/finbert"
OUTPUT_DIR = "./output"
TRAIN_DATA = "./data/train.jsonl"
VAL_DATA = "./data/val.jsonl"

LABEL_NAMES = ["negative", "neutral", "positive"]
LABEL_TO_ID = {name: i for i, name in enumerate(LABEL_NAMES)}
ID_TO_LABEL = {i: name for i, name in enumerate(LABEL_NAMES)}


def load_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    preds = np.argmax(predictions, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )
    acc = accuracy_score(labels, preds)
    return {
        "accuracy": round(acc, 4),
        "f1": round(f1, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
    }


def main():
    print("=" * 70)
    print("  FinBERT Sentiment Analysis — CPU Fine-tuning Pipeline")
    print("=" * 70)

    if not os.path.exists(TRAIN_DATA):
        print(f"Train data not found at {TRAIN_DATA}. Running prepare_data.py first...")
        from prepare_data import prepare_financial_sentiment_data
        prepare_financial_sentiment_data()

    print(f"\n[1/5] Loading datasets from {TRAIN_DATA}...")
    train_raw = load_jsonl(TRAIN_DATA)
    val_raw = load_jsonl(VAL_DATA) if os.path.exists(VAL_DATA) else []

    print(f"      Train rows: {len(train_raw)}, Validation rows: {len(val_raw)}")

    print(f"\n[2/5] Loading tokenizer and model: {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=3,
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
    )

    def tokenize_func(batch):
        enc = tokenizer(
            batch["text"],
            truncation=True,
            max_length=128,
            padding=False,
        )
        enc["labels"] = [LABEL_TO_ID[label] for label in batch["label"]]
        return enc

    print("\n[3/5] Tokenizing dataset...")
    train_dataset = Dataset.from_list(train_raw).map(
        tokenize_func, batched=True, remove_columns=["text", "label"]
    )
    val_dataset = None
    if val_raw:
        val_dataset = Dataset.from_list(val_raw).map(
            tokenize_func, batched=True, remove_columns=["text", "label"]
        )

    # Training arguments tuned for 8GB RAM & CPU / GPU
    batch_size = 4
    gradient_accumulation_steps = 4  # Effective batch size = 16
    epochs = 3  # 3 epochs for full convergence

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        logging_steps=10,
        eval_strategy="epoch" if val_dataset else "no",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True if val_dataset else False,
        metric_for_best_model="f1",
        use_cpu=True,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )

    print("\n[4/5] Starting fine-tuning on CPU (Intel Core i5)...")
    start_time = time.time()
    trainer.train()
    duration = time.time() - start_time

    print(f"\n[5/5] Training finished in {duration/60:.2f} minutes.")
    print(f"      Saving final model to {OUTPUT_DIR}...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    if val_dataset:
        print("\n=== Final Validation Metrics ===")
        metrics = trainer.evaluate()
        for k, v in metrics.items():
            print(f"  {k}: {v}")

    print(f"\n Success! Model is saved to {OUTPUT_DIR}/. You can now run infer.py to test.")


if __name__ == "__main__":
    main()
