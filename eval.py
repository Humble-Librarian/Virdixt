"""
Evaluation script to compute full classification metrics on validation set.
"""

import json
import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_PATH = "./output"
VAL_DATA = "./data/val.jsonl"
LABEL_NAMES = ["negative", "neutral", "positive"]


def evaluate():
    print(f"Loading model from {MODEL_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.eval()

    print(f"Reading validation data from {VAL_DATA}...")
    val_rows = []
    with open(VAL_DATA, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                val_rows.append(json.loads(line))

    y_true = []
    y_pred = []

    print(f"Evaluating {len(val_rows)} validation samples...")
    for item in val_rows:
        text = item["text"]
        label = item["label"]
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        with torch.no_grad():
            logits = model(**inputs).logits
        pred_idx = logits.argmax(dim=-1).item()

        y_true.append(label)
        y_pred.append(LABEL_NAMES[pred_idx])

    print("\n" + "=" * 60)
    print("           CLASSIFICATION REPORT")
    print("=" * 60)
    print(classification_report(y_true, y_pred, target_names=LABEL_NAMES, digits=4))

    print("\n" + "=" * 60)
    print("           CONFUSION MATRIX")
    print("=" * 60)
    cm = confusion_matrix(y_true, y_pred, labels=LABEL_NAMES)
    print(f"{'':>12} Pred Negative  Pred Neutral  Pred Positive")
    for idx, name in enumerate(LABEL_NAMES):
        print(f"Actual {name:<8}: {cm[idx][0]:>12} {cm[idx][1]:>12} {cm[idx][2]:>12}")


if __name__ == "__main__":
    evaluate()
