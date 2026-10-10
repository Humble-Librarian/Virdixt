"""
Evaluation script to compute full classification metrics on validation set.
"""

import sys
import json
import argparse
import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from transformers import AutoTokenizer, AutoModelForSequenceClassification

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

VAL_DATA = "./data/val.jsonl"
LABEL_NAMES = ["negative", "neutral", "positive"]


def evaluate(model_path: str = "./output_candidate"):
    print(f"Loading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
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
    parser = argparse.ArgumentParser(description="Evaluate FinBERT model on validation set")
    parser.add_argument("--model", type=str, default="./output_candidate", help="Path to model directory")
    args = parser.parse_args()
    evaluate(model_path=args.model)
