import os
import json
import random
from collections import Counter
from datasets import load_dataset


def prepare_financial_sentiment_data(
    output_dir: str = "data",
    max_train_samples: int | None = 1500,  # 1500 samples is ideal for fast CPU fine-tuning (~10 mins)
    val_samples: int = 300
):
    """
    Downloads financial sentiment data from HuggingFace,
    cleans it, and prepares data/train.jsonl & data/val.jsonl.
    """
    os.makedirs(output_dir, exist_ok=True)
    train_path = os.path.join(output_dir, "train.jsonl")
    val_path = os.path.join(output_dir, "val.jsonl")

    print("[1/3] Loading financial sentiment dataset from HuggingFace...")
    ds = load_dataset("zeroshot/twitter-financial-news-sentiment")

    # Mapping: 0 = Bearish/Negative, 1 = Bullish/Positive, 2 = Neutral
    label_map = {0: "negative", 1: "positive", 2: "neutral"}

    raw_train = ds["train"]
    raw_val = ds["validation"]

    print(f"      Downloaded {len(raw_train)} raw train rows, {len(raw_val)} raw val rows.")

    train_rows = []
    for item in raw_train:
        train_rows.append({
            "text": item["text"].strip(),
            "label": label_map[int(item["label"])]
        })

    val_rows = []
    for item in raw_val:
        val_rows.append({
            "text": item["text"].strip(),
            "label": label_map[int(item["label"])]
        })

    random.seed(42)
    random.shuffle(train_rows)
    random.shuffle(val_rows)

    if max_train_samples and max_train_samples < len(train_rows):
        print(f"[2/3] Subsetting to {max_train_samples} samples for fast CPU training...")
        train_rows = train_rows[:max_train_samples]

    if val_samples and val_samples < len(val_rows):
        val_rows = val_rows[:val_samples]

    # Write train.jsonl
    with open(train_path, "w", encoding="utf-8") as f:
        for r in train_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Write val.jsonl
    with open(val_path, "w", encoding="utf-8") as f:
        for r in val_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[3/3] Data files created successfully:")
    print(f"      -> {train_path} ({len(train_rows)} samples)")
    print(f"      -> {val_path} ({len(val_rows)} samples)")

    counts = Counter(r["label"] for r in train_rows)
    print("\nTrain Label Distribution:")
    for label, count in sorted(counts.items()):
        print(f"  {label:>10}: {count:>5} ({count/len(train_rows)*100:.1f}%)")


if __name__ == "__main__":
    prepare_financial_sentiment_data()
