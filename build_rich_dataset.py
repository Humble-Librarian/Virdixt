"""
Master Dataset Assembler
Merges real financial data + synthetic corporate data + complex contrasting sentences.
Produces an exact 1:1:1 balanced train and validation set.
"""

import os
import json
import random
from collections import Counter
from datasets import load_dataset
from synthetic_builder import generate_synthetic_samples
from complex_sentence_injector import generate_complex_samples


def build_master_dataset(
    output_dir: str = "./data",
    real_target_per_class: int = 1200,
    synthetic_per_class: int = 600,
    complex_per_class: int = 300
):
    os.makedirs(output_dir, exist_ok=True)
    print("=" * 70)
    print("  Building Balanced Financial Sentiment Dataset with Complex Nuance")
    print("=" * 70)

    # 1. Ingest Real Verified Dataset (Twitter Financial News)
    print("\n[1/4] Ingesting real-world financial sentiment dataset...")
    ds = load_dataset("zeroshot/twitter-financial-news-sentiment")
    t_map = {0: "negative", 1: "positive", 2: "neutral"}

    real_buckets = {"negative": [], "neutral": [], "positive": []}
    for item in ds["train"]:
        text = item["text"].strip()
        label = t_map[int(item["label"])]
        real_buckets[label].append(text)

    random.seed(42)
    for k in real_buckets:
        random.shuffle(real_buckets[k])

    # Sample balanced subset from real data
    all_rows = []
    for lbl in ["negative", "neutral", "positive"]:
        sampled = real_buckets[lbl][:real_target_per_class]
        for t in sampled:
            all_rows.append({"text": t, "label": lbl, "type": "real"})
    print(f"      Selected {len(all_rows)} balanced real samples ({real_target_per_class}/class).")

    # 2. Add Domain-Grounded Synthetic Accounting/Corporate Data
    print(f"\n[2/4] Generating {synthetic_per_class*3} synthetic corporate finance sentences...")
    synth_data = generate_synthetic_samples(samples_per_class=synthetic_per_class)
    for r in synth_data:
        r["type"] = "synthetic"
        all_rows.append(r)

    # 3. Add Hard Multi-Clause Complex Sentences
    print(f"\n[3/4] Injecting {complex_per_class*3} multi-clause contrasting complex sentences...")
    complex_data = generate_complex_samples(samples_per_class=complex_per_class)
    for r in complex_data:
        r["type"] = "complex"
        all_rows.append(r)

    # 4. Shuffle, split into train and val, and save
    random.shuffle(all_rows)

    total_per_class = real_target_per_class + synthetic_per_class + complex_per_class
    total_samples = total_per_class * 3

    print(f"\n[4/4] Dataset Total: {len(all_rows)} rows ({total_per_class} per class exact 1:1:1 balance).")

    split_idx = int(len(all_rows) * 0.85)
    train_rows = all_rows[:split_idx]
    val_rows = all_rows[split_idx:]

    train_path = os.path.join(output_dir, "train.jsonl")
    val_path = os.path.join(output_dir, "val.jsonl")

    with open(train_path, "w", encoding="utf-8") as f:
        for r in train_rows:
            f.write(json.dumps({"text": r["text"], "label": r["label"]}, ensure_ascii=False) + "\n")

    with open(val_path, "w", encoding="utf-8") as f:
        for r in val_rows:
            f.write(json.dumps({"text": r["text"], "label": r["label"]}, ensure_ascii=False) + "\n")

    print(f"\n Successfully saved dataset files:")
    print(f"  -> Train: {train_path} ({len(train_rows)} samples)")
    print(f"  -> Val  : {val_path} ({len(val_rows)} samples)")

    train_counts = Counter(r["label"] for r in train_rows)
    print("\nTrain Split Distribution:")
    for lbl, cnt in sorted(train_counts.items()):
        print(f"  {lbl:>10}: {cnt:>5} ({cnt/len(train_rows)*100:.1f}%)")

    type_counts = Counter(r["type"] for r in train_rows)
    print("\nComposition by Source Type:")
    for t, cnt in sorted(type_counts.items()):
        print(f"  {t:>10}: {cnt:>5} ({cnt/len(train_rows)*100:.1f}%)")


if __name__ == "__main__":
    build_master_dataset()
