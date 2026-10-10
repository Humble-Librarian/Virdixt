"""
Advanced FinBERT Fine-Tuning with ULMFiT & FinBERT Strategies:
1. STL + GU         : Slanted Triangular Learning Rate + Gradual Unfreezing
2. GU + DFT         : Gradual Unfreezing + Discriminative Fine-Tuning (Layer-wise LR Decay)
3. STL + GU + DFT   : Full Trifecta (Slanted Triangular LR + Gradual Unfreezing + Discriminative Fine-Tuning)

Based on Dogu Araci (2019) 'FinBERT: Financial Sentiment Analysis with Pre-trained Language Models'
and Howard & Ruder (2018) 'Universal Language Model Fine-tuning for Text Classification'.
"""

import os
import sys
import json
import math
import time
import argparse
import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
    DataCollatorWithPadding,
    TrainerCallback,
)

# Optimize thread allocation for CPU execution
torch.set_num_threads(4)

MODEL_NAME = "ProsusAI/finbert"
TRAIN_DATA = "./data/train.jsonl"
VAL_DATA = "./data/val.jsonl"

LABEL_NAMES = ["negative", "neutral", "positive"]
LABEL_TO_ID = {name: i for i, name in enumerate(LABEL_NAMES)}
ID_TO_LABEL = {i: name for i, name in enumerate(LABEL_NAMES)}


# =========================================================================
# 1. DISCRIMINATIVE FINE-TUNING (DFT) / LAYER-WISE LEARNING RATE DECAY
# =========================================================================
def get_dft_parameter_groups(
    model: torch.nn.Module,
    base_lr: float = 2e-5,
    decay_factor: float = 0.95,
    weight_decay: float = 0.01,
) -> list[dict]:
    """Assigns layer-wise discriminative learning rates:
    
    Classifier & Pooler -> base_lr
    Transformer Layer 11 -> base_lr * (decay^1)
    Transformer Layer 10 -> base_lr * (decay^2)
    ...
    Transformer Layer 0  -> base_lr * (decay^12)
    Embeddings           -> base_lr * (decay^13)
    """
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
    param_groups = []

    # 1. Classifier & Pooler parameters (Top Level)
    top_decay_params = []
    top_no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "bert" not in name or "pooler" in name or "classifier" in name or "score" in name:
            if any(nd in name for nd in no_decay):
                top_no_decay_params.append(param)
            else:
                top_decay_params.append(param)

    if top_decay_params:
        param_groups.append({"params": top_decay_params, "lr": base_lr, "weight_decay": weight_decay})
    if top_no_decay_params:
        param_groups.append({"params": top_no_decay_params, "lr": base_lr, "weight_decay": 0.0})

    # 2. 12 Transformer Encoder Layers (Descending LR)
    for layer_idx in range(11, -1, -1):
        layer_lr = base_lr * (decay_factor ** (12 - layer_idx))
        layer_decay = []
        layer_no_decay = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if f"encoder.layer.{layer_idx}." in name:
                if any(nd in name for nd in no_decay):
                    layer_no_decay.append(param)
                else:
                    layer_decay.append(param)

        if layer_decay:
            param_groups.append({"params": layer_decay, "lr": layer_lr, "weight_decay": weight_decay})
        if layer_no_decay:
            param_groups.append({"params": layer_no_decay, "lr": layer_lr, "weight_decay": 0.0})

    # 3. Embeddings Layer (Lowest LR)
    embed_lr = base_lr * (decay_factor ** 13)
    embed_decay = []
    embed_no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "embeddings." in name:
            if any(nd in name for nd in no_decay):
                embed_no_decay.append(param)
            else:
                embed_decay.append(param)

    if embed_decay:
        param_groups.append({"params": embed_decay, "lr": embed_lr, "weight_decay": weight_decay})
    if embed_no_decay:
        param_groups.append({"params": embed_no_decay, "lr": embed_lr, "weight_decay": 0.0})

    return param_groups


# =========================================================================
# 2. GRADUAL UNFREEZING (GU) SCHEDULER
# =========================================================================
class GradualUnfreezingManager:
    """Controls layer-by-layer progressive unfreezing across training epochs.
    
    Epoch 1: Only Classifier & Pooler unfrozen.
    Epoch 2: Top Transformer Encoder Layers (8-11) unfrozen.
    Epoch 3: All Encoder Layers (0-7) & Embeddings unfrozen.
    """

    @staticmethod
    def apply_unfreezing_schedule(model: torch.nn.Module, epoch: int, total_epochs: int = 3):
        if epoch == 0:
            # Epoch 1: Freeze everything except classifier and pooler
            for name, param in model.named_parameters():
                if "classifier" in name or "pooler" in name or "score" in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            unfrozen_str = "Classifier Head & Pooler"
        elif epoch == 1:
            # Epoch 2: Unfreeze top 4 encoder layers (8 to 11)
            for name, param in model.named_parameters():
                if any(f"encoder.layer.{i}." in name for i in range(8, 12)) or "classifier" in name or "pooler" in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            unfrozen_str = "Top Layers (8-11) + Classifier"
        else:
            # Epoch 3+: Unfreeze all parameters
            for name, param in model.named_parameters():
                param.requires_grad = True
            unfrozen_str = "ALL Layers (Embeddings, Layers 0-11, Classifier)"

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"    [GU Stage] Epoch {epoch + 1}/{total_epochs} -> Unfrozen: {unfrozen_str} ({trainable:,} / {total:,} trainable params)")


# =========================================================================
# 3. SLANTED TRIANGULAR LEARNING RATE (STL) SCHEDULER
# =========================================================================
class SlantedTriangularLRScheduler:
    """Implements Slanted Triangular Learning Rate (Howard & Ruder, 2018).
    
    Linearly increases learning rate from base_lr/ratio up to base_lr for cut_frac of steps,
    then linearly decays back down to base_lr/ratio.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        total_steps: int,
        cut_frac: float = 0.10,
        ratio: float = 32.0,
    ):
        self.optimizer = optimizer
        self.total_steps = max(1, total_steps)
        self.cut_frac = cut_frac
        self.cut_step = int(self.total_steps * cut_frac)
        self.ratio = ratio
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        self.step_num = 0
        # Set initial learning rates at start
        self._update_lr(0)

    def _update_lr(self, step_idx: int):
        t = step_idx
        T = self.total_steps
        cut = self.cut_step
        ratio = self.ratio

        if t < cut:
            p = t / cut if cut > 0 else 1.0
        else:
            p = 1.0 - (t - cut) / (T - cut) if (T - cut) > 0 else 0.0

        scale = (1.0 + p * (ratio - 1.0)) / ratio

        for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            group["lr"] = base_lr * scale

    def step(self):
        self.step_num += 1
        self._update_lr(self.step_num)

    def get_last_lr(self) -> list[float]:
        return [group["lr"] for group in self.optimizer.param_groups]


# =========================================================================
# 4. DATA LOADING & EVALUATION
# =========================================================================
def load_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def evaluate_model(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="macro", zero_division=0
    )
    # Per-class recall to monitor negative recall
    _, class_recalls, _, _ = precision_recall_fscore_support(
        all_labels, all_preds, average=None, labels=[0, 1, 2], zero_division=0
    )
    neg_recall = class_recalls[0] if len(class_recalls) > 0 else 0.0

    return {
        "accuracy": acc,
        "macro_f1": f1,
        "macro_precision": precision,
        "macro_recall": recall,
        "negative_recall": neg_recall,
    }


# =========================================================================
# 5. MASTER ADVANCED TRAINING RUNNER
# =========================================================================
def train_finbert(
    strategy: str = "stl_gu_dft",
    epochs: int = 3,
    batch_size: int = 4,
    grad_accum_steps: int = 4,
    base_lr: float = 2e-5,
    decay_factor: float = 0.95,
    output_dir: str = "./output_advanced",
):
    print("=" * 80)
    print(f"  FINBERT ADVANCED TRAINING: STRATEGY = [{strategy.upper()}]")
    print(f"  Components Enabled:")
    print(f"    • STL (Slanted Triangular LR) : {'YES' if 'stl' in strategy else 'NO (Constant/Cosine)'}")
    print(f"    • GU  (Gradual Unfreezing)   : {'YES' if 'gu' in strategy else 'NO (All Unfrozen)'}")
    print(f"    • DFT (Discriminative LR)    : {'YES' if 'dft' in strategy else 'NO (Uniform LR)'}")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[1/5] Compute device: {device}")

    # Load datasets
    train_raw = load_jsonl(TRAIN_DATA)
    val_raw = load_jsonl(VAL_DATA) if os.path.exists(VAL_DATA) else []
    print(f"      Train samples: {len(train_raw)}, Validation samples: {len(val_raw)}")

    # Tokenizer & Model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=3,
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
    ).to(device)

    def tokenize_fn(items):
        enc = tokenizer([x["text"] for x in items], truncation=True, max_length=128, padding=True, return_tensors="pt")
        enc["labels"] = torch.tensor([LABEL_TO_ID[x["label"]] for x in items], dtype=torch.long)
        return enc

    # DataLoader
    train_loader = DataLoader(
        train_raw,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=tokenize_fn,
    )
    val_loader = DataLoader(
        val_raw,
        batch_size=8,
        shuffle=False,
        collate_fn=tokenize_fn,
    ) if val_raw else None

    total_update_steps = (len(train_loader) // grad_accum_steps) * epochs

    best_f1 = 0.0
    start_time = time.time()

    for epoch in range(epochs):
        print(f"\n--- Epoch {epoch + 1}/{epochs} ---")
        
        # 1. Apply GU (Gradual Unfreezing) if enabled
        if "gu" in strategy:
            GradualUnfreezingManager.apply_unfreezing_schedule(model, epoch, total_epochs=epochs)
        else:
            for p in model.parameters():
                p.requires_grad = True

        # 2. Build Optimizer (DFT vs Uniform)
        if "dft" in strategy:
            param_groups = get_dft_parameter_groups(model, base_lr=base_lr, decay_factor=decay_factor)
        else:
            param_groups = [{"params": [p for p in model.parameters() if p.requires_grad], "lr": base_lr, "weight_decay": 0.01}]

        optimizer = AdamW(param_groups)

        # 3. Build Scheduler (STL vs Linear Warmup)
        remaining_steps = (len(train_loader) // grad_accum_steps) * (epochs - epoch)
        if "stl" in strategy:
            scheduler = SlantedTriangularLRScheduler(optimizer, total_steps=total_update_steps, cut_frac=0.10)
            # Fast-forward scheduler to current epoch position if re-instantiated
            scheduler.step_num = (len(train_loader) // grad_accum_steps) * epoch
        else:
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_update_steps * 0.1), num_training_steps=total_update_steps)

        # 4. Training Loop
        model.train()
        running_loss = 0.0
        step_count = 0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss / grad_accum_steps
            loss.backward()

            running_loss += loss.item() * grad_accum_steps

            if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                if "stl" in strategy:
                    scheduler.step()
                else:
                    scheduler.step()
                optimizer.zero_grad()
                step_count += 1

                if step_count % 30 == 0:
                    lr_sample = optimizer.param_groups[0]["lr"]
                    print(f"    [Step {step_count:>3d}] Batch Loss: {loss.item() * grad_accum_steps:.4f} | LR: {lr_sample:.2e}")

        epoch_loss = running_loss / len(train_loader)
        print(f"  -> Epoch {epoch + 1} Average Loss: {epoch_loss:.4f}")

        # 5. Validation Evaluation
        if val_loader:
            metrics = evaluate_model(model, val_loader, device)
            print(f"  -> Validation Results: Acc={metrics['accuracy']*100:.2f}% | F1={metrics['macro_f1']:.4f} | Neg-Recall={metrics['negative_recall']*100:.2f}%")
            
            if metrics["macro_f1"] > best_f1:
                best_f1 = metrics["macro_f1"]
                os.makedirs(output_dir, exist_ok=True)
                model.save_pretrained(output_dir)
                tokenizer.save_pretrained(output_dir)
                print(f"  ⭐ New Best Model Saved (Macro-F1: {best_f1:.4f}) to {output_dir}")

    total_time = time.time() - start_time
    print(f"\n[DONE] Training complete in {total_time / 60:.2f} minutes. Best Macro-F1: {best_f1:.4f}")
    return best_f1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FinBERT Advanced Fine-Tuning (STL, GU, DFT)")
    parser.add_argument(
        "--strategy",
        type=str,
        default="stl_gu_dft",
        choices=["stl_gu", "gu_dft", "stl_gu_dft", "baseline"],
        help="Fine-tuning strategy: stl_gu, gu_dft, stl_gu_dft, or baseline",
    )
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size per step")
    parser.add_argument("--base_lr", type=float, default=2e-5, help="Base learning rate for classifier")
    parser.add_argument("--decay_factor", type=float, default=0.95, help="Layer-wise LR decay factor for DFT")
    parser.add_argument("--output_dir", type=str, default="./output", help="Directory to save best checkpoint")

    args = parser.parse_args()
    train_finbert(
        strategy=args.strategy,
        epochs=args.epochs,
        batch_size=args.batch_size,
        base_lr=args.base_lr,
        decay_factor=args.decay_factor,
        output_dir=args.output_dir,
    )
