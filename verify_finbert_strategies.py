"""
Verification and Comparison Suite for FinBERT Fine-Tuning Strategies:
1. STL + GU         (Slanted Triangular Learning Rate + Gradual Unfreezing)
2. GU + DFT         (Gradual Unfreezing + Discriminative Fine-Tuning / Layer-wise LR Decay)
3. STL + GU + DFT   (Full Triad)
"""

import sys
import os
import torch

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from train_advanced import (
    get_dft_parameter_groups,
    GradualUnfreezingManager,
    SlantedTriangularLRScheduler,
    MODEL_NAME,
    ID_TO_LABEL,
    LABEL_TO_ID,
)
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def verify_strategies():
    print("=" * 80)
    print("  VERIFYING FINBERT ADVANCED STRATEGIES: STL+GU, GU+DFT, STL+GU+DFT")
    print("=" * 80)

    print("\n[1] Initializing FinBERT model structure...")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=3,
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
    )

    # -------------------------------------------------------------
    # Strategy 1: STL + GU
    # -------------------------------------------------------------
    print("\n[2] Testing Strategy 1: STL + GU (Slanted Triangular LR + Gradual Unfreezing)")
    # Epoch 1 GU test
    GradualUnfreezingManager.apply_unfreezing_schedule(model, epoch=0, total_epochs=3)
    trainable_p1 = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert trainable_p1 < 1000000, "Epoch 1 should only train classifier head"
    
    # STL Scheduler test
    optimizer_stl = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-5)
    stl_sched = SlantedTriangularLRScheduler(optimizer_stl, total_steps=100, cut_frac=0.10, ratio=32.0)
    lr_start = stl_sched.get_last_lr()[0]
    for _ in range(10):  # Warmup
        stl_sched.step()
    lr_peak = stl_sched.get_last_lr()[0]
    for _ in range(90):  # Decay
        stl_sched.step()
    lr_end = stl_sched.get_last_lr()[0]
    print(f"    • STL Warmup: Start LR = {lr_start:.2e} -> Peak LR = {lr_peak:.2e} -> End LR = {lr_end:.2e}")
    assert lr_peak > lr_start and lr_peak > lr_end, "STL curve should peak at cut_step and decay"
    print("    ✅ Strategy 1 (STL + GU) verified successfully!")

    # -------------------------------------------------------------
    # Strategy 2: GU + DFT
    # -------------------------------------------------------------
    print("\n[3] Testing Strategy 2: GU + DFT (Gradual Unfreezing + Discriminative Fine-Tuning)")
    # Unfreeze all for DFT parameter grouping test
    GradualUnfreezingManager.apply_unfreezing_schedule(model, epoch=2, total_epochs=3)
    param_groups = get_dft_parameter_groups(model, base_lr=2e-5, decay_factor=0.95)
    print(f"    • Total DFT Parameter Groups Created: {len(param_groups)}")
    
    lrs = [g["lr"] for g in param_groups]
    print(f"    • Top Layer LR (Classifier/Pooler) : {max(lrs):.2e}")
    print(f"    • Bottom Layer LR (Embeddings)     : {min(lrs):.2e}")
    print(f"    • Layer-wise Decay Factor (xi)    : 0.95")
    assert max(lrs) == 2e-5, "Top layer LR should equal base_lr"
    assert min(lrs) < max(lrs), "Embedding LR must be lower than classifier LR"
    print("    ✅ Strategy 2 (GU + DFT) verified successfully!")

    # -------------------------------------------------------------
    # Strategy 3: STL + GU + DFT
    # -------------------------------------------------------------
    print("\n[4] Testing Strategy 3: STL + GU + DFT (Full Triad)")
    optimizer_triad = torch.optim.AdamW(param_groups)
    stl_dft_sched = SlantedTriangularLRScheduler(optimizer_triad, total_steps=200, cut_frac=0.10)
    
    # Step through schedule
    for _ in range(20):
        stl_dft_sched.step()
    
    top_lr_peak = stl_dft_sched.get_last_lr()[0]
    bot_lr_peak = stl_dft_sched.get_last_lr()[-1]
    print(f"    • Peak Top Layer LR    : {top_lr_peak:.2e}")
    print(f"    • Peak Bottom Layer LR : {bot_lr_peak:.2e}")
    assert top_lr_peak > bot_lr_peak, "Discriminative relationship must hold under STL scheduling"
    print("    ✅ Strategy 3 (STL + GU + DFT) verified successfully!")

    print("\n" + "=" * 80)
    print("  ALL 3 ADVANCED FINBERT FINE-TUNING STRATEGIES VERIFIED (100% SUCCESS)!")
    print("=" * 80)


if __name__ == "__main__":
    verify_strategies()
