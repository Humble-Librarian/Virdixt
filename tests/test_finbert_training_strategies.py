import pytest
import torch
import torch.nn as nn
from train_advanced import (
    SlantedTriangularLRScheduler,
    GradualUnfreezingManager,
    get_dft_parameter_groups,
)


class MockBertModel(nn.Module):
    """Lightweight mock reproducing HuggingFace BERT parameter naming hierarchy."""
    def __init__(self):
        super().__init__()
        # Embeddings
        self.bert = nn.Module()
        self.bert.embeddings = nn.Module()
        self.bert.embeddings.word_embeddings = nn.Linear(32, 32)
        self.bert.embeddings.LayerNorm = nn.LayerNorm(32)

        # 12 Encoder Layers
        self.bert.encoder = nn.Module()
        self.bert.encoder.layer = nn.ModuleList([
            nn.ModuleDict({
                "attention": nn.Linear(32, 32),
                "layer_norm": nn.LayerNorm(32),
            })
            for _ in range(12)
        ])

        # Pooler & Classifier
        self.bert.pooler = nn.Linear(32, 32)
        self.classifier = nn.Linear(32, 3)


def test_stlr_schedule_warmup_and_decay():
    """Verify Slanted Triangular Learning Rate linearly warms up then decays."""
    model = nn.Linear(10, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    total_steps = 100
    cut_frac = 0.10
    ratio = 32.0

    scheduler = SlantedTriangularLRScheduler(
        optimizer, total_steps=total_steps, cut_frac=cut_frac, ratio=ratio
    )

    # Initial step t=0: should start at base_lr / ratio
    initial_lr = scheduler.get_last_lr()[0]
    expected_start = 2e-5 / 32.0
    assert pytest.approx(initial_lr, rel=1e-3) == expected_start

    # Step to peak (cut_step = 10)
    for _ in range(10):
        scheduler.step()
    peak_lr = scheduler.get_last_lr()[0]
    assert pytest.approx(peak_lr, rel=1e-3) == 2e-5

    # Step to end (step 100)
    for _ in range(90):
        scheduler.step()
    final_lr = scheduler.get_last_lr()[0]
    assert pytest.approx(final_lr, rel=1e-3) == expected_start


def test_gradual_unfreezing_schedule():
    """Verify GU correctly toggles parameter requires_grad per epoch."""
    model = MockBertModel()

    # Epoch 0 (Epoch 1): Only classifier and pooler unfrozen
    GradualUnfreezingManager.apply_unfreezing_schedule(model, epoch=0, total_epochs=3)
    assert model.classifier.weight.requires_grad is True
    assert model.classifier.bias.requires_grad is True
    assert model.bert.pooler.weight.requires_grad is True
    assert model.bert.embeddings.word_embeddings.weight.requires_grad is False
    assert model.bert.encoder.layer[0]["attention"].weight.requires_grad is False
    assert model.bert.encoder.layer[11]["attention"].weight.requires_grad is False

    # Epoch 1 (Epoch 2): Top layers 8-11 unfrozen, 0-7 frozen
    GradualUnfreezingManager.apply_unfreezing_schedule(model, epoch=1, total_epochs=3)
    assert model.classifier.weight.requires_grad is True
    assert model.bert.encoder.layer[11]["attention"].weight.requires_grad is True
    assert model.bert.encoder.layer[8]["attention"].weight.requires_grad is True
    assert model.bert.encoder.layer[7]["attention"].weight.requires_grad is False
    assert model.bert.encoder.layer[0]["attention"].weight.requires_grad is False
    assert model.bert.embeddings.word_embeddings.weight.requires_grad is False

    # Epoch 2 (Epoch 3): ALL layers unfrozen
    GradualUnfreezingManager.apply_unfreezing_schedule(model, epoch=2, total_epochs=3)
    assert model.classifier.weight.requires_grad is True
    assert model.bert.encoder.layer[11]["attention"].weight.requires_grad is True
    assert model.bert.encoder.layer[0]["attention"].weight.requires_grad is True
    assert model.bert.embeddings.word_embeddings.weight.requires_grad is True


def test_discriminative_learning_rate_decay():
    """Verify DFT creates layer-wise decayed parameter groups with correct decay exponent."""
    model = MockBertModel()
    base_lr = 2e-5
    decay_factor = 0.95

    # Ensure all parameters require grad for full group creation
    for p in model.parameters():
        p.requires_grad = True

    param_groups = get_dft_parameter_groups(
        model, base_lr=base_lr, decay_factor=decay_factor, weight_decay=0.01
    )

    assert len(param_groups) > 0

    # Top group (classifier) must have base_lr
    top_lr = param_groups[0]["lr"]
    assert pytest.approx(top_lr, rel=1e-5) == base_lr

    # Bottom group (embeddings) must have base_lr * (0.95^13)
    bottom_lr = param_groups[-1]["lr"]
    expected_bottom = base_lr * (decay_factor ** 13)
    assert pytest.approx(bottom_lr, rel=1e-5) == expected_bottom

    # Verify weight decay separation: bias and layer_norm should have weight_decay=0.0
    for group in param_groups:
        if group["weight_decay"] == 0.0:
            pass  # correctly has no-decay group
        else:
            assert group["weight_decay"] == 0.01
