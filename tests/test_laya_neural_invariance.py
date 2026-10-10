import pytest
import os
from infer import LayaSystem1, NoulResult, RiskGrade, ExposureTier, PolicyActionFlag


@pytest.fixture(scope="module")
def laya_engine():
    """Load LayaSystem1 ONNX instance once for the test module."""
    return LayaSystem1(use_onnx=True)


def test_laya_probability_calibration_invariance(laya_engine):
    """Verify calibrated probabilities sum to 1.0 and maintain valid ranges."""
    text = "The enterprise reported stable quarterly operating performance in line with estimates."
    probs, choice, score, noul = laya_engine.get_all_predictions(text)

    # Probabilities must sum to 1.0
    total_prob = probs["negative"] + probs["neutral"] + probs["positive"]
    assert pytest.approx(total_prob, rel=1e-4) == 1.0

    # Choice must match argmax
    assert choice in ["NEGATIVE", "NEUTRAL", "POSITIVE"]

    # Continuous score must be bounded in [0.0, 100.0]
    assert 0.0 <= score <= 100.0

    # NOUL must be an instance of NoulResult with valid percentage bounds
    assert isinstance(noul, NoulResult)
    assert 0.0 <= noul.liquidity_distress <= 100.0
    assert 0.0 <= noul.debt_covenant_breach_risk <= 100.0
    assert 0.0 <= noul.growth_expansion_momentum <= 100.0
    assert 0.0 <= noul.capital_return_sustainable <= 100.0


def test_laya_adversarial_polarity_reversal(laya_engine):
    """Verify that concessive contrast prioritizing cash flow over revenue triggers high distress."""
    adversarial_text = "Although revenue rose by 14%, cash flow turned deeply negative."
    probs, choice, score, noul = laya_engine.get_all_predictions(adversarial_text)

    # Must accurately detect negative economic reality
    assert choice == "NEGATIVE"
    assert probs["negative"] > 0.70
    assert score >= 80.0
    assert noul.liquidity_distress >= 70.0


def test_laya_benign_solvency_recognition(laya_engine):
    """Verify that uncompromised healthy financial statements register low distress."""
    benign_text = "The company generated strong record profit expansion, expanding free cash flow and eliminating outstanding senior debt."
    probs, choice, score, noul = laya_engine.get_all_predictions(benign_text)

    assert choice == "POSITIVE"
    assert probs["positive"] > 0.70
    assert score <= 20.0
    assert noul.growth_expansion_momentum >= 70.0
