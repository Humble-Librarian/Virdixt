"""
Jev-Inspired System-1 Calibrated Decision Engine for FinBERT.
Features:
- Temperature Scaling for Calibrated Probabilities
- Asymmetric Financial Risk Gating (Early Risk Trigger)
- Strict Typed Structured Output
"""

import sys
from typing import TypedDict, Literal
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_PATH = "./output"
LABEL_NAMES = ["negative", "neutral", "positive"]

# Jev System-1 Typed Decision Contract
class FinancialDecision(TypedDict):
    text: str
    decision: Literal["NEGATIVE", "NEUTRAL", "POSITIVE"]
    confidence: float
    calibrated_scores: dict[str, float]
    risk_level: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    actionable_signal: bool


def load_jev_engine(model_path: str = MODEL_PATH):
    print(f"Loading FinBERT backbone from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()
    return tokenizer, model


def evaluate_decision(
    text: str,
    tokenizer,
    model,
    temperature: float = 1.25,        # Jev Calibration: Temperature scaling softens overconfident logits
    risk_threshold: float = 0.35       # Asymmetric Gating: P(neg) >= 35% triggers risk
) -> FinancialDecision:
    """Executes a calibrated System-1 structured financial decision."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    
    with torch.no_grad():
        logits = model(**inputs).logits

    # 1. Jev Calibration: Temperature Scaling
    calibrated_logits = logits / temperature
    probs = F.softmax(calibrated_logits, dim=-1).squeeze()

    scores = {
        "negative": round(probs[0].item(), 4),
        "neutral": round(probs[1].item(), 4),
        "positive": round(probs[2].item(), 4)
    }

    # 2. Jev Asymmetric Risk Gating
    neg_prob = scores["negative"]
    pos_prob = scores["positive"]

    if neg_prob >= 0.60:
        decision = "NEGATIVE"
        risk_level = "CRITICAL"
        actionable = True
    elif neg_prob >= risk_threshold:
        decision = "NEGATIVE"
        risk_level = "HIGH"
        actionable = True
    elif pos_prob >= 0.55:
        decision = "POSITIVE"
        risk_level = "LOW"
        actionable = True
    else:
        decision = "NEUTRAL"
        risk_level = "LOW"
        actionable = False

    top_prob = max(scores.values())

    return {
        "text": text,
        "decision": decision,
        "confidence": round(top_prob * 100, 2),
        "calibrated_scores": scores,
        "risk_level": risk_level,
        "actionable_signal": actionable
    }


def demo():
    tokenizer, model = load_jev_engine()

    test_suite = [
        "Consolidated revenues expanded by 14% YoY, but operating cash flow turned deeply negative due to rising debt costs.",
        "The corporation reached record quarterly gross margins of 45% and announced an accelerated share buyback.",
        "The board of directors convened on Monday to review standard quarterly governance filings.",
        "Supplier default and severe inventory write-downs caused net losses to widen significantly.",
        "Despite foreign exchange headwinds, organic ARR grew 18% beating all street expectations."
    ]

    print("\n" + "=" * 85)
    print("      JEV-CALIBRATED SYSTEM-1 FINANCIAL DECISION ENGINE")
    print("=" * 85)

    for s in test_suite:
        res = evaluate_decision(s, tokenizer, model)
        print(f"\n Text     : {res['text']}")
        print(f" -> Decision   : {res['decision']:<8} (Confidence: {res['confidence']}%)")
        print(f" -> Risk Level : {res['risk_level']:<8} | Actionable: {res['actionable_signal']}")
        print(f" -> Breakdown  : {res['calibrated_scores']}")
    print("=" * 85)


if __name__ == "__main__":
    demo()
