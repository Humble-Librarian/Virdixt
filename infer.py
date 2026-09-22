"""
Jev-Inspired System-1 Calibrated Decision Engine for FinBERT.
Features:
- Regex/Dictionary Fast Token Pruner (1,000 tokens -> 150 dense signal tokens)
- Temperature Scaling for Calibrated Probabilities
- Asymmetric Financial Risk Gating (Early Risk Trigger)
- Strict Typed Structured Output (Native Type Zero-Cost Contract)
"""

import sys
import re
from typing import TypedDict, Literal
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_PATH = "./output"
LABEL_NAMES = ["negative", "neutral", "positive"]

# Financial anchor pattern for regex pruning
FINANCIAL_ANCHORS = re.compile(
    r'(\$|€|£|%|\b(revenue|profit|loss|losses|ebitda|margin|margins|debt|guidance|'
    r'covenant|covenants|churn|dividend|cash|sales|impairment|bankruptcy|layoff|layoffs|'
    r'cost|costs|default|defaulted|restructuring|restructure|capex|downgrade|downgraded|'
    r'upgraded|quarterly|annual|fiscal|growth|contracted|expanded|guidance|arr|eps)\b)',
    re.IGNORECASE
)


# Jev System-1 Typed Decision Contract
class FinancialDecision(TypedDict):
    raw_text: str
    pruned_text: str
    token_reduction_pct: float
    decision: Literal["NEGATIVE", "NEUTRAL", "POSITIVE"]
    confidence: float
    calibrated_scores: dict[str, float]
    risk_level: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    actionable_signal: bool


def prune_financial_tokens(raw_text: str, max_chars: int = 1000) -> tuple[str, float]:
    """
    Jev System-1 Pre-Processing: Drops conversational fluff and extracts
    high-density financial signal sentences before neural inference.
    """
    sentences = re.split(r'(?<=[.!?])\s+', raw_text.strip())
    relevant = [s.strip() for s in sentences if FINANCIAL_ANCHORS.search(s)]

    # Fallback to original text if no specific anchor matched
    pruned = " ".join(relevant) if relevant else raw_text.strip()
    pruned = pruned[:max_chars]

    raw_len = max(len(raw_text.split()), 1)
    pruned_len = len(pruned.split())
    reduction_pct = round(max(0.0, (1.0 - (pruned_len / raw_len)) * 100), 1)

    return pruned, reduction_pct


def load_jev_engine(model_path: str = MODEL_PATH):
    print(f"Loading FinBERT backbone from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()
    return tokenizer, model


def evaluate_decision(
    raw_text: str,
    tokenizer,
    model,
    apply_pruning: bool = True,
    temperature: float = 1.25,        # Jev Calibration: Softens overconfident logits
    risk_threshold: float = 0.35       # Asymmetric Gating: P(neg) >= 35% triggers risk
) -> FinancialDecision:
    """Executes an end-to-end calibrated System-1 structured financial decision."""
    
    # 1. Jev Layer 1: Token Pruning (1,000 -> 150 tokens)
    if apply_pruning:
        pruned_text, reduction_pct = prune_financial_tokens(raw_text)
    else:
        pruned_text, reduction_pct = raw_text, 0.0

    inputs = tokenizer(pruned_text, return_tensors="pt", truncation=True, max_length=128)
    
    with torch.no_grad():
        logits = model(**inputs).logits

    # 2. Jev Layer 2: Temperature Scaling for Calibrated Probabilities
    calibrated_logits = logits / temperature
    probs = F.softmax(calibrated_logits, dim=-1).squeeze()

    scores = {
        "negative": round(probs[0].item(), 4),
        "neutral": round(probs[1].item(), 4),
        "positive": round(probs[2].item(), 4)
    }

    # 3. Jev Layer 3: Asymmetric Risk Gating
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
        "raw_text": raw_text,
        "pruned_text": pruned_text,
        "token_reduction_pct": reduction_pct,
        "decision": decision,
        "confidence": round(top_prob * 100, 2),
        "calibrated_scores": scores,
        "risk_level": risk_level,
        "actionable_signal": actionable
    }


def demo():
    tokenizer, model = load_jev_engine()

    test_suite = [
        # Example 1: Multi-paragraph earnings transcript with conversational fluff
        (
            "Good morning ladies and gentlemen, welcome to our third quarter conference call. "
            "I'd like to thank everyone for joining us today on this webcast. "
            "Consolidated revenues expanded by 14% YoY, but operating cash flow turned deeply negative due to rising debt costs. "
            "Please note that forward-looking statements are subject to safe harbor provisions. "
            "I will now turn the call over to our investor relations team for closing remarks."
        ),
        # Example 2: Positive earnings beat
        (
            "The corporation reached record quarterly gross margins of 45% and announced an accelerated share buyback. "
            "Our headquarters remain fully staffed and operational."
        ),
        # Example 3: Neutral administrative filing
        (
            "The board of directors convened on Monday to review standard quarterly governance filings. "
            "No executive compensation changes were enacted during the meeting."
        ),
        # Example 4: Critical corporate financial distress
        (
            "Regarding our ongoing operational review, the audit committee concluded its review today. "
            "Supplier default and severe inventory write-downs caused net losses to widen significantly. "
            "Management is actively evaluating all strategic alternatives."
        )
    ]

    print("\n" + "=" * 95)
    print("      JEV-CALIBRATED SYSTEM-1 FINANCIAL DECISION ENGINE (WITH TOKEN PRUNER)")
    print("=" * 95)

    for i, s in enumerate(test_suite, 1):
        res = evaluate_decision(s, tokenizer, model)
        print(f"\n[Case {i}]")
        print(f" Raw Input ({len(s.split())} words) : \"{s[:80]}...\"")
        print(f" Pruned    ({len(res['pruned_text'].split())} words) : \"{res['pruned_text']}\"")
        print(f" -> Token Compression  : {res['token_reduction_pct']}% pruned")
        print(f" -> Decision           : {res['decision']:<8} (Calibrated Confidence: {res['confidence']}%)")
        print(f" -> Risk Level         : {res['risk_level']:<8} | Actionable Signal: {res['actionable_signal']}")
        print(f" -> Score Breakdown    : {res['calibrated_scores']}")
    print("=" * 95)


if __name__ == "__main__":
    demo()
