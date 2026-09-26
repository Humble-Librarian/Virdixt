"""
Aspect-Based Financial Sentiment Analysis (ABSA) Engine
Decomposes complex financial documents into domain-specific operational aspects
and evaluates sentiment polarity, distress probability, and ERP policy flags per aspect.
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any


@dataclass
class AspectResult:
    aspect_name: str
    detected: bool
    sentiment: str
    distress_score: float
    confidence: float
    key_sentences: List[str] = field(default_factory=list)
    risk_level: str = "NEUTRAL"


class ABSAEngine:
    """Decomposes financial text into 5 distinct operational and solvency aspects."""

    ASPECT_PATTERNS = {
        "Top-Line & Growth": [
            r"\b(revenue|sales|bookings|arr|mrr|growth|organic growth|top-line|customer acquisition|volume|shipments)\b"
        ],
        "Cost & Profitability": [
            r"\b(margin|gross margin|operating margin|cogs|cost of goods|operating expenses|opex|sg&a|overhead|ebitda|gross profit)\b"
        ],
        "Liquidity & Cash Flow": [
            r"\b(cash|cash flow|operating cash flow|free cash flow|fcf|working capital|cash burn|burn rate|liquidity buffer|drain)\b"
        ],
        "Debt & Capital Solvency": [
            r"\b(debt|total debt|credit facility|covenant|leverage|interest coverage|insolvency|borrowing|maturities|liabilities)\b"
        ],
        "Audit & Governance Risk": [
            r"\b(audit|auditor|going concern|material weakness|sec|investigation|restatement|qualification|litigation|adverse)\b"
        ],
    }

    def __init__(self, system1_engine: Any):
        self.system1 = system1_engine
        self.compiled_aspects = {
            name: [re.compile(p, re.IGNORECASE) for p in patterns]
            for name, patterns in self.ASPECT_PATTERNS.items()
        }

    def extract_aspect_sentences(self, document: str) -> Dict[str, List[str]]:
        """Extracts relevant sentences associated with each financial aspect."""
        # Split document into clean sentences
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", document) if len(s.strip()) > 10]
        aspect_sentences: Dict[str, List[str]] = {k: [] for k in self.compiled_aspects}

        for s in sentences:
            for aspect_name, patterns in self.compiled_aspects.items():
                if any(p.search(s) for p in patterns):
                    aspect_sentences[aspect_name].append(s)

        return aspect_sentences

    def evaluate(self, document: str) -> List[AspectResult]:
        """Evaluates sentiment and distress across each distinct financial aspect."""
        extracted = self.extract_aspect_sentences(document)
        results = []

        for aspect_name, sents in extracted.items():
            if not sents:
                results.append(
                    AspectResult(
                        aspect_name=aspect_name,
                        detected=False,
                        sentiment="NOT_MENTIONED",
                        distress_score=0.0,
                        confidence=0.0,
                        key_sentences=[],
                        risk_level="UNREPORTED",
                    )
                )
                continue

            joined_text = " ".join(sents[:4])  # Take up to top 4 sentences
            probs = self.system1.get_calibrated_probs(joined_text)
            choice = self.system1.choice(joined_text)
            score = self.system1.score(joined_text)

            if score > 70 or probs["negative"] > 0.60:
                risk = "CRITICAL"
            elif score > 35 or probs["negative"] > 0.35:
                risk = "WARNING"
            elif probs["positive"] > 0.60:
                risk = "SAFE"
            else:
                risk = "NEUTRAL"

            conf = max(probs["negative"], probs["positive"], probs["neutral"]) * 100.0

            results.append(
                AspectResult(
                    aspect_name=aspect_name,
                    detected=True,
                    sentiment=choice,
                    distress_score=score,
                    confidence=conf,
                    key_sentences=sents[:2],
                    risk_level=risk,
                )
            )

        return results
