"""
Linguistic Deception, Corporate Hedging & Syntactic Obfuscation Detector
Computes Epistemic Uncertainty, Passive Voice Evasion, Gunning-Fog Obfuscation,
and Corporate Double-Speak indices on corporate earnings filings and disclosures.
"""

import re
import math
from dataclasses import dataclass, field
from typing import List, Dict, Any


@dataclass
class HedgingAnalysisResult:
    hedging_score: float  # 0.0 to 100.0 (0=Direct, 100=Extreme Hedging)
    hedging_level: str  # DIRECT, MODERATE, HIGH_UNCERTAINTY, EXTREME_EVASION
    passive_evasion_score: float  # 0.0 to 100.0
    gunning_fog_index: float  # Grade reading level (>18 = Severe Corporate Obfuscation)
    obfuscation_risk: str  # CLEAR, COMPLEX, OBFUSCATED
    detected_hedges: List[str] = field(default_factory=list)
    detected_euphemisms: List[str] = field(default_factory=list)
    passive_phrases: List[str] = field(default_factory=list)


class LinguisticHedgingDetector:
    """Computational linguistics module for detecting executive evasion and obfuscation."""

    # Epistemic modals and uncertainty markers
    HEDGE_TERMS = [
        r"\b(might|could|possibly|potentially|seems to|appears to|is expected to|management believes|believed to be)\b",
        r"\b(preliminarily|subject to verification|under assessment|indicative only|tentatively|speculatively)\b",
        r"\b(approximate|roughly|in the vicinity of|not unreasonable to assume|cannot be ruled out)\b",
    ]

    # Passive voice evasion constructions (deflecting agency/responsibility)
    PASSIVE_PATTERNS = [
        r"\b(was|were|has been|have been|is being|are being)\s+([a-z]+ed|[a-z]+en)\b",
        r"\b(adjustments were (?:made|recognized|deemed necessary)|errors were (?:discovered|identified))\b",
        r"\b(covenants were (?:breached|impacted|waived)|losses were (?:incurred|recorded))\b",
    ]

    # Corporate euphemisms and smoke-screens
    EUPHEMISMS = [
        r"\b(headwinds|macro headwinds|challenging (?:environment|backdrop)|transitory pressure)\b",
        r"\b(strategic realignment|right-sizing|streamlining operations|operational optimization)\b",
        r"\b(one-off adjustment|non-recurring charge|normalization of demand|tough comparisons)\b",
        r"\b(sub-optimal performance|transitional quarter|market recalibration)\b",
    ]

    def __init__(self):
        self.hedge_regexes = [re.compile(p, re.IGNORECASE) for p in self.HEDGE_TERMS]
        self.passive_regexes = [re.compile(p, re.IGNORECASE) for p in self.PASSIVE_PATTERNS]
        self.euphemism_regexes = [re.compile(p, re.IGNORECASE) for p in self.EUPHEMISMS]

    def _count_syllables(self, word: str) -> int:
        """Heuristic syllable counter for Gunning-Fog readability computation."""
        word = word.lower().strip()
        if len(word) <= 3:
            return 1
        word = re.sub(r'(?:[^laeiouy]|ed|es|e)$', '', word)
        word = re.sub(r'^y', '', word)
        syllables = len(re.findall(r'[aeiouy]{1,2}', word))
        return max(1, syllables)

    def analyze(self, document: str) -> HedgingAnalysisResult:
        """Performs full linguistic audit on document text."""
        if not document or not document.strip():
            return HedgingAnalysisResult(
                hedging_score=0.0,
                hedging_level="DIRECT",
                passive_evasion_score=0.0,
                gunning_fog_index=0.0,
                obfuscation_risk="CLEAR",
            )

        words = re.findall(r"\b[a-zA-Z]+\b", document)
        total_words = max(1, len(words))
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", document) if len(s.strip()) > 5]
        total_sentences = max(1, len(sentences))

        # 1. Detect Hedges
        detected_hedges = []
        hedge_count = 0
        for reg in self.hedge_regexes:
            matches = reg.findall(document)
            for m in matches:
                item = m if isinstance(m, str) else m[0]
                detected_hedges.append(item.lower())
                hedge_count += 1

        # 2. Detect Euphemisms
        detected_euphemisms = []
        for reg in self.euphemism_regexes:
            matches = reg.findall(document)
            for m in matches:
                item = m if isinstance(m, str) else m[0]
                detected_euphemisms.append(item.lower())

        # 3. Detect Passive Evasion
        detected_passives = []
        passive_count = 0
        for reg in self.passive_regexes:
            matches = reg.findall(document)
            for m in matches:
                item = " ".join(m) if isinstance(m, tuple) else m
                detected_passives.append(item.lower())
                passive_count += 1

        # 4. Gunning-Fog Index Calculation: 0.4 * ((words / sentences) + 100 * (complex_words / words))
        complex_words = sum(1 for w in words if self._count_syllables(w) >= 3)
        avg_sentence_len = total_words / total_sentences
        pct_complex = (complex_words / total_words) * 100.0
        gunning_fog = 0.4 * (avg_sentence_len + pct_complex)

        # 5. Calculate Normalized Scores
        hedge_density = (hedge_count / total_words) * 100.0
        hedging_score = min(100.0, hedge_density * 25.0 + len(detected_euphemisms) * 10.0)

        passive_density = (passive_count / total_sentences) * 100.0
        passive_score = min(100.0, passive_density * 15.0)

        if hedging_score > 60.0:
            hedge_level = "EXTREME_EVASION"
        elif hedging_score > 35.0:
            hedge_level = "HIGH_UNCERTAINTY"
        elif hedging_score > 15.0:
            hedge_level = "MODERATE"
        else:
            hedge_level = "DIRECT"

        if gunning_fog > 18.0:
            obf_risk = "OBFUSCATED (High Complexity Smoke-screen)"
        elif gunning_fog > 14.0:
            obf_risk = "COMPLEX (Institutional Standard)"
        else:
            obf_risk = "CLEAR"

        return HedgingAnalysisResult(
            hedging_score=hedging_score,
            hedging_level=hedge_level,
            passive_evasion_score=passive_score,
            gunning_fog_index=gunning_fog,
            obfuscation_risk=obf_risk,
            detected_hedges=list(set(detected_hedges))[:5],
            detected_euphemisms=list(set(detected_euphemisms))[:5],
            passive_phrases=list(set(detected_passives))[:5],
        )
