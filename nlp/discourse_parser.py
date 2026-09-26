"""
Rhetorical Structure Theory (RST) & Concessive Discourse Parser
Deconstructs complex multi-clause financial sentences into Discourse Nuclei
(primary economic reality) vs Discourse Satellites (concessive/rhetorical buffers).
"""

import re
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional


@dataclass
class ConcessiveClausePair:
    raw_sentence: str
    satellite_clause: str  # The concessive buffer (e.g. "Although revenue grew 5%")
    nucleus_clause: str  # The dominant economic truth (e.g. "losses widened 40%")
    satellite_sentiment_hint: str  # POSITIVE / NEGATIVE
    nucleus_sentiment_hint: str  # POSITIVE / NEGATIVE
    is_deceptive_buffer: bool  # True if positive satellite masks negative nucleus


@dataclass
class DiscourseAnalysisResult:
    has_concessive_structures: bool
    total_concessive_sentences: int
    pairs: List[ConcessiveClausePair] = field(default_factory=list)
    dominant_nucleus_focus: str = ""


class RhetoricalDiscourseParser:
    """Extracts and separates Nuclei from Satellites in financial disclosures."""

    CONCESSIVE_MARKERS = [
        r"\b(although|even though|though|whereas|while|whilst|in spite of|despite|notwithstanding)\b",
    ]
    ADVERSATIVE_CONNECTORS = [
        r"\b(however|nonetheless|nevertheless|yet|but|conversely|on the other hand)\b",
    ]

    def __init__(self):
        self.concessive_reg = re.compile(
            r"^(?:although|even though|though|whereas|while|despite|in spite of|notwithstanding)\s+(.*?),\s*(.*)$",
            re.IGNORECASE
        )
        self.adversative_reg = re.compile(
            r"^(.*?),\s*(?:however|nonetheless|nevertheless|yet|but)\s+(.*)$",
            re.IGNORECASE
        )

    def _quick_sentiment_hint(self, clause: str) -> str:
        """Fast heuristic polarity check for clause classification."""
        pos_words = {"growth", "grew", "surpassed", "record", "profit", "expansion", "positive", "increased", "gained", "beat"}
        neg_words = {"decline", "declined", "loss", "losses", "breach", "default", "drain", "surged", "contracted", "headwind", "adverse", "impairment"}

        words = set(re.findall(r"\b[a-z]+\b", clause.lower()))
        pos_match = len(words.intersection(pos_words))
        neg_match = len(words.intersection(neg_words))

        if neg_match > pos_match:
            return "NEGATIVE"
        elif pos_match > neg_match:
            return "POSITIVE"
        return "NEUTRAL"

    def parse(self, document: str) -> DiscourseAnalysisResult:
        """Parses document text and extracts all concessive and adversative clause pairs."""
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", document) if len(s.strip()) > 15]
        pairs: List[ConcessiveClausePair] = []

        for s in sentences:
            # Check for leading concessive clause: "Although X, Y"
            m_conc = self.concessive_reg.match(s)
            if m_conc:
                satellite = m_conc.group(1).strip()
                nucleus = m_conc.group(2).strip()
                sat_sent = self._quick_sentiment_hint(satellite)
                nuc_sent = self._quick_sentiment_hint(nucleus)
                is_deceptive = (sat_sent == "POSITIVE" and nuc_sent == "NEGATIVE")

                pairs.append(
                    ConcessiveClausePair(
                        raw_sentence=s,
                        satellite_clause=satellite,
                        nucleus_clause=nucleus,
                        satellite_sentiment_hint=sat_sent,
                        nucleus_sentiment_hint=nuc_sent,
                        is_deceptive_buffer=is_deceptive,
                    )
                )
                continue

            # Check for midpoint adversative connector: "X, however Y"
            m_adv = self.adversative_reg.match(s)
            if m_adv:
                nucleus = m_adv.group(2).strip()
                satellite = m_adv.group(1).strip()
                sat_sent = self._quick_sentiment_hint(satellite)
                nuc_sent = self._quick_sentiment_hint(nucleus)
                is_deceptive = (sat_sent == "POSITIVE" and nuc_sent == "NEGATIVE")

                pairs.append(
                    ConcessiveClausePair(
                        raw_sentence=s,
                        satellite_clause=satellite,
                        nucleus_clause=nucleus,
                        satellite_sentiment_hint=sat_sent,
                        nucleus_sentiment_hint=nuc_sent,
                        is_deceptive_buffer=is_deceptive,
                    )
                )

        dominant_focus = ""
        if pairs:
            nuclei = [p.nucleus_clause for p in pairs if p.is_deceptive_buffer]
            if nuclei:
                dominant_focus = " ".join(nuclei[:2])

        return DiscourseAnalysisResult(
            has_concessive_structures=len(pairs) > 0,
            total_concessive_sentences=len(pairs),
            pairs=pairs,
            dominant_nucleus_focus=dominant_focus,
        )
