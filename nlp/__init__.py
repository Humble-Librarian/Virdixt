"""
Virdixt Advanced NLP & Computational Linguistics Subsystem
Aspect-Based Sentiment Analysis (ABSA), Linguistic Hedging Detection,
Rhetorical Discourse Parsing, and Quantitative Forensic Accounting.
"""

from .absa_engine import ABSAEngine, AspectResult
from .linguistic_hedging import LinguisticHedgingDetector, HedgingAnalysisResult
from .discourse_parser import RhetoricalDiscourseParser, DiscourseAnalysisResult
from .forensic_accounting import ForensicAccountingEngine, ForensicScoreResult

__all__ = [
    "ABSAEngine",
    "AspectResult",
    "LinguisticHedgingDetector",
    "HedgingAnalysisResult",
    "RhetoricalDiscourseParser",
    "DiscourseAnalysisResult",
    "ForensicAccountingEngine",
    "ForensicScoreResult",
]
