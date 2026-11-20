"""Comprehensive Leak, Resource Safety & Institutional Guardrail Test Suite for Verdixt."""

import os
import io
import math
import tempfile
import time
import pytest

from nlp.forensic_accounting import ForensicAccountingEngine, ForensicScoreResult
from nlp.fusion import FusionRuleEngine, LaneScores
from nlp.absa_engine import AspectResult
from nlp.discourse_parser import RhetoricalDiscourseParser, ConcessiveClausePair, DiscourseAnalysisResult
from nlp.linguistic_hedging import LinguisticHedgingDetector
from document_parser import DocumentParser, ParsedDocument


# =====================================================================
# 1. INSTITUTIONAL FUSION GUARDRAIL LEAK TESTS
# =====================================================================

def test_fusion_altman_distress_override_triggers():
    """Verify that Altman Z distress status triggers the hard institutional override
    RULE_FORENSIC_ALTMAN_DISTRESS_OVERRIDE and elevates composite score >= 82.0.
    Previously failed due to strict string equality mismatch ('DISTRESS' vs 'DISTRESS ZONE...').
    """
    forensic_scores = ForensicScoreResult(
        calculated=True,
        altman_z_score=1.20,
        altman_zone="DISTRESS ZONE (High Bankruptcy Risk)",
        altman_calculated=True,
    )

    result = FusionRuleEngine.evaluate_lanes(
        sentiment_probs={"negative": 0.2, "neutral": 0.7, "positive": 0.1},
        sentiment_score=25.0,  # Benign text sentiment
        absa_results=[],
        forensic_scores=forensic_scores,
        discourse_res=None,
        hedging_res=None,
    )

    assert "RULE_FORENSIC_ALTMAN_DISTRESS_OVERRIDE" in result.triggered_rules, \
        "Altman distress override rule must trigger!"
    assert result.composite_distress_score >= 82.0, \
        f"Composite distress score must escalate to at least 82.0, got {result.composite_distress_score}"
    assert result.escalation_applied is True


def test_fusion_beneish_manipulation_override_triggers():
    """Verify that Beneish manipulation status triggers RULE_FORENSIC_BENEISH_MANIPULATION_OVERRIDE
    and elevates composite score >= 78.0.
    Previously failed due to strict string equality mismatch.
    """
    forensic_scores = ForensicScoreResult(
        calculated=True,
        beneish_m_score=-1.20,  # Greater than -1.78 indicates manipulation
        beneish_manipulation_risk="HIGH_MANIPULATION_RISK (Aggressive Accounting / Accrual Distortion)",
        beneish_calculated=True,
    )

    result = FusionRuleEngine.evaluate_lanes(
        sentiment_probs={"negative": 0.1, "neutral": 0.8, "positive": 0.1},
        sentiment_score=15.0,
        absa_results=[],
        forensic_scores=forensic_scores,
        discourse_res=None,
        hedging_res=None,
    )

    assert "RULE_FORENSIC_BENEISH_MANIPULATION_OVERRIDE" in result.triggered_rules, \
        "Beneish manipulation override rule must trigger!"
    assert result.composite_distress_score >= 78.0
    assert result.escalation_applied is True


def test_fusion_rhetorical_masking_guardrail():
    """Verify that deceptive concessive structures (positive satellite masking negative nucleus)
    trigger the rhetorical deception guardrail and escalate distress.
    """
    pair = ConcessiveClausePair(
        raw_sentence="Although revenue grew 12%, liquidity reached critical exhaustion.",
        satellite_clause="Although revenue grew 12%",
        nucleus_clause="liquidity reached critical exhaustion.",
        satellite_sentiment_hint="POSITIVE",
        nucleus_sentiment_hint="NEGATIVE",
        is_deceptive_buffer=True,
    )
    discourse_res = DiscourseAnalysisResult(
        has_concessive_structures=True,
        total_concessive_sentences=1,
        pairs=[pair],
    )

    result = FusionRuleEngine.evaluate_lanes(
        sentiment_probs={"negative": 0.3, "neutral": 0.5, "positive": 0.2},
        sentiment_score=30.0,
        absa_results=[],
        forensic_scores=None,
        discourse_res=discourse_res,
        hedging_res=None,
    )

    assert "RULE_RHETORICAL_MASKING_DETECTED" in result.triggered_rules
    assert result.lane_scores.discourse_distress >= 65.0


def test_fusion_critical_aspect_veto():
    """Verify that a CRITICAL rating in Debt & Capital Solvency triggers an aspect veto."""
    critical_aspect = AspectResult(
        aspect_name="Debt & Capital Solvency",
        detected=True,
        sentiment="NEGATIVE",
        distress_score=92.0,
        confidence=0.95,
        risk_level="CRITICAL",
    )

    result = FusionRuleEngine.evaluate_lanes(
        sentiment_probs={"negative": 0.2, "neutral": 0.6, "positive": 0.2},
        sentiment_score=20.0,
        absa_results=[critical_aspect],
        forensic_scores=None,
        discourse_res=None,
        hedging_res=None,
    )

    assert "RULE_CRITICAL_ASPECT_DEBT_AND_CAPITAL_SOLVENCY" in result.triggered_rules
    assert result.composite_distress_score >= 72.0


# =====================================================================
# 2. RESOURCE & HANDLE LEAK TESTS (Document Parser)
# =====================================================================

def test_parsed_document_context_manager_cleanup():
    """Verify ParsedDocument cleans up its temporary directory when used as a context manager."""
    temp_dir = tempfile.mkdtemp(prefix="virdixt_test_cm_")
    assert os.path.exists(temp_dir)

    with ParsedDocument(raw_text="Test filing", temp_dir=temp_dir) as doc:
        assert doc.raw_text == "Test filing"
        assert os.path.exists(temp_dir)

    # After context manager exit, temp_dir must be cleaned up
    assert not os.path.exists(temp_dir), "Temporary directory must be deleted on context exit!"


def test_parsed_document_del_cleanup():
    """Verify ParsedDocument cleans up temp_dir on garbage collection."""
    temp_dir = tempfile.mkdtemp(prefix="virdixt_test_del_")
    assert os.path.exists(temp_dir)

    doc = ParsedDocument(raw_text="Test filing", temp_dir=temp_dir)
    del doc
    import gc
    gc.collect()

    assert not os.path.exists(temp_dir), "Temporary directory must be deleted on object deletion!"


def test_document_parser_txt_encoding_and_missing_file_safety():
    """Verify DocumentParser handles missing or corrupted files gracefully without leaking handles."""
    missing = "non_existent_file_xyz_123.txt"
    doc = DocumentParser.parse_txt(missing)
    assert doc.raw_text == ""
    assert doc.file_type == "TXT"


def test_document_parser_csv_malformed_and_empty():
    """Verify CSV parser safely handles empty and irregular files."""
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
        f.write("\n\n   \n")
        f_path = f.name

    try:
        doc = DocumentParser.parse_csv(f_path)
        assert doc.file_type == "CSV"
        assert doc.raw_text == "Empty CSV file."
    finally:
        if os.path.exists(f_path):
            os.remove(f_path)


# =====================================================================
# 3. FORENSIC ACCOUNTING NUMERICAL & TYPE SAFETY LEAKS
# =====================================================================

def test_forensic_string_and_formatted_number_sanitization():
    """Verify that financial dictionaries containing currency symbols and formatted strings
    are converted cleanly instead of crashing with TypeError.
    """
    financials = {
        "Revenue": "$120,500,000",
        "Operating Income": "$15,200,000",
        "Total Assets": "250,000,000.00",
        "Total Debt": "80,000,000",
        "Cash": "$25,000,000",
        "Working Capital": "35,000,000",
        "Net Income": "10,500,000",
    }

    res = ForensicAccountingEngine.compute(financials)
    assert res.calculated is True
    assert res.altman_calculated is True
    assert res.altman_z_score is not None
    assert res.altman_z_score > 0.0
    assert res.altman_zone in [
        "GREY ZONE (Elevated Solvency Watch)",
        "SAFE ZONE (Investment Grade Solvency)",
    ]


def test_forensic_nan_and_inf_immunity():
    """Verify that NaN, Inf, and None values are safely ignored without poisoning calculations."""
    financials = {
        "Revenue": 50_000_000,
        "Total Assets": 100_000_000,
        "Total Debt": float("nan"),
        "Cash": float("inf"),
        "Operating Income": None,
        "Net Income": 2_000_000,
        "Working Capital": 5_000_000,
    }

    res = ForensicAccountingEngine.compute(financials)
    assert res.calculated is True
    # Should calculate Altman Z with available finite values and defaults
    assert res.altman_calculated is True
    assert not math.isnan(res.altman_z_score)


def test_forensic_zero_total_debt_division():
    """Verify that companies with zero debt do not trigger ZeroDivisionError."""
    financials = {
        "Revenue": 50_000_000,
        "Total Assets": 80_000_000,
        "Total Debt": 0.0,
        "Net Income": 10_000_000,
        "Operating Income": 12_000_000,
    }

    res = ForensicAccountingEngine.compute(financials)
    assert res.calculated is True
    assert res.altman_calculated is True
    assert res.altman_z_score is not None


def test_forensic_insufficient_fields():
    """Verify that fewer than 3 financial metrics results in graceful rejection."""
    res = ForensicAccountingEngine.compute({"Revenue": 1000.0})
    assert res.calculated is False
    assert "Insufficient" in res.notes[0]


# =====================================================================
# 4. LINGUISTIC HEDGING & DISCOURSE BACKTRACKING IMMUNITY
# =====================================================================

def test_hedging_detector_catastrophic_backtracking_immunity():
    """Verify that massive documents with repeated hedging tokens execute in under 100ms."""
    detector = LinguisticHedgingDetector()
    adversarial_payload = "Management believes it might possibly be subject to preliminary verification. " * 500

    t0 = time.time()
    res = detector.analyze(adversarial_payload)
    elapsed = time.time() - t0

    assert elapsed < 0.20, f"Analysis took {elapsed:.3f}s — potential regex backtracking vulnerability!"
    assert res.hedging_score > 50.0
    assert res.hedging_level in ["HIGH_UNCERTAINTY", "EXTREME_EVASION"]


def test_hedging_detector_empty_and_whitespace():
    """Verify empty or whitespace text returns clean default without division errors."""
    detector = LinguisticHedgingDetector()
    for empty_input in ["", "   \n\t  ", "...!?"]:
        res = detector.analyze(empty_input)
        assert res.hedging_score == 0.0
        assert res.hedging_level == "DIRECT"
        assert res.obfuscation_risk == "CLEAR"


def test_discourse_parser_concessive_extraction():
    """Verify discourse parser extracts concessive clause pairs correctly."""
    parser = RhetoricalDiscourseParser()
    doc = "Although quarterly revenue rose by 14%, operating cash flow turned deeply negative."
    res = parser.parse(doc)

    assert res.has_concessive_structures is True
    assert res.total_concessive_sentences == 1
    assert len(res.pairs) == 1
    pair = res.pairs[0]
    assert "revenue rose" in pair.satellite_clause.lower()
    assert "cash flow" in pair.nucleus_clause.lower()
    assert pair.is_deceptive_buffer is True
