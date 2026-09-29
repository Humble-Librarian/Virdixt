import sys
import os
import pytest
import time
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from infer import evaluate_completeness, FinancialAdvisor, AnchorTokenPruner, NoulResult
from nlp.absa_engine import AspectResult
from nlp.forensic_accounting import ForensicScoreResult

class FakeSystem1:
    def choice(self, text: str, temperature: float = 1.25) -> str:
        return "POSITIVE"

    def score(self, text: str) -> float:
        return 0.0

    def get_calibrated_probs(self, text: str, temperature: float = 1.25):
        return {"negative": 0.02, "neutral": 0.08, "positive": 0.90}

    def noul(self, text: str):
        return NoulResult(
            liquidity_distress=0.02 * 100.0,
            debt_covenant_breach_risk=0.03 * 100.0,
            growth_expansion_momentum=0.90 * 100.0,
            capital_return_sustainable=0.73 * 100.0,
        )

@pytest.fixture
def advisor():
    return FinancialAdvisor(FakeSystem1(), AnchorTokenPruner())


def test_evaluate_completeness_speed_under_500ms():
    """Verify completeness calculation completes in under 1ms (far below the 500ms constraint)."""
    t0 = time.perf_counter()
    for _ in range(1000):
        evaluate_completeness(document="Hello world "*100, pruned_text="Hello world "*100, absa_results=None, forensic_res=None)
    t1 = time.perf_counter()
    avg_ms = ((t1 - t0) / 1000.0) * 1000.0
    
    assert avg_ms < 1.0, f"Average execution speed was {avg_ms:.4f} ms, expected < 1 ms"


def test_thin_document_returns_unknown_state(advisor):
    """A document with almost zero active signals (< 2) returns THIN state."""
    res = advisor.advise("Hello")
    assert res.completeness == "THIN"
    assert "Very short input" in res.completeness_notes


def test_full_document_returns_signals_count(advisor):
    """A rich document with financial metrics returns FULL availability."""
    text = (
        "The company reported revenue of $92M and operating profit. "
        "Management believes that strategic realignment might possibly be a temporary issue. "
        "Although gross margin declined, net revenue expanded. " * 10
    )
    distress_dict = {
        "Total_Assets": 300000000,
        "Revenue": 92000000,
        "Gross_Profit": 28000000,
        "Operating_Income": -20000000,
        "Net_Income": -15000000,
        "Total_Debt": 145000000,
        "Cash_and_Cash_Equivalents": 4200000,
    }
    
    res = advisor.advise(text, financial_dict=distress_dict)
    
    assert res.completeness == "FULL"
    assert res.completeness == "FULL"

def test_pruning_note_does_not_penalize_label():
    """Verify that a pruning note alone does not downgrade FULL to PARTIAL."""
    # A rich document that gets pruned, but otherwise has no missing signals
    doc = "Hello "*100
    pruned = "Hello "*50
    # Provide fake objects that pass all signal checks
    class FakeForensic:
        calculated = True
    class FakeABSA:
        detected = True
    absa_res = [FakeABSA(), FakeABSA()]
    
    label, notes = evaluate_completeness(doc, pruned, absa_res, FakeForensic())
    
    assert label == "FULL"
    assert len(notes) == 1
    assert "Document was reduced from 100 to 50 words by keyword pruning" in notes[0]

def test_advisor_result_to_dict_serializable(advisor):
    res = advisor.advise("Hello")
    d = res.to_dict()
    json_str = json.dumps(d)
    assert isinstance(json_str, str)
    assert d["completeness"] == "THIN"
    # Ensure Enum is stringified
    assert isinstance(d["risk_grade"], str)
    assert d["risk_grade"] == res.risk_grade.value
