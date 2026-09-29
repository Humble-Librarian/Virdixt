import sys
import os
import io
import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from infer import FinancialAdvisor, AnchorTokenPruner, RiskGrade, ExposureTier, PolicyActionFlag, POLICY_TABLE, display_advisor_report, NoulResult
from rich.console import Console

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
            debt_covenant_breach_risk=(0.02 * 0.85 + 0.08 * 0.15) * 100.0,
            growth_expansion_momentum=0.90 * 100.0,
            capital_return_sustainable=(0.90 * 0.80 + 0.08 * 0.20) * 100.0,
        )

@pytest.fixture
def advisor():
    return FinancialAdvisor(FakeSystem1(), AnchorTokenPruner())

BENIGN_TEXT = "The company reported excellent revenue growth and positive cash flows across all sectors."
EVASIVE_TEXT = "Management believes that the strategic realignment might possibly be a one-off adjustment due to transitory pressure and macro headwinds. It could potentially lead to right-sizing of the business. We tentatively expect approximate results to follow. Adjustments were made."

DISTRESS_DICT = {
    "Total_Assets": 150000000,
    "Revenue": 92000000,
    "Gross_Profit": 28000000,
    "Operating_Expenses": 48000000,
    "Operating_Income": -20000000,
    "Total_Debt": 145000000,
    "Cash_and_Cash_Equivalents": 4200000,
    "Burn_Rate": 6500000
}

SAFE_DICT = {
    "Revenue": 150000000,
    "Operating_Income": 50000000,
    "Total_Debt": 10000000,
    "Cash_and_Cash_Equivalents": 80000000,
    "Total_Assets": 300000000,
    "Net_Income": 40000000
}

def test_distress_dict_escalates_minimal_to_critical(advisor, capsys):
    res1 = advisor.advise(BENIGN_TEXT, financial_dict=DISTRESS_DICT)
    assert res1.base_grade == RiskGrade.MINIMAL
    assert res1.risk_grade == RiskGrade.CRITICAL
    assert res1.exposure_tier == ExposureTier.TIER_4_BLOCKED
    assert res1.action_flag == PolicyActionFlag.FREEZE_PURCHASE_ORDERS
    assert len(res1.override_reasons) == 1
    assert "Altman Z in distress zone" in res1.override_reasons[0]
    
    # Check POLICY_TABLE consistency
    exp_tier, act_flag, recs = POLICY_TABLE[res1.risk_grade]
    assert res1.exposure_tier == exp_tier
    assert res1.action_flag == act_flag
    assert res1.action_recommendations == recs

    display_advisor_report(res1, source_title="Case 1 Test")
    captured = capsys.readouterr().out
    captured_clean = __import__("re").sub(r"\x1b\[.*?m", "", captured)
    assert "Escalated from MINIMAL to CRITICAL" in captured_clean

def test_safe_dict_stays_minimal(advisor):
    res2 = advisor.advise(BENIGN_TEXT, financial_dict=SAFE_DICT)
    assert res2.base_grade == RiskGrade.MINIMAL
    assert res2.risk_grade == RiskGrade.MINIMAL
    assert not res2.override_reasons

    exp_tier, act_flag, recs = POLICY_TABLE[res2.risk_grade]
    assert res2.exposure_tier == exp_tier
    assert res2.action_flag == act_flag
    assert res2.action_recommendations == recs

def test_no_dict_stays_minimal(advisor):
    res3 = advisor.advise(BENIGN_TEXT)
    assert res3.base_grade == RiskGrade.MINIMAL
    assert res3.risk_grade == RiskGrade.MINIMAL
    assert not res3.override_reasons

    exp_tier, act_flag, recs = POLICY_TABLE[res3.risk_grade]
    assert res3.exposure_tier == exp_tier
    assert res3.action_flag == act_flag
    assert res3.action_recommendations == recs

def test_evasive_text_escalates_minimal_to_warning(advisor):
    res4 = advisor.advise(EVASIVE_TEXT)
    assert res4.base_grade == RiskGrade.MINIMAL
    assert res4.risk_grade == RiskGrade.WARNING
    assert len(res4.override_reasons) == 1
    assert "Extreme hedging language" in res4.override_reasons[0]

    exp_tier, act_flag, recs = POLICY_TABLE[res4.risk_grade]
    assert res4.exposure_tier == exp_tier
    assert res4.action_flag == act_flag
    assert res4.action_recommendations == recs

def test_policy_table_consistency_across_all_grades(advisor):
    for g in RiskGrade:
        assert g in POLICY_TABLE
        assert len(POLICY_TABLE[g]) == 3
