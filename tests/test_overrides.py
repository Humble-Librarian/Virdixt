import sys
import os
import types

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from infer import apply_overrides, RiskGrade, POLICY_TABLE

def test_overrides():
    # no forensic, no hedging -> grade unchanged, reasons empty
    grade, reasons = apply_overrides(RiskGrade.MINIMAL, None, None)
    assert grade == RiskGrade.MINIMAL
    assert not reasons

    # forensic distress zone but calculated=False -> NO override
    forensic = types.SimpleNamespace(calculated=False, altman_zone="DISTRESS ZONE (High Bankruptcy Risk)")
    grade, reasons = apply_overrides(RiskGrade.MINIMAL, forensic, None)
    assert grade == RiskGrade.MINIMAL

    # Altman GREY zone -> no override
    forensic = types.SimpleNamespace(calculated=True, altman_zone="GREY ZONE")
    grade, reasons = apply_overrides(RiskGrade.MINIMAL, forensic, None)
    assert grade == RiskGrade.MINIMAL
    
    # Altman distress + base MINIMAL -> CRITICAL, one reason
    forensic = types.SimpleNamespace(calculated=True, altman_zone="DISTRESS ZONE (High Bankruptcy Risk)")
    grade, reasons = apply_overrides(RiskGrade.MINIMAL, forensic, None)
    assert grade == RiskGrade.CRITICAL
    assert len(reasons) == 1
    assert "Altman Z in distress zone" in reasons[0]

    # extreme hedging + base MINIMAL -> WARNING
    hedging = types.SimpleNamespace(hedging_level="EXTREME_EVASION")
    grade, reasons = apply_overrides(RiskGrade.MINIMAL, None, hedging)
    assert grade == RiskGrade.WARNING
    assert len(reasons) == 1

    # extreme hedging + base CRITICAL -> stays CRITICAL (never lowered)
    grade, reasons = apply_overrides(RiskGrade.CRITICAL, None, hedging)
    assert grade == RiskGrade.CRITICAL
    assert len(reasons) == 1

    # both triggers + base MONITOR -> CRITICAL, two reasons
    forensic = types.SimpleNamespace(calculated=True, altman_zone="DISTRESS ZONE (High Bankruptcy Risk)")
    hedging = types.SimpleNamespace(hedging_level="EXTREME_EVASION")
    grade, reasons = apply_overrides(RiskGrade.MONITOR, forensic, hedging)
    assert grade == RiskGrade.CRITICAL
    assert len(reasons) == 2

    # for every RiskGrade, tier/action/recs come from one consistent table row
    for g in RiskGrade:
        assert g in POLICY_TABLE
        assert len(POLICY_TABLE[g]) == 3

    print("All tests passed.")

if __name__ == "__main__":
    test_overrides()
