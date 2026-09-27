"""
5-Lane Composite Fusion & Institutional Rule Engine
Fuses signals across all 5 analytical dimensions:
1. Global Calibrated Sentiment & NOUL (FinBERT)
2. Aspect-Based Operational Risk (ABSA)
3. Quantitative Forensic Accounting (Altman Z / Beneish M / Piotroski F)
4. Rhetorical Structure & Nucleus Masking (RST Discourse)
5. Linguistic Hedging, Evasion & Obfuscation (Hedging / Gunning-Fog)

Calculates composite institutional distress scores and applies deterministic
governance guardrails with sub-millisecond in-memory execution.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum

from .absa_engine import AspectResult
from .forensic_accounting import ForensicScoreResult
from .discourse_parser import DiscourseAnalysisResult
from .linguistic_hedging import HedgingAnalysisResult


@dataclass
class LaneScores:
    sentiment_distress: float = 0.0
    aspect_distress: float = 0.0
    forensic_distress: float = 0.0
    discourse_distress: float = 0.0
    hedging_distress: float = 0.0


@dataclass
class FusionEvaluation:
    composite_distress_score: float
    raw_weighted_score: float
    lane_scores: LaneScores
    active_weights: Dict[str, float]
    triggered_rules: List[str]
    escalation_applied: bool
    governance_notes: List[str]


class FusionRuleEngine:
    """
    Sub-millisecond 5-lane fusion engine executing deterministic institutional guardrails
    and multi-factor risk calibration.
    """

    # Baseline theoretical weight distribution
    BASE_WEIGHTS = {
        "sentiment": 0.30,
        "aspect": 0.25,
        "forensic": 0.25,
        "discourse": 0.10,
        "hedging": 0.10,
    }

    # Weights when quantitative balance-sheet forensics are not provided
    TEXT_ONLY_WEIGHTS = {
        "sentiment": 0.35,
        "aspect": 0.35,
        "forensic": 0.00,
        "discourse": 0.15,
        "hedging": 0.15,
    }

    @classmethod
    def evaluate_lanes(
        cls,
        sentiment_probs: Dict[str, float],
        sentiment_score: float,
        absa_results: List[AspectResult],
        forensic_scores: Optional[ForensicScoreResult],
        discourse_res: Optional[DiscourseAnalysisResult],
        hedging_res: Optional[HedgingAnalysisResult],
    ) -> FusionEvaluation:
        """
        Synthesizes all 5 intelligence lanes into a calibrated institutional verdict.
        """
        # --- 1. LANE NORMALIZATIONS (0.0 to 100.0 scale) ---
        
        # Lane 1: Global Sentiment & NOUL
        s_sentiment = float(sentiment_score)

        # Lane 2: Aspect-Based Operational Risk (ABSA)
        s_aspect = 0.0
        detected_aspects = [a for a in absa_results if a.detected]
        critical_aspect_names = {"Debt & Capital Solvency", "Liquidity & Cash Flow", "Audit & Governance Risk"}
        
        if detected_aspects:
            # Weight critical solvency/governance aspects heavier than general growth aspects
            aspect_scores = []
            for a in detected_aspects:
                weight = 1.5 if a.aspect_name in critical_aspect_names else 1.0
                aspect_scores.append(a.distress_score * weight)
            s_aspect = min(100.0, max(aspect_scores))  # Peak vulnerability drives aspect risk
        else:
            s_aspect = s_sentiment * 0.7  # Default fallback if no specific aspects tagged

        # Lane 3: Forensic Accounting Benchmarks
        s_forensic = 0.0
        has_forensics = forensic_scores is not None and forensic_scores.calculated

        if has_forensics:
            f_components = []
            # Altman Z Component
            if forensic_scores.altman_z_score is not None:
                z = forensic_scores.altman_z_score
                if z < 1.81:
                    f_components.append(95.0)  # Distress Zone
                elif z < 2.99:
                    f_components.append(50.0)  # Grey Zone
                else:
                    f_components.append(10.0)  # Safe Zone

            # Beneish M Component
            if forensic_scores.beneish_m_score is not None:
                m = forensic_scores.beneish_m_score
                if m > -1.78:
                    f_components.append(90.0)  # High Manipulation Risk
                else:
                    f_components.append(15.0)

            # Piotroski F Component
            if forensic_scores.piotroski_f_score is not None:
                f_score = forensic_scores.piotroski_f_score
                f_components.append((9 - f_score) * (100.0 / 9.0))

            s_forensic = sum(f_components) / len(f_components) if f_components else 0.0

        # Lane 4: Rhetorical Discourse Parsing (RST Concessive & Masking)
        s_discourse = 0.0
        deceptive_masking_count = 0
        if discourse_res and discourse_res.has_concessive_structures:
            deceptive_pairs = [p for p in discourse_res.pairs if p.is_deceptive_buffer]
            deceptive_masking_count = len(deceptive_pairs)
            if deceptive_masking_count > 0:
                s_discourse = min(100.0, 40.0 + (deceptive_masking_count * 25.0))
            else:
                s_discourse = min(100.0, discourse_res.total_concessive_sentences * 12.0)

        # Lane 5: Linguistic Hedging & Deception Detection
        s_hedging = 0.0
        if hedging_res:
            base_hedge = hedging_res.hedging_score
            passive_adj = hedging_res.passive_evasion_score * 0.3
            fog_adj = 20.0 if hedging_res.gunning_fog_index > 18.0 else (10.0 if hedging_res.gunning_fog_index > 14.0 else 0.0)
            euphemism_adj = min(25.0, len(hedging_res.detected_euphemisms) * 8.0)
            s_hedging = min(100.0, base_hedge + passive_adj + fog_adj + euphemism_adj)

        lane_scores = LaneScores(
            sentiment_distress=s_sentiment,
            aspect_distress=s_aspect,
            forensic_distress=s_forensic,
            discourse_distress=s_discourse,
            hedging_distress=s_hedging,
        )

        # --- 2. MULTI-FACTOR WEIGHTED FUSION ---
        weights = cls.BASE_WEIGHTS if has_forensics else cls.TEXT_ONLY_WEIGHTS
        raw_weighted = (
            weights["sentiment"] * s_sentiment
            + weights["aspect"] * s_aspect
            + weights["forensic"] * s_forensic
            + weights["discourse"] * s_discourse
            + weights["hedging"] * s_hedging
        )

        composite_score = raw_weighted
        triggered_rules: List[str] = []
        governance_notes: List[str] = []
        escalation_applied = False

        # --- 3. DETERMINISTIC INSTITUTIONAL OVERRIDE GUARDRAILS ---

        # Guardrail 1: Forensic Insolvency / Manipulation Hard Veto
        if has_forensics:
            if forensic_scores.altman_zone == "DISTRESS":
                triggered_rules.append("RULE_FORENSIC_ALTMAN_DISTRESS_OVERRIDE")
                governance_notes.append("Quantitative Altman Z-Score indicates severe insolvency risk (<1.81).")
                composite_score = max(composite_score, 82.0)
                escalation_applied = True

            if forensic_scores.beneish_manipulation_risk == "HIGH_MANIPULATION_RISK":
                triggered_rules.append("RULE_FORENSIC_BENEISH_MANIPULATION_OVERRIDE")
                governance_notes.append("Beneish M-Score flagged high probability of financial earnings manipulation (>-1.78).")
                composite_score = max(composite_score, 78.0)
                escalation_applied = True

        # Guardrail 2: Critical Aspect Failure (Debt / Liquidity / Audit)
        for a in absa_results:
            if a.detected and a.risk_level == "CRITICAL" and a.aspect_name in critical_aspect_names:
                triggered_rules.append(f"RULE_CRITICAL_ASPECT_{a.aspect_name.upper().replace(' ', '_').replace('&', 'AND')}")
                governance_notes.append(f"Critical distress identified specifically in '{a.aspect_name}' aspect ({a.distress_score:.1f}%).")
                composite_score = max(composite_score, 72.0)
                escalation_applied = True

        # Guardrail 3: Rhetorical Deception Masking Filter
        if deceptive_masking_count >= 1:
            triggered_rules.append("RULE_RHETORICAL_MASKING_DETECTED")
            governance_notes.append(f"Detected {deceptive_masking_count} deceptive concessive buffer(s) masking core negative realities.")
            # If sentiment was overly optimistic, apply an inversion penalty
            if sentiment_probs.get("positive", 0.0) > 0.40:
                composite_score += 15.0
                escalation_applied = True

        # Guardrail 4: Executive Evasion & Obfuscation Escalation
        if hedging_res:
            if hedging_res.hedging_level in ["EXTREME_EVASION", "HIGH_UNCERTAINTY"] and hedging_res.gunning_fog_index > 16.0:
                triggered_rules.append("RULE_LINGUISTIC_SMOKESCREEN_ESCALATION")
                governance_notes.append("High syntactic obfuscation (Fog > 16) combined with extreme epistemic hedging detected.")
                composite_score += 10.0
                escalation_applied = True

        # Cap composite score strictly inside [0.0, 100.0]
        composite_score = max(0.0, min(100.0, composite_score))

        return FusionEvaluation(
            composite_distress_score=composite_score,
            raw_weighted_score=raw_weighted,
            lane_scores=lane_scores,
            active_weights=weights,
            triggered_rules=triggered_rules,
            escalation_applied=escalation_applied,
            governance_notes=governance_notes,
        )
