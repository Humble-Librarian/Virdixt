"""
Virdixt Cross-Modal Dissonance Engine (Path #5)
Synthesizes acoustic prosody, hesitation latency, and linguistic deception (FinBERT + Hedging)
to detect executive dissonance during unscripted earnings call Q&A.

Optimized for Zero-VRAM / Low-Footprint CPU execution.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

from .prosody import AcousticStressExtractor, ProsodyMetrics
from nlp.linguistic_hedging import LinguisticHedgingDetector, HedgingAnalysisResult


@dataclass
class DissonanceResult:
    """Institutional Forensic Cross-Modal Evaluation."""
    cmdi_score: float                  # Cross-Modal Dissonance Index (0.0 to 10.0 scale)
    dissonance_tier: str               # CONGRUENT_CALM, MODERATE_TENSION, HIGH_DISSONANCE, ACUTE_DECEPTION_RISK
    acoustic_stress_score: float       # 0.0 to 100.0
    hesitation_latency: float          # Delay in seconds
    positive_sentiment_prob: float     # 0.0 to 1.0
    hedging_score: float               # 0.0 to 100.0
    forensic_flags: List[str] = field(default_factory=list)
    action_note: str = ""
    prosody_details: Optional[ProsodyMetrics] = None
    hedging_details: Optional[HedgingAnalysisResult] = None


class CrossModalDissonanceEngine:
    """
    Computes Cross-Modal Dissonance by diffing physiological acoustic stress
    against verbal cheerleading and epistemic hedging.
    """

    def __init__(
        self,
        stress_extractor: Optional[AcousticStressExtractor] = None,
        hedging_detector: Optional[LinguisticHedgingDetector] = None,
    ):
        self.stress_extractor = stress_extractor or AcousticStressExtractor()
        self.hedging_detector = hedging_detector or LinguisticHedgingDetector()

    def evaluate_turn(
        self,
        audio_data: Union[str, np.ndarray],
        text: str,
        hesitation_latency: float = 0.0,
        speaker_id: Optional[str] = None,
        sentiment_probs: Optional[Dict[str, float]] = None,
        sampling_rate: int = 16000,
    ) -> DissonanceResult:
        """
        Executes complete cross-modal audit on an unscripted executive Q&A response.
        """
        # 1. Acoustic Prosody & Vocal Stress
        prosody = self.stress_extractor.extract_stress(
            audio_data=audio_data,
            sampling_rate=sampling_rate,
            speaker_id=speaker_id,
        )

        # 2. Linguistic Hedging & Evasion
        hedging = self.hedging_detector.analyze(text)

        # 3. Sentiment Probabilities (FinBERT or heuristic fallback)
        if sentiment_probs is None:
            sentiment_probs = self._estimate_sentiment_heuristic(text)

        pos_prob = float(sentiment_probs.get("positive", 0.0))
        neg_prob = float(sentiment_probs.get("negative", 0.0))

        # 4. Behavioral Hesitation Penalty
        # Normal conversation pause: 0.2 - 0.8s. Severe cognitive stalling: > 2.0s
        hesitation_penalty = min(1.0, max(0.0, (hesitation_latency - 1.0) / 2.5))

        # 5. Combined Audio-Behavioral Stress (0.0 to 1.0)
        audio_stress_normalized = (prosody.stress_score / 100.0) * 0.75 + hesitation_penalty * 0.25

        # 6. Verbal Masking / Cheerleading Factor
        # High positive words + heavy hedging = attempting to mask underlying bad news
        verbal_masking = pos_prob * (1.0 + (hedging.hedging_score / 100.0))

        # 7. Cross-Modal Dissonance Index (CMDI)
        # CMDI surges when Audio Stress is high AND verbal positivity/hedging is high
        cmdi_raw = audio_stress_normalized * (1.0 + verbal_masking) * 2.5
        cmdi_score = round(float(np.clip(cmdi_raw, 0.0, 10.0)), 2)

        # 8. Tier Classification & Forensic Flags
        flags: List[str] = []
        if prosody.z_f0 > 2.0:
            flags.append(f"Acute pitch surge: F0 is +{prosody.z_f0:.1f}σ above speaker baseline")
        if prosody.z_jitter > 2.0:
            flags.append(f"Vocal fold tremor: Jitter is +{prosody.z_jitter:.1f}σ above baseline")
        if hesitation_latency >= 2.5:
            flags.append(f"Cognitive stalling: {hesitation_latency:.1f}s hesitation before answering")
        if pos_prob > 0.70 and prosody.stress_score > 60.0:
            flags.append("Verbal-Acoustic Divergence: Highly bullish verbal tone contradicts elevated vocal distress")
        if hedging.hedging_score >= 50.0:
            flags.append(f"Linguistic Evasion: Heavy epistemic hedging detected ({hedging.hedging_level})")

        if cmdi_score < 1.5:
            tier = "CONGRUENT_CALM"
            action_note = "Acoustic composure matches verbal content. Low deceptive risk."
        elif cmdi_score < 3.0:
            tier = "MODERATE_TENSION"
            action_note = "Noticeable vocal strain or mild evasion; monitor guidance sensitivity."
        elif cmdi_score < 5.0:
            tier = "HIGH_DISSONANCE"
            action_note = "Significant divergence: executive is experiencing acute cognitive/vocal distress while defending position."
        else:
            tier = "ACUTE_DECEPTION_RISK"
            action_note = "CRITICAL FORENSIC ALERT: Maximum cross-modal conflict. Narrative is likely actively constructed under distress."

        return DissonanceResult(
            cmdi_score=cmdi_score,
            dissonance_tier=tier,
            acoustic_stress_score=round(prosody.stress_score, 1),
            hesitation_latency=round(hesitation_latency, 2),
            positive_sentiment_prob=round(pos_prob, 3),
            hedging_score=round(hedging.hedging_score, 1),
            forensic_flags=flags,
            action_note=action_note,
            prosody_details=prosody,
            hedging_details=hedging,
        )

    def _estimate_sentiment_heuristic(self, text: str) -> Dict[str, float]:
        """
        Lightweight heuristic fallback for sentiment when ONNX FinBERT is offline.
        Uses financial lexicon cues.
        """
        text_lower = text.lower()
        bullish = ["confident", "strong", "accelerating", "momentum", "robust", "optimistic", "growth", "record", "exceeding"]
        bearish = ["decline", "pressure", "headwinds", "challenging", "slowdown", "deterioration", "deficit", "covenant", "impairment"]

        bull_count = sum(1 for w in bullish if w in text_lower)
        bear_count = sum(1 for w in bearish if w in text_lower)

        total = bull_count + bear_count
        if total == 0:
            return {"negative": 0.20, "neutral": 0.60, "positive": 0.20}

        p_pos = max(0.05, min(0.90, bull_count / (total + 1e-5)))
        p_neg = max(0.05, min(0.90, bear_count / (total + 1e-5)))
        p_neu = max(0.05, 1.0 - (p_pos + p_neg))
        s = p_pos + p_neg + p_neu
        return {"negative": p_neg / s, "neutral": p_neu / s, "positive": p_pos / s}
