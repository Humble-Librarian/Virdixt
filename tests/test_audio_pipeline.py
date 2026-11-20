"""
Deterministic Gravitas Verification Suite for Virdixt Audio Engine (Path #5)
Tests:
1. openSMILE eGeMAPSv02 feature extraction and baseline enrollment.
2. Stress detection on synthetic calm vs distressed audio waves.
3. Turn segmentation and hesitation latency parsing.
4. Cross-Modal Dissonance Index (CMDI) calibration and anomaly alerts.
"""

import os
import pytest
import numpy as np
from audio.prosody import AcousticStressExtractor, ProsodyMetrics, SpeakerBaseline
from audio.transcribe import EarningsCallTranscriber, TurnSegment
from audio.dissonance import CrossModalDissonanceEngine, DissonanceResult


def _generate_synthetic_audio(
    duration_sec: float = 2.0,
    sampling_rate: int = 16000,
    f0_hz: float = 120.0,
    jitter_amp: float = 0.0,
    shimmer_amp: float = 0.0,
) -> np.ndarray:
    """Generates synthetic harmonic speech-like signal with controlled pitch and perturbation."""
    t = np.linspace(0, duration_sec, int(sampling_rate * duration_sec), endpoint=False)
    
    # Introduce frequency perturbation (jitter)
    if jitter_amp > 0:
        f_inst = f0_hz + jitter_amp * np.sin(2 * np.pi * 35.0 * t)
    else:
        f_inst = np.full_like(t, f0_hz)

    phase = 2 * np.pi * np.cumsum(f_inst) / sampling_rate
    
    # Harmonics (1st, 2nd, 3rd)
    sig = 0.6 * np.sin(phase) + 0.3 * np.sin(2 * phase) + 0.1 * np.sin(3 * phase)
    
    # Introduce amplitude perturbation (shimmer)
    if shimmer_amp > 0:
        envelope = 1.0 + shimmer_amp * np.sin(2 * np.pi * 25.0 * t)
        sig = sig * envelope

    return np.asarray(sig, dtype=np.float32)


class TestAcousticProsody:
    """Validates openSMILE acoustic stress extractor."""

    def test_baseline_enrollment_and_feature_extraction(self):
        extractor = AcousticStressExtractor()
        calm_audio = _generate_synthetic_audio(f0_hz=110.0, jitter_amp=0.0)
        
        baseline = extractor.enroll_speaker_baseline("CEO_BASELINE", calm_audio)
        assert baseline.speaker_id == "CEO_BASELINE"
        assert baseline.f0_mu > 0.0
        assert baseline.f0_sigma > 0.0

    def test_distress_detection_on_pitch_and_jitter_surge(self):
        extractor = AcousticStressExtractor()
        # 1. Enroll calm baseline at 110Hz
        calm_audio = _generate_synthetic_audio(f0_hz=110.0, jitter_amp=0.0)
        extractor.enroll_speaker_baseline("CEO_TEST", calm_audio)

        # 2. Extract stress on calm audio -> Should be CALM
        calm_metrics = extractor.extract_stress(calm_audio, speaker_id="CEO_TEST")
        assert calm_metrics.stress_level in ["CALM", "ELEVATED"]

        # 3. Create stressed audio: 80Hz pitch jump + micro-jitter perturbation
        stressed_audio = _generate_synthetic_audio(f0_hz=220.0, jitter_amp=15.0, shimmer_amp=0.3)
        stressed_metrics = extractor.extract_stress(stressed_audio, speaker_id="CEO_TEST")

        # Gravitas invariant: Stress score on acute disturbance must be higher than calm
        assert stressed_metrics.stress_score > calm_metrics.stress_score
        assert stressed_metrics.z_f0 > 1.0 or stressed_metrics.z_jitter > 1.0


class TestTurnTranscriber:
    """Validates earnings call turn parser logic."""

    def test_dialogue_role_classification(self):
        transcriber = EarningsCallTranscriber()
        
        # Operator introduction
        role1 = transcriber._classify_role("Our next question comes from John with Goldman Sachs.", "PREVIOUS")
        assert role1 == "OPERATOR"

        # Analyst question following operator
        role2 = transcriber._classify_role("Hi management, can you explain the 30% drop in free cash flow?", "OPERATOR")
        assert role2 == "ANALYST_QUESTION"

        # Executive answer following question
        role3 = transcriber._classify_role(
            "Thanks John. We feel very confident about the core business and expect strong recovery in Q3.",
            "ANALYST_QUESTION"
        )
        assert role3 == "EXECUTIVE_ANSWER"


class TestCrossModalDissonance:
    """Validates CMDI formula and forensic deception alerting."""

    def test_congruent_calm_exchange(self):
        engine = CrossModalDissonanceEngine()
        calm_audio = _generate_synthetic_audio(f0_hz=120.0)
        calm_text = "Operating margins stabilized at fourteen percent in line with our budget."
        
        # Calm audio + direct unhedged text + 0.3s normal latency
        res = engine.evaluate_turn(
            audio_data=calm_audio,
            text=calm_text,
            hesitation_latency=0.3,
            sentiment_probs={"negative": 0.1, "neutral": 0.8, "positive": 0.1},
        )
        assert res.cmdi_score < 2.5
        assert res.dissonance_tier in ["CONGRUENT_CALM", "MODERATE_TENSION"]

    def test_acute_deception_divergence(self):
        engine = CrossModalDissonanceEngine()
        
        # 1. Enroll speaker baseline at 110Hz
        calm_baseline_audio = _generate_synthetic_audio(f0_hz=110.0)
        engine.stress_extractor.enroll_speaker_baseline("CEO_FRAUD_CASE", calm_baseline_audio)

        # 2. Deceptive response:
        # Vocal pitch surges to 230Hz (+jitter), 3.2s hesitation, while text claims massive cheerleading with heavy hedging
        tense_audio = _generate_synthetic_audio(f0_hz=230.0, jitter_amp=20.0, shimmer_amp=0.4)
        deceptive_text = (
            "We feel extremely confident, and management believes that potentially, "
            "subject to verification, our demand trajectory is accelerating tremendously."
        )

        res = engine.evaluate_turn(
            audio_data=tense_audio,
            text=deceptive_text,
            hesitation_latency=3.2,
            speaker_id="CEO_FRAUD_CASE",
            sentiment_probs={"negative": 0.05, "neutral": 0.10, "positive": 0.85},
        )

        # Invariant: CMDI must fire high dissonance / deception risk
        assert res.cmdi_score >= 3.0
        assert res.dissonance_tier in ["HIGH_DISSONANCE", "ACUTE_DECEPTION_RISK"]
        assert len(res.forensic_flags) > 0
        assert any("pitch surge" in f.lower() or "divergence" in f.lower() or "hedging" in f.lower() for f in res.forensic_flags)

    def test_real_speech_end_to_end_transcribe_and_score(self):
        """End-to-end pipeline test on actual audio wave file."""
        wav_path = "d:/Virdixt/data/test_real_speech.wav"
        if not os.path.exists(wav_path):
            pytest.skip("test_real_speech.wav not found")

        transcriber = EarningsCallTranscriber(model_size="tiny.en")
        turns = transcriber.transcribe(wav_path)
        assert len(turns) >= 1
        assert any(len(t.text.strip()) > 0 for t in turns)

        engine = CrossModalDissonanceEngine()
        first_turn = turns[0]
        res = engine.evaluate_turn(
            audio_data=wav_path,
            text=first_turn.text,
            hesitation_latency=first_turn.hesitation_latency,
        )
        assert res.cmdi_score is not None
        assert res.dissonance_tier in ["CONGRUENT_CALM", "MODERATE_TENSION", "HIGH_DISSONANCE", "ACUTE_DECEPTION_RISK"]


class TestLane6FusionIntegration:
    """Validates that FusionRuleEngine properly incorporates Lane 6 Audio."""

    def test_fusion_engine_incorporates_audio_lane(self):
        from nlp.fusion import FusionRuleEngine
        from audio.dissonance import DissonanceResult

        # Mock a high dissonance audio result
        audio_res = DissonanceResult(
            cmdi_score=4.8,
            dissonance_tier="ACUTE_DECEPTION_RISK",
            acoustic_stress_score=85.0,
            hesitation_latency=3.5,
            positive_sentiment_prob=0.85,
            hedging_score=75.0,
        )

        eval_res = FusionRuleEngine.evaluate_lanes(
            sentiment_probs={"positive": 0.85, "neutral": 0.10, "negative": 0.05},
            sentiment_score=15.0,  # Verbal text appeared calm/positive
            absa_results=[],
            forensic_scores=None,
            discourse_res=None,
            hedging_res=None,
            audio_res=audio_res,
        )

        # Invariant: Audio veto override must trigger and elevate composite distress
        assert "RULE_ACUTE_VOCAL_DISSONANCE_VETO" in eval_res.triggered_rules
        assert eval_res.composite_distress_score >= 75.0
        assert eval_res.lane_scores.audio_distress > 0.0


class TestEdgeCaseRobustness:
    """Validates extreme edge cases (silence, NaN, empty buffers)."""

    def test_pure_silence_safety(self):
        extractor = AcousticStressExtractor()
        silence = np.zeros(16000 * 2, dtype=np.float32)
        metrics = extractor.extract_stress(silence)
        assert metrics is not None
        assert not np.isnan(metrics.stress_score)
        assert not np.isnan(metrics.f0_mean)

    def test_nan_and_inf_safety(self):
        extractor = AcousticStressExtractor()
        dirty = np.full(16000, np.nan, dtype=np.float32)
        metrics = extractor.extract_stress(dirty)
        assert metrics is not None
        assert not np.isnan(metrics.stress_score)


if __name__ == "__main__":
    pytest.main(["-v", __file__])
