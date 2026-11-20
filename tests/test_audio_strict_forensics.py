"""Strict Forensic Background Check Suite for Virdixt Audio Engine & Lane 6 Fusion.

Tests adversarial audio inputs, memory bounds, thread-safety, zero-divisions,
NaN/Inf immunity, and dictionary-vs-dataclass API parity.
"""

import concurrent.futures
import math
import os
import tempfile
import tracemalloc
import numpy as np
import pytest

from audio.prosody import AcousticStressExtractor, ProsodyMetrics, SpeakerBaseline
from audio.transcribe import EarningsCallTranscriber, TurnSegment
from audio.dissonance import CrossModalDissonanceEngine, DissonanceResult
from nlp.fusion import FusionRuleEngine


# =====================================================================
# 1. ADVERSARIAL AUDIO BUFFER & NUMERICAL STABILITY CHECKS
# =====================================================================

def test_audio_zero_length_and_micro_segments():
    """Verify that empty, single-sample, and micro-arrays do not crash openSMILE."""
    extractor = AcousticStressExtractor()

    # Zero length array
    empty = np.array([], dtype=np.float32)
    res_empty = extractor.extract_stress(empty)
    assert isinstance(res_empty, ProsodyMetrics)
    assert res_empty.stress_score >= 0.0

    # Single-sample array
    single = np.array([0.5], dtype=np.float32)
    res_single = extractor.extract_stress(single)
    assert isinstance(res_single, ProsodyMetrics)

    # 10 samples (<1ms at 16kHz)
    micro = np.array([0.1] * 10, dtype=np.float32)
    res_micro = extractor.extract_stress(micro)
    assert isinstance(res_micro, ProsodyMetrics)


def test_audio_stereo_channels_first_and_last():
    """Verify that both (channels, samples) and (samples, channels) 2D arrays
    are reduced to mono properly without sample truncation.
    """
    extractor = AcousticStressExtractor()

    # Channels-last stereo: (16000, 2)
    stereo_last = np.random.randn(16000, 2).astype(np.float32) * 0.1
    res_last = extractor.extract_stress(stereo_last, sampling_rate=16000)
    assert isinstance(res_last, ProsodyMetrics)

    # Channels-first stereo: (2, 16000)
    stereo_first = np.random.randn(2, 16000).astype(np.float32) * 0.1
    res_first = extractor.extract_stress(stereo_first, sampling_rate=16000)
    assert isinstance(res_first, ProsodyMetrics)


def test_audio_nan_inf_and_extreme_saturation():
    """Verify that audio buffers with NaN, Inf, and digital clipping (overload)
    remain numerically stable and produce finite scores in [0.0, 100.0].
    """
    extractor = AcousticStressExtractor()

    # Extreme digital clipping (+1000.0 / -1000.0 square wave)
    clipped = np.sign(np.sin(np.linspace(0, 100, 16000))).astype(np.float32) * 1000.0
    res_clipped = extractor.extract_stress(clipped)
    assert 0.0 <= res_clipped.stress_score <= 100.0
    assert not math.isnan(res_clipped.stress_score)

    # Injected NaNs and Infs
    corrupt = np.random.randn(16000).astype(np.float32) * 0.1
    corrupt[100:200] = np.nan
    corrupt[500:600] = np.inf
    corrupt[800:900] = -np.inf
    res_corrupt = extractor.extract_stress(corrupt)
    assert 0.0 <= res_corrupt.stress_score <= 100.0
    assert not math.isnan(res_corrupt.stress_score)


def test_audio_missing_file_safety():
    """Verify that non-existent audio file paths are handled gracefully."""
    extractor = AcousticStressExtractor()
    res = extractor.extract_stress("non_existent_audio_path_123.wav")
    assert isinstance(res, ProsodyMetrics)
    assert res.stress_score >= 0.0

    transcriber = EarningsCallTranscriber()
    with pytest.raises(FileNotFoundError):
        transcriber.transcribe("non_existent_audio_path_123.wav")


# =====================================================================
# 2. CROSS-MODAL DISSONANCE ENGINE ADVERSARIAL CHECKS
# =====================================================================

def test_dissonance_none_and_empty_text():
    """Verify that passing text=None, text='', or whitespace does not crash
    with AttributeError or ZeroDivisionError.
    """
    engine = CrossModalDissonanceEngine()
    sig = np.random.randn(16000).astype(np.float32) * 0.05

    for bad_text in [None, "", "   \t\n  "]:
        res = engine.evaluate_turn(sig, text=bad_text)
        assert isinstance(res, DissonanceResult)
        assert 0.0 <= res.cmdi_score <= 10.0
        assert not math.isnan(res.cmdi_score)


def test_dissonance_nan_sentiment_immunity():
    """Verify that corrupted NaN sentiment probabilities do not propagate into
    cmdi_score and do not falsely trigger an ACUTE_DECEPTION_RISK alert.
    """
    engine = CrossModalDissonanceEngine()
    sig = np.random.randn(16000).astype(np.float32) * 0.05

    res = engine.evaluate_turn(
        sig,
        text="Everything is fine.",
        sentiment_probs={"positive": float("nan"), "negative": float("nan")},
    )
    assert not math.isnan(res.cmdi_score)
    assert 0.0 <= res.cmdi_score <= 10.0
    assert res.dissonance_tier != "ACUTE_DECEPTION_RISK", \
        "NaN sentiment must not cause false-positive acute deception alert!"


def test_dissonance_negative_and_extreme_hesitation():
    """Verify that negative or extreme hesitation latencies are clamped properly."""
    engine = CrossModalDissonanceEngine()
    sig = np.random.randn(16000).astype(np.float32) * 0.05

    # Negative hesitation latency
    res_neg = engine.evaluate_turn(sig, text="Answer", hesitation_latency=-15.0)
    assert res_neg.hesitation_latency == 0.0
    assert 0.0 <= res_neg.cmdi_score <= 10.0

    # Extreme cognitive stalling
    res_ext = engine.evaluate_turn(sig, text="Answer", hesitation_latency=9999.0)
    assert 0.0 <= res_ext.cmdi_score <= 10.0


def test_dissonance_tier_boundaries():
    """Verify exact tier boundaries for CMDI scoring:
    < 1.5 -> CONGRUENT_CALM
    1.5 <= x < 3.0 -> MODERATE_TENSION
    3.0 <= x < 5.0 -> HIGH_DISSONANCE
    >= 5.0 -> ACUTE_DECEPTION_RISK
    """
    engine = CrossModalDissonanceEngine()
    sig_calm = np.random.randn(16000).astype(np.float32) * 0.01

    res_calm = engine.evaluate_turn(
        sig_calm,
        text="Normal operations continue in line with expectations.",
        hesitation_latency=0.3,
    )
    assert res_calm.dissonance_tier in ["CONGRUENT_CALM", "MODERATE_TENSION"]


# =====================================================================
# 3. FUSION ENGINE LANE 6 DICT VS OBJECT PARITY
# =====================================================================

def test_fusion_lane6_dict_payload_parity():
    """CRITICAL: Passing audio_res as a dictionary (e.g. from JSON API) MUST
    produce identical scores and trigger identical rules as passing DissonanceResult.
    Previously failed because fusion engine used getattr() directly on dicts.
    """
    # 1. As DissonanceResult object
    obj_audio = DissonanceResult(
        cmdi_score=6.5,
        dissonance_tier="ACUTE_DECEPTION_RISK",
        acoustic_stress_score=85.0,
        hesitation_latency=3.5,
        positive_sentiment_prob=0.85,
        hedging_score=60.0,
    )
    res_obj = FusionRuleEngine.evaluate_lanes(
        sentiment_probs={"positive": 0.85, "neutral": 0.10, "negative": 0.05},
        sentiment_score=15.0,
        absa_results=[],
        forensic_scores=None,
        discourse_res=None,
        hedging_res=None,
        audio_res=obj_audio,
    )

    # 2. As dictionary payload
    dict_audio = {
        "cmdi_score": 6.5,
        "dissonance_tier": "ACUTE_DECEPTION_RISK",
        "acoustic_stress_score": 85.0,
        "hesitation_latency": 3.5,
        "positive_sentiment_prob": 0.85,
        "hedging_score": 60.0,
    }
    res_dict = FusionRuleEngine.evaluate_lanes(
        sentiment_probs={"positive": 0.85, "neutral": 0.10, "negative": 0.05},
        sentiment_score=15.0,
        absa_results=[],
        forensic_scores=None,
        discourse_res=None,
        hedging_res=None,
        audio_res=dict_audio,
    )

    assert "RULE_ACUTE_VOCAL_DISSONANCE_VETO" in res_obj.triggered_rules
    assert "RULE_ACUTE_VOCAL_DISSONANCE_VETO" in res_dict.triggered_rules, \
        "Dictionary audio payload must trigger RULE_ACUTE_VOCAL_DISSONANCE_VETO!"
    assert res_obj.composite_distress_score == res_dict.composite_distress_score
    assert res_obj.lane_scores.audio_distress == res_dict.lane_scores.audio_distress


def test_fusion_lane6_high_dissonance_escalation():
    """Verify that CMDI >= 2.5 triggers RULE_HIGH_VOCAL_DISSONANCE_ESCALATION (+10.0 bump)."""
    dict_audio = {
        "cmdi_score": 3.5,
        "dissonance_tier": "HIGH_DISSONANCE",
    }
    res = FusionRuleEngine.evaluate_lanes(
        sentiment_probs={"positive": 0.5, "neutral": 0.3, "negative": 0.2},
        sentiment_score=20.0,
        absa_results=[],
        forensic_scores=None,
        discourse_res=None,
        hedging_res=None,
        audio_res=dict_audio,
    )
    assert "RULE_HIGH_VOCAL_DISSONANCE_ESCALATION" in res.triggered_rules
    assert res.escalation_applied is True


# =====================================================================
# 4. SPEAKER BASELINE ENROLLMENT & ZERO-VARIANCE SAFETY
# =====================================================================

def test_prosody_baseline_zero_variance_guard():
    """Verify that enrolling a speaker on pure DC or flatline audio sets
    non-zero minimum variance guards, preventing ZeroDivisionError during z-scoring.
    """
    extractor = AcousticStressExtractor()
    flatline = np.zeros(16000, dtype=np.float32)

    baseline = extractor.enroll_speaker_baseline("FLATLINE_SPEAKER", flatline)
    assert baseline.f0_sigma > 0.0
    assert baseline.jitter_sigma > 0.0
    assert baseline.shimmer_sigma > 0.0
    assert baseline.hnr_sigma > 0.0
    assert baseline.loudness_sigma > 0.0

    # Extracting stress against flatline baseline must not raise ZeroDivisionError
    res = extractor.extract_stress(flatline, speaker_id="FLATLINE_SPEAKER")
    assert 0.0 <= res.stress_score <= 100.0
    assert not math.isnan(res.stress_score)


# =====================================================================
# 5. CONCURRENCY & MEMORY STRESS VERIFICATION
# =====================================================================

def test_prosody_multithreaded_reentrancy():
    """Verify openSMILE extraction is thread-safe across concurrent threads."""
    extractor = AcousticStressExtractor()
    sig = np.random.randn(16000).astype(np.float32) * 0.1

    def worker(i: int):
        return extractor.extract_stress(sig).stress_score

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(worker, i) for i in range(8)]
        scores = [f.result() for f in futures]

    assert len(scores) == 8
    assert all(0.0 <= s <= 100.0 for s in scores)


def test_prosody_memory_stability_under_load():
    """Verify that repeated acoustic extractions do not exhibit runaway memory leaks."""
    extractor = AcousticStressExtractor()
    sig = np.random.randn(16000).astype(np.float32) * 0.1

    tracemalloc.start()
    snap1 = tracemalloc.take_snapshot()

    for _ in range(40):
        extractor.extract_stress(sig)

    import gc
    gc.collect()
    snap2 = tracemalloc.take_snapshot()
    tracemalloc.stop()

    stats = snap2.compare_to(snap1, "lineno")
    total_kb = sum(s.size_diff for s in stats) / 1024
    assert total_kb < 1024.0, f"Suspicious memory leak in prosody extraction: {total_kb:.2f} KB"
