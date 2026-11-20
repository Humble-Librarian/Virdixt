"""
Virdixt Multimodal Acoustic Forensic Engine (Path #5)
Analyzes executive vocal stress, acoustic prosody, unscripted hesitation latency,
and cross-modal linguistic dissonance on corporate earnings calls.
Optimized for zero-VRAM CPU execution (AVX2 INT8 / openSMILE C++).
"""

from .prosody import AcousticStressExtractor, ProsodyMetrics
from .transcribe import EarningsCallTranscriber, TurnSegment
from .dissonance import CrossModalDissonanceEngine, DissonanceResult

__all__ = [
    "AcousticStressExtractor",
    "ProsodyMetrics",
    "EarningsCallTranscriber",
    "TurnSegment",
    "CrossModalDissonanceEngine",
    "DissonanceResult",
]
