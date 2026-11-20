"""
Virdixt Earnings Call Audio Transcriber & Turn Parser
Performs CPU-optimized speech-to-text with word/segment timestamps using faster-whisper (CTranslate2 INT8).
Extracts Q&A speaker turns, analyst-to-executive handoffs, and cognitive hesitation latency.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union
from math import gcd
import re
import numpy as np
import soundfile as sf
import scipy.signal as sps


@dataclass
class TurnSegment:
    """Represents an isolated speech turn in an earnings call."""
    turn_index: int
    start_time: float               # Start timestamp in seconds
    end_time: float                 # End timestamp in seconds
    text: str                       # Transcribed text
    hesitation_latency: float = 0.0 # Silence/pause before turn onset (seconds)
    role: str = "EXECUTIVE_ANSWER"  # OPERATOR, ANALYST_QUESTION, EXECUTIVE_ANSWER
    speaker_id: str = "SPEAKER_00"


class EarningsCallTranscriber:
    """
    Zero-VRAM CPU transcriber for earnings call audio.
    Runs CTranslate2 INT8 Whisper on Intel AVX2 with automatic VAD silence detection.
    """

    OPERATOR_CUES = re.compile(
        r"\b(next question comes from|first question from|we will now begin the q&a|operator|line is open)\b",
        re.IGNORECASE,
    )
    EXECUTIVE_RESPONSE_CUES = re.compile(
        r"^(thanks|thank you|sure|yeah|well|as you know|to answer your question|on your first question|let me take that|i'll let|so look)\b",
        re.IGNORECASE,
    )
    ANALYST_INTRO_CUES = re.compile(
        r"\b(hi|hello|thanks for taking my question|good morning|can you speak to|could you talk about)\b",
        re.IGNORECASE,
    )

    def __init__(
        self,
        model_size: str = "tiny.en",  # ponytail: default to tiny.en for low-spec CPU, user can select small.en
        device: str = "cpu",
        compute_type: str = "int8",
        cpu_threads: int = 4,
    ):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.cpu_threads = cpu_threads
        self._model = None

    def _get_model(self):
        """Lazy loader for Whisper model to save memory until actual inference."""
        if self._model is None:
            from faster_whisper import WhisperModel
            self._model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
                cpu_threads=self.cpu_threads,
            )
        return self._model

    def transcribe(
        self,
        audio_source: Union[str, np.ndarray],
        sampling_rate: int = 16000,
        beam_size: int = 3,
        word_timestamps: bool = True,
    ) -> List[TurnSegment]:
        """
        Transcribes audio source and parses Q&A dialogue turns with hesitation latencies.
        Accepts either an audio filepath (WAV/MP3/FLAC) or raw 1D/2D float32 numpy array.
        """
        import os
        # Senior Architect: Normalize audio to 16kHz float32 mono array to bypass PyAV version incompatibilities
        if isinstance(audio_source, str):
            if not os.path.exists(audio_source):
                raise FileNotFoundError(f"Audio file not found: {audio_source}")
            sig, sr = sf.read(audio_source, dtype="float32")
            if sig.ndim > 1:
                if sig.shape[0] < sig.shape[-1]:
                    sig = np.mean(sig, axis=0)
                else:
                    sig = np.mean(sig, axis=-1)
            sig = np.squeeze(sig)
            if sr != 16000:
                g = gcd(16000, sr)
                sig = sps.resample_poly(sig, 16000 // g, sr // g)
            audio_array = np.asarray(sig, dtype=np.float32)
        else:
            audio_array = np.asarray(audio_source, dtype=np.float32)
            if audio_array.ndim > 1:
                if audio_array.shape[0] < audio_array.shape[-1]:
                    audio_array = np.mean(audio_array, axis=0)
                else:
                    audio_array = np.mean(audio_array, axis=-1)
            audio_array = np.squeeze(audio_array)
            if sampling_rate != 16000 and len(audio_array) > 0:
                g = gcd(16000, sampling_rate)
                audio_array = sps.resample_poly(audio_array, 16000 // g, sampling_rate // g)
            audio_array = np.asarray(audio_array, dtype=np.float32)

        model = self._get_model()
        segments_gen, info = model.transcribe(
            audio_array,
            beam_size=beam_size,
            word_timestamps=word_timestamps,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=400),
        )

        turns: List[TurnSegment] = []
        last_end_time = 0.0
        turn_idx = 0
        current_role = "OPERATOR"

        for seg in segments_gen:
            text = seg.text.strip()
            if not text:
                continue

            # Calculate hesitation latency: silence between previous turn end and current turn start
            hesitation = max(0.0, float(seg.start - last_end_time)) if last_end_time > 0.0 else 0.0

            # Deterministic Role Tagging (Operator -> Analyst Question -> Executive Answer)
            role = self._classify_role(text, current_role)
            current_role = role

            turn = TurnSegment(
                turn_index=turn_idx,
                start_time=float(seg.start),
                end_time=float(seg.end),
                text=text,
                hesitation_latency=hesitation,
                role=role,
                speaker_id="OPERATOR" if role == "OPERATOR" else ("ANALYST" if role == "ANALYST_QUESTION" else "EXECUTIVE"),
            )
            turns.append(turn)
            last_end_time = float(seg.end)
            turn_idx += 1

        return turns

    def _classify_role(self, text: str, previous_role: str) -> str:
        """Heuristic dialog flow state machine for earnings call Q&A structure."""
        if self.OPERATOR_CUES.search(text):
            return "OPERATOR"

        if previous_role == "OPERATOR":
            # Right after operator announces the analyst, it's the analyst question
            return "ANALYST_QUESTION"

        if previous_role == "ANALYST_QUESTION":
            # If executive responds with typical thank-you / response cues
            if self.EXECUTIVE_RESPONSE_CUES.search(text):
                return "EXECUTIVE_ANSWER"
            # If the analyst is still asking questions
            if "?" in text or self.ANALYST_INTRO_CUES.search(text):
                return "ANALYST_QUESTION"
            return "EXECUTIVE_ANSWER"

        # If previous was executive answer, check if operator intervened or analyst followed up
        if "?" in text and len(text.split()) < 35:
            return "ANALYST_QUESTION"

        return "EXECUTIVE_ANSWER"
