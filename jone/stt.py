"""Transcription locale de la parole française (EF-02), Whisper large-v3-turbo via
faster-whisper (quantification int8 pour limiter la mémoire GPU de la Jetson).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class Transcription:
    text: str
    avg_logprob: float = 0.0
    no_speech_prob: float = 0.0
    compression_ratio: float = 1.0
    audio_s: float = 0.0
    elapsed_ms: float = 0.0


class WhisperSTT:
    def __init__(self, model="large-v3-turbo", device="cuda", compute_type="int8_float16",
                 beam_size=1, language="fr"):
        from faster_whisper import WhisperModel

        t0 = time.perf_counter()
        try:
            self.model = WhisperModel(model, device=device, compute_type=compute_type)
        except (ValueError, RuntimeError) as exc:
            # CTranslate2 compilé sans CUDA (wheel aarch64 standard) : repli CPU.
            log.warning("Whisper sur %s impossible (%s), repli sur CPU int8", device, exc)
            self.model = WhisperModel(model, device="cpu", compute_type="int8")
        self.beam_size = beam_size
        self.language = language
        log.info("Whisper %s chargé en %.1f s", model, time.perf_counter() - t0)

    @classmethod
    def from_config(cls, cfg: dict) -> "WhisperSTT":
        return cls(**cfg["stt"])

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> Transcription:
        t0 = time.perf_counter()
        samples = audio.astype(np.float32) / 32768.0 if audio.dtype == np.int16 else audio
        segments, _info = self.model.transcribe(
            samples,
            language=self.language,
            beam_size=self.beam_size,
            vad_filter=False,  # déjà fait par Silero en amont
            condition_on_previous_text=False,
            without_timestamps=True,
        )
        segments = list(segments)
        elapsed = (time.perf_counter() - t0) * 1000
        text = " ".join(s.text.strip() for s in segments).strip()
        if not segments:
            return Transcription("", no_speech_prob=1.0, audio_s=len(samples) / sample_rate,
                                 elapsed_ms=elapsed)
        return Transcription(
            text=text,
            avg_logprob=float(np.mean([s.avg_logprob for s in segments])),
            no_speech_prob=float(max(s.no_speech_prob for s in segments)),
            compression_ratio=float(max(s.compression_ratio for s in segments)),
            audio_s=len(samples) / sample_rate,
            elapsed_ms=elapsed,
        )
