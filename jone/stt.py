"""Transcription locale de la parole (EF-02), Whisper large-v3-turbo via
faster-whisper (quantification int8 pour limiter la mémoire GPU de la Jetson).

La langue est détectée automatiquement parmi `languages` (français et anglais par
défaut). Si Whisper détecte une autre langue — fréquent sur un énoncé très court —,
l'audio est retranscrit dans la langue autorisée la plus probable.
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
    language: str = "fr"
    language_prob: float = 1.0


class WhisperSTT:
    def __init__(self, model="large-v3-turbo", device="cuda", compute_type="int8_float16",
                 beam_size=1, language=None, languages=("fr", "en")):
        from faster_whisper import WhisperModel

        t0 = time.perf_counter()
        try:
            self.model = WhisperModel(model, device=device, compute_type=compute_type)
        except (ValueError, RuntimeError) as exc:
            # CTranslate2 compilé sans CUDA (wheel aarch64 standard) : repli CPU.
            log.warning("Whisper sur %s impossible (%s), repli sur CPU int8", device, exc)
            self.model = WhisperModel(model, device="cpu", compute_type="int8")
        self.beam_size = beam_size
        self.language = language  # None : détection automatique
        self.languages = list(languages or ([language] if language else []))
        log.info("Whisper %s chargé en %.1f s", model, time.perf_counter() - t0)

    @classmethod
    def from_config(cls, cfg: dict) -> "WhisperSTT":
        return cls(**cfg["stt"])

    def _run(self, samples: np.ndarray, language: str | None):
        segments, info = self.model.transcribe(
            samples,
            language=language,
            beam_size=self.beam_size,
            vad_filter=False,  # déjà fait par Silero en amont
            condition_on_previous_text=False,
            without_timestamps=True,
        )
        return list(segments), info

    def _best_allowed(self, info) -> str:
        probs = dict(getattr(info, "all_language_probs", None) or [])
        return max(self.languages, key=lambda lang: probs.get(lang, 0.0))

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> Transcription:
        t0 = time.perf_counter()
        samples = audio.astype(np.float32) / 32768.0 if audio.dtype == np.int16 else audio
        segments, info = self._run(samples, self.language)
        language = self.language or info.language
        language_prob = 1.0 if self.language else float(info.language_probability)
        if self.languages and language not in self.languages:
            detected = language
            language = self._best_allowed(info)
            log.info("Langue détectée %s non prise en charge, retranscription en %s", detected, language)
            segments, _ = self._run(samples, language)

        elapsed = (time.perf_counter() - t0) * 1000
        text = " ".join(s.text.strip() for s in segments).strip()
        if not segments:
            return Transcription("", no_speech_prob=1.0, audio_s=len(samples) / sample_rate,
                                 elapsed_ms=elapsed, language=language, language_prob=language_prob)
        return Transcription(
            text=text,
            avg_logprob=float(np.mean([s.avg_logprob for s in segments])),
            no_speech_prob=float(max(s.no_speech_prob for s in segments)),
            compression_ratio=float(max(s.compression_ratio for s in segments)),
            audio_s=len(samples) / sample_rate,
            elapsed_ms=elapsed,
            language=language,
            language_prob=language_prob,
        )
