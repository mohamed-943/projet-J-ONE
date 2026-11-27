"""Transcription locale de la parole (EF-02), Whisper large-v3-turbo via
faster-whisper (quantification int8 pour limiter la mémoire GPU de la Jetson).

La langue est choisie parmi `languages` (français et anglais par défaut). La
détection d'un modèle Whisper coûte une passe complète de son encodeur : sur la
Jetson sans GPU, 2,5 s pour Whisper small. Elle est donc confiée à un petit modèle
(`detect_model`, « tiny » par défaut, environ dix fois plus rapide), puis le grand
modèle transcrit une seule fois dans la langue retenue. Quand la détection hésite
entre les langues autorisées, on retient `default_language`.
"""

from __future__ import annotations

import logging
import os
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


def _load_model(model, device, compute_type, cpu_threads):
    from faster_whisper import WhisperModel

    try:
        return WhisperModel(model, device=device, compute_type=compute_type, cpu_threads=cpu_threads)
    except (ValueError, RuntimeError) as exc:
        # CTranslate2 compilé sans CUDA (wheel aarch64 standard) : repli CPU.
        log.warning("Whisper %s sur %s impossible (%s), repli sur CPU int8", model, device, exc)
        return WhisperModel(model, device="cpu", compute_type="int8", cpu_threads=cpu_threads)


class WhisperSTT:
    def __init__(self, model="large-v3-turbo", device="cuda", compute_type="int8_float16",
                 beam_size=1, language=None, languages=("fr", "en"), default_language="fr",
                 language_margin=0.65, detect_model="tiny", cpu_threads=0):
        t0 = time.perf_counter()
        # 0 : tous les cœurs (CTranslate2 n'en prend que 4 par défaut).
        cpu_threads = cpu_threads or os.cpu_count() or 4
        self.model = _load_model(model, device, compute_type, cpu_threads)
        self.beam_size = beam_size
        self.language = language  # langue imposée : pas de détection
        self.languages = list(languages or ([language] if language else []))
        self.default_language = default_language
        self.language_margin = language_margin
        self.detector = None
        if not language and len(self.languages) > 1 and detect_model and detect_model != model:
            self.detector = _load_model(detect_model, device, compute_type, cpu_threads)
        log.info("Whisper %s chargé en %.1f s (%d cœurs CPU%s)", model, time.perf_counter() - t0,
                 cpu_threads, f", langue détectée par {detect_model}" if self.detector else "")

    @classmethod
    def from_config(cls, cfg: dict) -> "WhisperSTT":
        return cls(**cfg["stt"])

    def _run(self, samples: np.ndarray, language: str | None):
        segments, info = self.model.transcribe(
            samples,
            language=language,
            beam_size=self.beam_size,
            # Pas de nouveaux décodages à température croissante quand le texte est
            # douteux : chacun coûte une passe complète (13 à 29 s mesurées sur la
            # Jetson), et le filtre de transcription écarte de toute façon ces cas.
            temperature=0.0,
            vad_filter=False,  # déjà fait par Silero en amont
            condition_on_previous_text=False,
            without_timestamps=True,
        )
        return list(segments), info

    def _choose(self, info) -> tuple[str, float]:
        """Langue autorisée la plus probable ; `default_language` si la détection
        hésite (part de la meilleure langue sous `language_margin`)."""
        probs = dict(getattr(info, "all_language_probs", None) or [])
        if not probs and getattr(info, "language", None):
            probs = {info.language: float(info.language_probability)}
        allowed = {lang: probs.get(lang, 0.0) for lang in self.languages}
        best = max(allowed, key=allowed.get)
        total = sum(allowed.values())
        share = allowed[best] / total if total else 0.0
        if share < self.language_margin and self.default_language in allowed:
            return self.default_language, share
        return best, share

    def detect_language(self, samples: np.ndarray) -> tuple[str, float]:
        # Le générateur de segments n'est pas parcouru : seule la détection est calculée.
        _, info = self.detector.transcribe(samples, language=None, beam_size=1,
                                           vad_filter=False, without_timestamps=True)
        return self._choose(info)

    def warmup(self) -> None:
        """Première inférence à vide : sinon le premier énoncé paie l'initialisation."""
        silence = np.zeros(16000, dtype=np.float32)
        t0 = time.perf_counter()
        if self.detector:
            self.detect_language(silence)
        self._run(silence, self.language or self.default_language)
        log.info("Whisper préchauffé en %.1f s", time.perf_counter() - t0)

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> Transcription:
        t0 = time.perf_counter()
        samples = audio.astype(np.float32) / 32768.0 if audio.dtype == np.int16 else audio
        if self.language:
            language, language_prob = self.language, 1.0
            segments, _ = self._run(samples, language)
        elif self.detector:
            language, language_prob = self.detect_language(samples)
            segments, _ = self._run(samples, language)
        else:
            segments, info = self._run(samples, None)
            detected = info.language
            language, language_prob = self._choose(info)
            if language != detected:
                log.info("Langue détectée %s, retranscription en %s", detected, language)
                segments, _ = self._run(samples, language)

        elapsed = (time.perf_counter() - t0) * 1000
        log.info("Transcription en %.0f ms (langue %s, %.0f %%)", elapsed, language, language_prob * 100)
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
