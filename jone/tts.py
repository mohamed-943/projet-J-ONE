"""Synthèse vocale avec Piper (EF-08) : voix fr_FR-siwis pour le français,
en_US-lessac pour l'anglais. `MultiVoiceTTS` choisit la voix selon la langue
détectée chez l'interlocuteur.

Sortie toujours convertie en PCM 16 kHz mono 16 bits (EF-09) : les voix « medium »
sont en 22,05 kHz, d'où le rééchantillonnage.
Compatible avec l'API de piper-tts >= 1.3 (`synthesize` → AudioChunk) et avec
l'ancienne (`synthesize_stream_raw`).
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np

from .audio_utils import to_pcm16k_mono

log = logging.getLogger(__name__)


class PiperTTS:
    def __init__(self, voice_model: str | Path, length_scale: float = 1.0, use_cuda: bool = False):
        from piper import PiperVoice

        path = Path(voice_model)
        if not path.exists():
            raise FileNotFoundError(
                f"Voix Piper introuvable : {path}. Lancez `python tools/download_models.py`."
            )
        t0 = time.perf_counter()
        self.voice = PiperVoice.load(str(path), use_cuda=use_cuda)
        self.sample_rate = int(self.voice.config.sample_rate)
        self.length_scale = length_scale
        log.info("Voix Piper %s chargée en %.1f s (%d Hz)", path.name,
                 time.perf_counter() - t0, self.sample_rate)

    @classmethod
    def from_config(cls, cfg: dict) -> "PiperTTS":
        from .config import resolve_path

        t = cfg["tts"]
        return cls(resolve_path(t["voice_model"]), t["length_scale"], t["use_cuda"])

    def _raw_int16(self, text: str) -> bytes:
        if hasattr(self.voice, "synthesize_stream_raw"):  # piper-tts < 1.3
            return b"".join(self.voice.synthesize_stream_raw(text, length_scale=self.length_scale))
        from piper import SynthesisConfig

        syn = SynthesisConfig(length_scale=self.length_scale)
        return b"".join(chunk.audio_int16_bytes for chunk in self.voice.synthesize(text, syn_config=syn))

    def synthesize(self, text: str) -> bytes:
        """Texte → octets PCM 16 kHz mono 16 bits."""
        raw = self._raw_int16(text)
        return to_pcm16k_mono(np.frombuffer(raw, dtype=np.int16), self.sample_rate)


class MultiVoiceTTS:
    """Une voix Piper par langue ; repli sur la voix par défaut si la langue n'a
    pas de voix installée."""

    def __init__(self, voices: dict[str, PiperTTS], default: str = "fr"):
        if default not in voices:
            raise ValueError(f"aucune voix pour la langue par défaut « {default} »")
        self.voices = voices
        self.default = default

    @classmethod
    def from_config(cls, cfg: dict) -> "MultiVoiceTTS":
        from .config import resolve_path

        t = cfg["tts"]
        default = t.get("default_language", "fr")
        voices = {default: PiperTTS(resolve_path(t["voice_model"]), t["length_scale"], t["use_cuda"])}
        for language, path in (t.get("voices") or {}).items():
            if language == default:
                continue
            try:
                voices[language] = PiperTTS(resolve_path(path), t["length_scale"], t["use_cuda"])
            except FileNotFoundError as exc:
                log.warning("%s Les réponses en « %s » utiliseront la voix « %s ».", exc, language, default)
        return cls(voices, default)

    def synthesize(self, text: str, language: str | None = None) -> bytes:
        return self.voices.get(language or self.default, self.voices[self.default]).synthesize(text)
