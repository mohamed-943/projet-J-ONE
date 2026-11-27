"""Interruption du robot pendant qu'il parle (barge-in).

Le micro du G1 est dans la tête, près du haut-parleur : pendant une réponse, il
entend aussi la voix du robot. Le détecteur écoute donc en parallèle de la lecture,
transcrit de courtes fenêtres d'audio quand le VAD y entend de la parole, et ne
coupe la parole que si la transcription ne ressemble pas à ce que le robot dit :

- un mot d'interruption (« Sibo », « stop », « arrête », « attends », « wait »…)
  absent de la réponse en cours ;
- ou au moins `min_new_words` mots absents de la réponse, avec une confiance
  suffisante.

Les mots que le robot prononce lui-même sont de l'écho et sont ignorés. Cette
méthode marche avec ou sans annulation d'écho par le robot ; avec annulation, les
fenêtres ne contiennent plus que la voix de l'utilisateur.
"""

from __future__ import annotations

import logging
import re
import threading
import time
import unicodedata
from collections import deque
from typing import Callable

import numpy as np

log = logging.getLogger(__name__)

DEFAULT_KEYWORDS = (
    "sibo", "sibô", "cibo", "sybo", "stop", "arrête", "arrete", "attends", "attendez", "tais-toi", "silence",
    "excuse-moi", "excusez-moi", "pardon", "wait", "hold on", "excuse me", "shut up", "enough",
)
_WORD = re.compile(r"[a-z0-9]+")


def _fold(text: str) -> str:
    text = unicodedata.normalize("NFD", text.lower().replace("’", "'"))
    return "".join(c for c in text if unicodedata.category(c) != "Mn")


def words(text: str) -> list[str]:
    return _WORD.findall(_fold(text))


def is_interruption(heard: str, avg_logprob: float, robot_text: str, keywords=DEFAULT_KEYWORDS,
                    min_new_words: int = 3, min_logprob: float = -0.8) -> str | None:
    """Motif de l'interruption, ou None si `heard` n'est que l'écho de `robot_text`."""
    robot = set(words(robot_text))
    heard_folded = " ".join(words(heard))
    for kw in keywords:
        kw_words = words(kw)
        if not kw_words or all(w in robot for w in kw_words):
            continue  # le robot dit lui-même ce mot (« Je suis Sibo ») : écho possible
        if re.search(r"\b" + re.escape(" ".join(kw_words)) + r"\b", heard_folded):
            return f"mot « {kw} »"
    new = [w for w in words(heard) if w not in robot and len(w) > 2]
    if len(new) >= min_new_words and avg_logprob >= min_logprob:
        return f"{len(new)} mots nouveaux"
    return None


def echo_ratio(heard: str, robot_text: str) -> float:
    """Part des mots entendus qui figurent dans ce que le robot vient de dire."""
    heard_words = [w for w in words(heard) if len(w) > 2]
    if not heard_words:
        return 0.0
    robot = set(words(robot_text))
    return sum(w in robot for w in heard_words) / len(heard_words)


class BargeInMonitor:
    """Écoute le micro pendant une réponse, dans un fil à part."""

    def __init__(self, mic, vad, stt, keywords=DEFAULT_KEYWORDS, window_s: float = 1.4,
                 hop_s: float = 0.5, min_speech_ratio: float = 0.3, min_new_words: int = 3,
                 min_logprob: float = -0.8, sample_rate: int = 16000, frame_samples: int = 512):
        self.mic, self.vad, self.stt = mic, vad, stt
        self.keywords = keywords
        self.min_speech_ratio = min_speech_ratio
        self.min_new_words = min_new_words
        self.min_logprob = min_logprob
        self.sample_rate, self.frame_samples = sample_rate, frame_samples
        self.window_frames = max(1, round(window_s * sample_rate / frame_samples))
        self.hop_frames = max(1, round(hop_s * sample_rate / frame_samples))
        self.triggered = threading.Event()
        self.reason: str | None = None
        self.heard: str = ""
        self.checks = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._frames: deque[np.ndarray] = deque(maxlen=self.window_frames)
        self._speech: deque[bool] = deque(maxlen=self.window_frames)

    @classmethod
    def from_config(cls, cfg: dict, mic, vad, stt) -> "BargeInMonitor":
        b, a = cfg["barge_in"], cfg["audio_input"]
        return cls(mic, vad, stt, keywords=tuple(b["keywords"]), window_s=b["window_s"],
                   hop_s=b["hop_s"], min_speech_ratio=b["min_speech_ratio"],
                   min_new_words=b["min_new_words"], min_logprob=b["min_logprob"],
                   sample_rate=a["sample_rate"], frame_samples=a["frame_samples"])

    def start(self, robot_text: Callable[[], str], language: str | None) -> None:
        self.triggered.clear()
        self._stop.clear()
        self.reason, self.heard, self.checks = None, "", 0
        self._frames.clear()
        self._speech.clear()
        if hasattr(self.vad, "reset"):
            self.vad.reset()
        self._thread = threading.Thread(target=self._run, args=(robot_text, language),
                                        name="barge-in", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None

    def recent_audio(self, seconds: float) -> np.ndarray:
        """Dernières secondes entendues : le début de la phrase de l'utilisateur."""
        n = max(1, round(seconds * self.sample_rate / self.frame_samples))
        frames = list(self._frames)[-n:]
        return np.concatenate(frames) if frames else np.zeros(0, dtype=np.int16)

    def _run(self, robot_text: Callable[[], str], language: str | None) -> None:
        since_check = 0
        while not self._stop.is_set() and not self.triggered.is_set():
            frame = self.mic.read(timeout=0.1)
            if frame is None:
                continue
            self._frames.append(frame)
            self._speech.append(self.vad(frame) >= 0.5)
            since_check += 1
            if since_check < self.hop_frames or len(self._frames) < self.window_frames:
                continue
            since_check = 0
            if sum(self._speech) < self.min_speech_ratio * len(self._speech):
                continue
            self.checks += 1
            try:
                tr = self.stt.transcribe(np.concatenate(self._frames), prefer=language)
            except Exception:
                log.exception("Transcription pendant la réponse impossible")
                continue
            reason = is_interruption(tr.text, tr.avg_logprob, robot_text(), self.keywords,
                                     self.min_new_words, self.min_logprob)
            if reason:
                self.reason, self.heard = reason, tr.text
                log.info("Interruption (%s) : %r", reason, tr.text)
                self.triggered.set()
