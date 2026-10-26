"""Détection du début et de la fin d'une prise de parole (EF-01).

`SileroVAD` donne une probabilité de parole par trame de 32 ms. `SpeechSegmenter`
transforme ce flux en énoncés complets :
- déclenchement quand la probabilité dépasse `threshold` ;
- fin après `min_silence_ms` sous `neg_threshold` (hystérésis, évite de couper
  l'utilisateur à chaque micro-pause) ;
- les énoncés plus courts que `min_speech_ms` sont jetés : claquement de porte,
  pas, choc sur la table ;
- `pre_roll_ms` d'audio avant le déclenchement est conservé pour ne pas perdre la
  première syllabe.
"""

from __future__ import annotations

from collections import deque
from typing import Callable

import numpy as np


class SileroVAD:
    def __init__(self, sample_rate: int = 16000):
        import torch
        from silero_vad import load_silero_vad

        torch.set_num_threads(1)
        self._torch = torch
        self.model = load_silero_vad()
        self.sample_rate = sample_rate

    def __call__(self, frame: np.ndarray) -> float:
        audio = self._torch.from_numpy(frame.astype(np.float32) / 32768.0)
        return float(self.model(audio, self.sample_rate).item())

    def reset(self) -> None:
        self.model.reset_states()


class SpeechSegmenter:
    def __init__(
        self,
        vad: Callable[[np.ndarray], float],
        sample_rate: int = 16000,
        frame_samples: int = 512,
        threshold: float = 0.5,
        neg_threshold: float = 0.35,
        min_speech_ms: int = 250,
        min_silence_ms: int = 600,
        pre_roll_ms: int = 300,
        max_utterance_s: float = 15.0,
    ):
        self.vad = vad
        frame_ms = 1000.0 * frame_samples / sample_rate
        self.threshold = threshold
        self.neg_threshold = neg_threshold
        self.min_speech_frames = max(1, round(min_speech_ms / frame_ms))
        self.min_silence_frames = max(1, round(min_silence_ms / frame_ms))
        self.max_frames = max(1, round(max_utterance_s * 1000 / frame_ms))
        self._pre_roll: deque[np.ndarray] = deque(maxlen=max(0, round(pre_roll_ms / frame_ms)))
        self.reset()

    @classmethod
    def from_config(cls, vad, cfg: dict) -> "SpeechSegmenter":
        a, v = cfg["audio_input"], cfg["vad"]
        return cls(vad, sample_rate=a["sample_rate"], frame_samples=a["frame_samples"], **v)

    @property
    def in_speech(self) -> bool:
        return self._triggered

    def reset(self) -> None:
        self._triggered = False
        self._frames: list[np.ndarray] = []
        self._speech_frames = 0
        self._silence_run = 0
        self._pre_roll.clear()
        if hasattr(self.vad, "reset"):
            self.vad.reset()

    def process(self, frame: np.ndarray) -> np.ndarray | None:
        """Ajoute une trame ; renvoie l'énoncé complet (int16) quand il se termine."""
        prob = self.vad(frame)

        if not self._triggered:
            if prob >= self.threshold:
                self._triggered = True
                self._frames = list(self._pre_roll)
                self._frames.append(frame)
                self._speech_frames = 1
                self._silence_run = 0
            else:
                self._pre_roll.append(frame)
            return None

        self._frames.append(frame)
        if prob >= self.neg_threshold:
            self._silence_run = 0
            if prob >= self.threshold:
                self._speech_frames += 1
        else:
            self._silence_run += 1

        ended = self._silence_run >= self.min_silence_frames
        too_long = len(self._frames) >= self.max_frames
        if not (ended or too_long):
            return None

        keep = self._speech_frames >= self.min_speech_frames
        frames = self._frames[: len(self._frames) - self._silence_run] if ended else self._frames
        self.reset()
        if not keep or not frames:
            return None
        return np.concatenate(frames)
