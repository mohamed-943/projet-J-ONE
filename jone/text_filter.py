"""Filtrage des transcriptions vides, parasites ou douteuses (EF-03).

Whisper « hallucine » des phrases typiques des sous-titres sur du bruit ou du
silence ; on les rejette avant tout appel au LLM, comme les transcriptions à
faible confiance ou anormalement répétitives.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from .stt import Transcription

# Hallucinations de Whisper en français (comparées après normalisation).
# Toujours rejetées : personne ne dit ça à un robot.
DEFAULT_BLACKLIST = [
    "sous-titres réalisés par la communauté d'amara.org",
    "sous-titres réalisés para la communauté d'amara.org",
    "sous-titrage st' 501",
    "sous-titrage société radio-canada",
    "sous-titrage fr",
    "merci d'avoir regardé cette vidéo",
    "merci d'avoir regardé",
    "abonnez-vous",
    "n'oubliez pas de vous abonner",
    "musique",
    "applaudissements",
    "rires",
]
# Phrases plausibles chez un vrai utilisateur, mais aussi hallucinations fréquentes
# sur du bruit : rejetées seulement si la confiance est faible.
SUSPICIOUS_SHORT = ["merci", "merci à tous", "merci beaucoup", "au revoir", "bonne journée"]
SUSPICIOUS_MAX_NO_SPEECH = 0.3
SUSPICIOUS_MIN_LOGPROB = -0.6


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text).lower().strip()
    text = text.replace("’", "'")
    text = re.sub(r"[\[\]()*♪♫]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip(" .!?,;:-")


_SUSPICIOUS = {normalize(s) for s in SUSPICIOUS_SHORT}


@dataclass
class FilterResult:
    accepted: bool
    text: str
    reason: str = ""


class TranscriptFilter:
    def __init__(self, min_chars=2, max_no_speech_prob=0.6, min_avg_logprob=-1.0,
                 max_compression_ratio=2.4, blacklist=None):
        self.min_chars = min_chars
        self.max_no_speech_prob = max_no_speech_prob
        self.min_avg_logprob = min_avg_logprob
        self.max_compression_ratio = max_compression_ratio
        self.blacklist = {normalize(b) for b in DEFAULT_BLACKLIST + list(blacklist or [])}

    @classmethod
    def from_config(cls, cfg: dict) -> "TranscriptFilter":
        return cls(**cfg["filter"])

    def check(self, t: Transcription) -> FilterResult:
        text = re.sub(r"\s+", " ", t.text).strip()
        norm = normalize(text)
        letters = sum(c.isalnum() for c in norm)

        if letters < self.min_chars:
            return FilterResult(False, text, "vide")
        if norm in self.blacklist:
            return FilterResult(False, text, "hallucination connue")
        if norm in _SUSPICIOUS and (t.no_speech_prob > SUSPICIOUS_MAX_NO_SPEECH
                                    or t.avg_logprob < SUSPICIOUS_MIN_LOGPROB):
            return FilterResult(False, text, "hallucination probable")
        if t.no_speech_prob > self.max_no_speech_prob and t.avg_logprob < self.min_avg_logprob:
            return FilterResult(False, text, "probablement pas de parole")
        if t.compression_ratio > self.max_compression_ratio:
            return FilterResult(False, text, "texte répétitif")
        if _has_repeated_ngram(norm):
            return FilterResult(False, text, "boucle de répétition")
        return FilterResult(True, text)


def _has_repeated_ngram(text: str, n: int = 3, max_repeats: int = 4) -> bool:
    words = text.split()
    if len(words) < n * max_repeats:
        return False
    counts: dict[tuple, int] = {}
    for i in range(len(words) - n + 1):
        gram = tuple(words[i : i + n])
        counts[gram] = counts.get(gram, 0) + 1
        if counts[gram] >= max_repeats:
            return True
    return False


class WakeWordGate:
    """Mot d'éveil optionnel (question ouverte du cahier des charges).

    Désactivé : écoute continue. Activé : un énoncé n'est traité que s'il contient
    « J-ONE » (ou une variante transcrite), sauf pendant `session_timeout_s` après
    le dernier échange, pour pouvoir enchaîner sans répéter le mot.
    """

    def __init__(self, enabled=False, words=(), session_timeout_s=30.0):
        self.enabled = enabled
        self.session_timeout_s = session_timeout_s
        variants = sorted({normalize(w) for w in words}, key=len, reverse=True)
        self._pattern = (
            re.compile(r"\b(" + "|".join(re.escape(v) for v in variants) + r")\b[\s,.!?]*")
            if variants else None
        )
        self._last_exchange = float("-inf")

    @classmethod
    def from_config(cls, cfg: dict) -> "WakeWordGate":
        return cls(**cfg["wake_word"])

    def admit(self, text: str, now: float) -> str | None:
        """Renvoie le texte à envoyer au LLM, ou None si l'énoncé est ignoré."""
        if not self.enabled:
            return text
        match = self._pattern.search(normalize(text)) if self._pattern else None
        if match:
            stripped = re.sub(self._pattern.pattern, "", text, count=1, flags=re.IGNORECASE)
            return stripped.lstrip(" ,.!?").strip() or "Bonjour"
        if now - self._last_exchange <= self.session_timeout_s:
            return text
        return None

    def mark_exchange(self, now: float) -> None:
        self._last_exchange = now
