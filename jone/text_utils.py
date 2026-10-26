"""Préparation du texte du LLM pour la synthèse vocale.

- `clean_for_speech` retire ce qui ne se prononce pas : markdown, puces, emoji,
  symboles, URL (EF-06) ;
- `SentenceSplitter` découpe le flux de jetons en phrases, pour lancer la synthèse
  dès la première phrase (réduction de la latence perçue, ENF-01) ;
- `ThinkStripper` retire les blocs <think>…</think> que Qwen3 peut émettre.
"""

from __future__ import annotations

import re
import unicodedata

_MARKDOWN = re.compile(r"(\*\*|__|\*|`+|~~|^#{1,6}\s*|^>\s*)", re.MULTILINE)
_BULLET = re.compile(r"^\s*(?:[-•*+]|\d+[.)])\s+", re.MULTILINE)
_URL = re.compile(r"https?://\S+|www\.\S+")
_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_SPACES = re.compile(r"\s+")

_SYMBOL_WORDS = {
    "%": " pour cent",
    "&": " et ",
    "+": " plus ",
    "=": " égale ",
    "€": " euros",
    "$": " dollars",
    "°C": " degrés",
    "°": " degrés",
    "/": " sur ",
}


def _is_emoji_or_symbol(ch: str) -> bool:
    cat = unicodedata.category(ch)
    # So : symboles divers (emoji, pictogrammes) ; Cs/Co : surrogates et usage privé.
    return cat in ("So", "Cs", "Co") or ch in "‍️"


def clean_for_speech(text: str) -> str:
    text = _LINK.sub(r"\1", text)
    text = _URL.sub("", text)
    text = _BULLET.sub("", text)
    text = _MARKDOWN.sub("", text)
    for sym, word in _SYMBOL_WORDS.items():  # avant le filtre : « ° » est un symbole
        text = text.replace(sym, word)
    text = "".join(ch for ch in text if not _is_emoji_or_symbol(ch))
    text = re.sub(r"[#*_<>|\\^{}\[\]]", " ", text)
    text = text.replace("\n", " ")
    text = _SPACES.sub(" ", text)
    text = re.sub(r"\s+([,.])", r"\1", text)
    return text.strip()


# Fin de phrase : ponctuation forte suivie d'un espace. On évite les abréviations
# courantes (M., Mme., Dr., etc.) et les nombres décimaux (3.5 n'a pas d'espace).
_SENTENCE_END = re.compile(r"(?<!\bM)(?<!\bDr)(?<!\bMme)(?<!\betc)([.!?…]+)[\"»)]?\s+")


class SentenceSplitter:
    """Accumule des fragments de texte et renvoie les phrases complètes."""

    def __init__(self, soft_limit: int = 140):
        self.soft_limit = soft_limit
        self._buf = ""

    def feed(self, delta: str) -> list[str]:
        self._buf += delta
        out: list[str] = []
        while True:
            m = _SENTENCE_END.search(self._buf)
            if m:
                out.append(self._buf[: m.end()].strip())
                self._buf = self._buf[m.end() :]
                continue
            if len(self._buf) > self.soft_limit:
                # Phrase très longue : on coupe à la dernière virgule pour ne pas
                # retarder la première sortie audio.
                cut = self._buf.rfind(", ", 0, self.soft_limit)
                if cut > 20:
                    out.append(self._buf[: cut + 1].strip())
                    self._buf = self._buf[cut + 2 :]
                    continue
            break
        return [s for s in out if s]

    def flush(self) -> list[str]:
        rest, self._buf = self._buf.strip(), ""
        return [rest] if rest else []


class ThinkStripper:
    """Filtre en flux les sections <think>…</think>, même coupées entre deux fragments."""

    OPEN, CLOSE = "<think>", "</think>"

    def __init__(self):
        self._buf = ""
        self._inside = False

    def feed(self, delta: str) -> str:
        self._buf += delta
        out = []
        while True:
            tag = self.CLOSE if self._inside else self.OPEN
            idx = self._buf.find(tag)
            if idx >= 0:
                if not self._inside:
                    out.append(self._buf[:idx])
                self._buf = self._buf[idx + len(tag) :]
                self._inside = not self._inside
                continue
            # Garder une éventuelle balise partielle en fin de tampon.
            keep = 0
            for k in range(min(len(tag) - 1, len(self._buf)), 0, -1):
                if tag.startswith(self._buf[-k:]):
                    keep = k
                    break
            emit = self._buf[: len(self._buf) - keep]
            self._buf = self._buf[len(self._buf) - keep :]
            if not self._inside:
                out.append(emit)
            break
        return "".join(out)
