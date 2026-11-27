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
    "fr": {
        "%": " pour cent",
        "&": " et ",
        "+": " plus ",
        "=": " égale ",
        "€": " euros",
        "$": " dollars",
        "°C": " degrés",
        "°": " degrés",
        "/": " sur ",
    },
    "en": {
        "%": " percent",
        "&": " and ",
        "+": " plus ",
        "=": " equals ",
        "€": " euros",
        "$": " dollars",
        "°C": " degrees Celsius",
        "°F": " degrees Fahrenheit",
        "°": " degrees",
        "/": " per ",
    },
}


def _is_emoji_or_symbol(ch: str) -> bool:
    cat = unicodedata.category(ch)
    # So : symboles divers (emoji, pictogrammes) ; Cs/Co : surrogates et usage privé.
    return cat in ("So", "Cs", "Co") or ch in "‍️"


def clean_for_speech(text: str, language: str = "fr") -> str:
    text = _LINK.sub(r"\1", text)
    text = _URL.sub("", text)
    text = _BULLET.sub("", text)
    text = _MARKDOWN.sub("", text)
    symbols = _SYMBOL_WORDS.get(language, _SYMBOL_WORDS["fr"])
    for sym, word in symbols.items():  # avant le filtre : « ° » est un symbole
        text = text.replace(sym, word)
    text = "".join(ch for ch in text if not _is_emoji_or_symbol(ch))
    text = re.sub(r"[#*_<>|\\^{}\[\]]", " ", text)
    text = text.replace("\n", " ")
    text = _SPACES.sub(" ", text)
    text = re.sub(r"\s+([,.])", r"\1", text)
    return text.strip()


# Fin de phrase : ponctuation forte suivie d'un espace. On évite les abréviations
# courantes (M., Mme., Dr., Mr., Mrs., etc.) et les nombres décimaux (3.5 n'a pas
# d'espace).
_SENTENCE_END = re.compile(
    r"(?<!\bM)(?<!\bDr)(?<!\bMme)(?<!\betc)(?<!\bMr)(?<!\bMrs)(?<!\bMs)([.!?…]+)[\"»)]?\s+"
)


class SentenceSplitter:
    """Accumule des fragments de texte et renvoie les phrases complètes.

    `first_clause_chars` : le premier morceau de la réponse part dès la première
    virgule passé ce nombre de caractères, sans attendre la fin de la phrase. Le
    robot commence ainsi à parler plus tôt ; la suite est synthétisée pendant qu'il
    prononce ce début. 0 désactive ce découpage.
    """

    def __init__(self, soft_limit: int = 140, first_clause_chars: int = 0):
        self.soft_limit = soft_limit
        self.first_clause_chars = first_clause_chars
        self._buf = ""
        self._emitted = False

    def feed(self, delta: str) -> list[str]:
        self._buf += delta
        out: list[str] = []
        while True:
            m = _SENTENCE_END.search(self._buf)
            if m:
                out.append(self._buf[: m.end()].strip())
                self._buf = self._buf[m.end() :]
                continue
            if self.first_clause_chars and not self._emitted and not out:
                cut = self._buf.find(", ", self.first_clause_chars)
                if cut > 0:
                    out.append(self._buf[: cut + 1].strip())
                    self._buf = self._buf[cut + 2 :]
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
        out = [s for s in out if s]
        if out:
            self._emitted = True
        return out

    def flush(self) -> list[str]:
        rest, self._buf = self._buf.strip(), ""
        return [rest] if rest else []


_WORDS = re.compile(r"[a-zàâäçéèêëîïôöùûüœ']+")
_FR_WORDS = frozenset(
    "le la les un une des du de est et je tu il elle nous vous ils sont pas que qui "
    "pour dans avec sur mais ou je suis c'est j'ai votre vos mon ma mes très bien oui "
    "bonjour merci au aux ce cette ces ne".split()
)
_EN_WORDS = frozenset(
    "the an is are am and i you he she we they not that who for in with on but or "
    "it's i'm your my very well yes hello thanks thank to of this these what do does "
    "can be".split()
)


def guess_language(text: str, default: str = "fr") -> str:
    """Français ou anglais, d'après les mots courants et les accents du texte.

    Sert à choisir la voix de synthèse d'après la réponse elle-même : un petit LLM
    local répond parfois en français à une question posée en anglais."""
    words = _WORDS.findall(text.lower())
    fr = sum(w in _FR_WORDS for w in words) + sum(ch in "àâçéèêëîïôùûœ" for ch in text.lower())
    en = sum(w in _EN_WORDS for w in words)
    if fr == en:
        return default
    return "fr" if fr > en else "en"


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
