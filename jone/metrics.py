"""Métriques du rapport de tests : taux d'erreur de mots (ENF-02) et statistiques
de latence (ENF-01)."""

from __future__ import annotations

import re
import unicodedata


def normalize_for_wer(text: str) -> list[str]:
    """Minuscules, sans ponctuation, apostrophes séparées (« l'école » → l école)."""
    text = unicodedata.normalize("NFC", text).lower().replace("’", "'")
    text = re.sub(r"[^\w\s'-]", " ", text)
    text = text.replace("'", " ").replace("-", " ")
    return text.split()


def word_errors(reference: str, hypothesis: str) -> tuple[int, int]:
    """(substitutions + suppressions + insertions, nombre de mots de la référence)."""
    ref, hyp = normalize_for_wer(reference), normalize_for_wer(hypothesis)
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i] + [0] * len(hyp)
        for j, h in enumerate(hyp, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h))
        prev = cur
    return prev[-1], len(ref)


def wer(reference: str, hypothesis: str) -> float:
    errors, n = word_errors(reference, hypothesis)
    return errors / n if n else float(errors > 0)


def corpus_wer(pairs: list[tuple[str, str]]) -> float:
    """WER global pondéré par la longueur des phrases (référence, hypothèse)."""
    errors = total = 0
    for ref, hyp in pairs:
        e, n = word_errors(ref, hyp)
        errors += e
        total += n
    return errors / total if total else 0.0


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    s = sorted(values)
    k = (len(s) - 1) * p / 100
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def summarize(values: list[float]) -> dict:
    return {
        "n": len(values),
        "moyenne": sum(values) / len(values) if values else None,
        "p50": percentile(values, 50),
        "p90": percentile(values, 90),
        "max": max(values) if values else None,
    }
