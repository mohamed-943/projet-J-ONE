"""Évaluation du front-end vocal sur un corpus étiqueté (docs/protocole_evaluation.md).

Chaque enregistrement du protocole passe par la même chaîne que dans le pipeline :
détection de parole (Silero VAD), transcription (Whisper), filtre des transcriptions
douteuses, mot d'éveil. On en tire quatre indicateurs :

- WER : taux d'erreur de mots de la transcription, par langue ;
- précision de reconnaissance des commandes : mot d'éveil détecté et mots-clés de la
  commande présents dans la transcription ;
- latence de la transcription (la latence de bout en bout vient du journal des
  échanges, mesurée en conversation réelle) ;
- FAR / FRR de la détection vocale, à deux niveaux : la parole (un bruit déclenche-t-il
  une réponse ? une phrase est-elle perdue ?) et le mot d'éveil.

Le niveau de la voix et le rapport signal/bruit (SNR) de chaque enregistrement
expliquent les résultats : en s'éloignant du robot, la voix baisse alors que le
bruit de fond reste le même.

Les proportions sont données avec leur intervalle de confiance de Wilson à 95 % :
avec quelques dizaines d'enregistrements, l'incertitude n'est pas négligeable.
"""

from __future__ import annotations

import math
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .metrics import corpus_wer, summarize

SPEECH_TYPES = ("phrase", "commande", "hors_cible")


@dataclass
class Item:
    id: str
    type: str  # phrase | commande | hors_cible | bruit
    langue: str
    texte: str  # référence ; consigne pour un bruit
    mots_cles: list[str] = field(default_factory=list)

    @property
    def is_speech(self) -> bool:
        return self.type in SPEECH_TYPES


def load_protocol(path: Path) -> list[Item]:
    items = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        cols = line.split("\t") + [""] * 5
        id_, type_, langue, texte, mots = (c.strip() for c in cols[:5])
        if type_ not in SPEECH_TYPES + ("bruit",):
            raise ValueError(f"{path} : type inconnu « {type_} » pour {id_}")
        items.append(Item(id_, type_, langue, texte, [m.strip() for m in mots.split("|") if m.strip()]))
    return items


@dataclass
class ItemResult:
    item: Item
    segments: int = 0  # énoncés découpés par le VAD
    transcription: str = ""  # tous les énoncés, avant filtre (pour le WER)
    accepted: str = ""  # énoncés acceptés par le filtre (ce que le LLM aurait reçu)
    rejections: list[str] = field(default_factory=list)
    wake: bool = False  # mot d'éveil reconnu dans un énoncé accepté
    language: str | None = None
    stt_ms: list[float] = field(default_factory=list)
    # (énoncé accepté sans tenir compte de la confiance, avg_logprob) : pour la courbe
    # FAR / FRR selon le seuil de confiance
    segment_scores: list[tuple[bool, float]] = field(default_factory=list)
    uncertain: bool = False  # tous les énoncés acceptés étaient incertains : le robot ferait répéter
    voice_db: float | None = None  # niveau de la voix (dBFS)
    noise_db: float | None = None  # bruit de fond (dBFS)

    @property
    def snr_db(self) -> float | None:
        if self.voice_db is None or self.noise_db is None:
            return None
        return self.voice_db - self.noise_db

    @property
    def detected(self) -> bool:
        """Le robot aurait répondu (en écoute continue)."""
        return bool(self.accepted)


def audio_levels(audio: np.ndarray, frame: int = 512) -> tuple[float, float]:
    """(niveau de la voix, bruit de fond) en dBFS, d'après l'énergie de trames de
    32 ms : 90e centile pour la voix, 10e pour le bruit. Ne dépend pas du VAD, dont
    on mesure justement les échecs."""
    n = len(audio) // frame
    if n == 0:
        return -120.0, -120.0
    frames = audio[: n * frame].astype(np.float64).reshape(n, frame) / 32768.0
    energy = np.mean(frames ** 2, axis=1)
    energy = energy[energy > 0]  # silence numérique (zéros ajoutés) : pas un bruit de fond
    if energy.size == 0:
        return -120.0, -120.0
    db = 10 * np.log10(energy)
    return float(np.percentile(db, 90)), float(np.percentile(db, 10))


def _median(values: list[float]) -> float | None:
    return float(np.median(values)) if values else None


def _fold(text: str) -> str:
    text = unicodedata.normalize("NFD", text.lower())
    return "".join(c for c in text if unicodedata.category(c) != "Mn")


def keywords_found(text: str, keywords: list[str]) -> bool:
    folded = _fold(text)
    return all(_fold(k) in folded for k in keywords)


def command_recognized(r: ItemResult) -> bool:
    return r.wake and keywords_found(r.accepted, r.item.mots_cles)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float | None, float | None, float | None]:
    """Proportion k/n et son intervalle de confiance de Wilson."""
    if n == 0:
        return None, None, None
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return p, max(0.0, centre - half), min(1.0, centre + half)


def _rate(k: int, n: int) -> dict:
    p, lo, hi = wilson(k, n)
    return {"k": k, "n": n, "taux": p, "ic95": [lo, hi]}


def compute_metrics(results: list[ItemResult]) -> dict:
    speech = [r for r in results if r.item.is_speech]
    noise = [r for r in results if r.item.type == "bruit"]
    commands = [r for r in results if r.item.type == "commande"]
    not_addressed = [r for r in results if r.item.type in ("hors_cible", "bruit")]

    wer = {"global": corpus_wer([(r.item.texte, r.transcription) for r in speech]), "n": len(speech)}
    for lang in sorted({r.item.langue for r in speech}):
        subset = [r for r in speech if r.item.langue == lang]
        wer[lang] = corpus_wer([(r.item.texte, r.transcription) for r in subset])

    return {
        "wer": wer,
        "reconnaissance_commandes": _rate(sum(command_recognized(r) for r in commands), len(commands)),
        "mots_cles_seuls": _rate(sum(keywords_found(r.transcription, r.item.mots_cles) for r in commands),
                                 len(commands)),
        "identification_langue": _rate(sum(r.language == r.item.langue for r in speech), len(speech)),
        "clarification": _rate(sum(r.detected and r.uncertain for r in speech), len(speech)),
        "latence_stt_ms": summarize([ms for r in results for ms in r.stt_ms]),
        "niveau_voix_db": _median([r.voice_db for r in speech if r.voice_db is not None]),
        "snr_db": _median([r.snr_db for r in speech if r.snr_db is not None]),
        "parole": {
            "FRR": _rate(sum(not r.detected for r in speech), len(speech)),
            "FAR": _rate(sum(r.detected for r in noise), len(noise)),
        },
        "mot_eveil": {
            "FRR": _rate(sum(not r.wake for r in commands), len(commands)),
            "FAR": _rate(sum(r.wake for r in not_addressed), len(not_addressed)),
        },
    }


CURVE_THRESHOLDS = (-1.6, -1.4, -1.3, -1.2, -1.1, -1.0, -0.9, -0.8, -0.7, -0.6, -0.5)


def confidence_curve(results: list[ItemResult], thresholds=CURVE_THRESHOLDS) -> list[dict]:
    """FRR (parole perdue) et FAR (bruit accepté) pour chaque seuil de rejet sur
    avg_logprob, les autres règles du filtre restant appliquées. Sert à choisir les
    seuils `reject_avg_logprob` et `uncertain_avg_logprob` à partir des données."""
    speech = [r for r in results if r.item.is_speech]
    noise = [r for r in results if r.item.type == "bruit"]

    def passes(r, t):
        return any(ok and lp >= t for ok, lp in r.segment_scores)

    return [{"seuil": t,
             "FRR": _rate(sum(not passes(r, t) for r in speech), len(speech)),
             "FAR": _rate(sum(passes(r, t) for r in noise), len(noise))} for t in thresholds]
