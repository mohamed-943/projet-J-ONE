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

Les proportions sont données avec leur intervalle de confiance de Wilson à 95 % :
avec quelques dizaines d'enregistrements, l'incertitude n'est pas négligeable.
"""

from __future__ import annotations

import math
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

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

    @property
    def detected(self) -> bool:
        """Le robot aurait répondu (en écoute continue)."""
        return bool(self.accepted)


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
        "latence_stt_ms": summarize([ms for r in results for ms in r.stt_ms]),
        "parole": {
            "FRR": _rate(sum(not r.detected for r in speech), len(speech)),
            "FAR": _rate(sum(r.detected for r in noise), len(noise)),
        },
        "mot_eveil": {
            "FRR": _rate(sum(not r.wake for r in commands), len(commands)),
            "FAR": _rate(sum(r.wake for r in not_addressed), len(not_addressed)),
        },
    }
