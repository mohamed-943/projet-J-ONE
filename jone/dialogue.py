"""Réponses du robot décidées sans LLM : commandes de langue et demandes de
répétition.

Un petit LLM local suit mal ces consignes : à « parle en anglais » dit en français,
Qwen2.5 3B répond « D'accord, je vais parler en anglais »… en français ; marqué
« transcription peu sûre », il fait répéter environ une fois sur trois seulement
(mesuré sur la Jetson). Ces cas sont donc traités par des règles et des phrases fixes.
"""

from __future__ import annotations

import re
import unicodedata

CLARIFY_REPLIES = {
    "fr": "Pardon, je n'ai pas bien compris. Pouvez-vous répéter ?",
    "en": "Sorry, I didn't quite catch that. Could you say it again?",
}
LANGUAGE_SWITCH_REPLIES = {
    "fr": "D'accord, je continue en français.",
    "en": "Sure, I'll speak English from now on.",
}

_TARGETS = {"anglais": "en", "english": "en", "francais": "fr", "french": "fr"}
_VERBS = {
    "parle", "parlez", "parler", "parlons", "reponds", "repondez", "repondre", "passe", "passez",
    "passer", "passons", "continue", "continuez", "continuer", "continuons", "switch", "switcher",
    "speak", "talk", "answer", "reply", "respond", "change", "go", "use",
}
# Négations et questions sur les capacités : « tu ne parles pas anglais »,
# « tu peux parler d'autres langues que le français ? »
_EXCLUDE = {"pas", "not", "dont", "autre", "autres", "other", "langue", "langues",
            "language", "languages", "combien", "how", "many"}
# Mots de politesse, ignorés pour juger qu'une phrase courte est une simple demande
# (« En français s'il te plaît »).
_POLITE = {"s", "il", "te", "vous", "plait", "please", "stp", "svp", "si", "possible", "merci",
           "thanks", "sarah", "maintenant", "now", "ok", "okay", "d", "accord", "oui", "yes"}
_WORD = re.compile(r"[a-z]+")  # « l'anglais » → l, anglais
MAX_WORDS = 12


def _fold(text: str) -> str:
    text = unicodedata.normalize("NFD", text.lower().replace("’", "'"))
    return "".join(c for c in text if unicodedata.category(c) != "Mn")


def language_command(text: str) -> str | None:
    """Langue demandée (« fr », « en ») si le texte est une demande de changement de
    langue, sinon None. Exemples : « Parle en anglais », « Maintenant on parle en
    anglais si possible », « Speak French please », « In English please »."""
    words = _WORD.findall(_fold(text))
    if not words or len(words) > MAX_WORDS or _EXCLUDE.intersection(words):
        return None
    targets = {_TARGETS[w] for w in words if w in _TARGETS}
    if len(targets) != 1:  # aucune langue, ou « du français à l'anglais » : ambigu
        return None
    if _VERBS.intersection(words) or len([w for w in words if w not in _POLITE]) <= 2:
        return targets.pop()
    return None
