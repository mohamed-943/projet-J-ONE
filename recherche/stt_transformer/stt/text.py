"""Normalisation et encodage du texte.

Décisions : EDA point 2 (caractères), prétraitement étape 5 (vocabulaire).
"""
import re
import unicodedata

import torch

# Caractères autorisés : lettres françaises + espace + ponctuation retenue
ALLOWED = set("abcdefghijklmnopqrstuvwxyz" "àâæçéèêëîïôœùûüÿ" " '-.,?!")

# Vocabulaire : le blank du CTC en position 0, puis les 49 caractères triés
BLANK = "_"
VOCAB = [BLANK] + sorted(ALLOWED)
CHAR_TO_ID = {char: i for i, char in enumerate(VOCAB)}
ID_TO_CHAR = {i: char for i, char in enumerate(VOCAB)}


def normalize(text):
    text = unicodedata.normalize("NFC", text)        # recolle les accents décomposés
    text = text.lower()
    text = re.sub(r"[’‘ʼ´]", "'", text)              # une seule apostrophe
    text = re.sub(r"[;:…«»()\"—–]", " ", text)       # ponctuation secondaire -> espace
    text = re.sub(r" +", " ", text).strip()
    return text


def is_valid(text):
    return set(text) <= ALLOWED


def encode(text):
    # Texte -> tenseur d'indices (la cible de la loss CTC)
    return torch.tensor([CHAR_TO_ID[char] for char in text], dtype=torch.long)


def decode(ids):
    # Indices -> texte
    return "".join(ID_TO_CHAR[i] for i in ids)
