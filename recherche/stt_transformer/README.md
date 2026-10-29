# Recherche — Transformer STT entraîné de zéro

Travail de recherche, séparé du pipeline de production (`jone/`) : concevoir et entraîner notre propre modèle de reconnaissance vocale en français (Transformer + CTC), puis le comparer à Whisper, le modèle utilisé par J-ONE.

Ce dossier ne modifie rien dans `jone/`.

## Contenu

| Fichier | Rôle |
|---|---|
| `00_signal_audio.ipynb` | Expériences pédagogiques : forme d'onde, FFT, STFT |
| `stt_01_exploration.ipynb` | Analyse exploratoire des données Common Voice (EDA en 7 points) |
| `JOURNAL_PROJET.md` | Journal des expériences, résultats et décisions, qui servira au rapport |

## Installation

Ce dossier a son propre environnement, géré avec [uv](https://docs.astral.sh/uv/) :

```bash
cd recherche/stt_transformer
uv sync
```

## Données

Les données ne sont **pas** versionnées. Les conditions de Mozilla Data Collective interdisent de les redistribuer. Pour les obtenir, créer un compte sur [mozilladatacollective.com](https://mozilladatacollective.com), télécharger les datasets ci-dessous, puis les décompresser dans `data/` :

- Common Voice Scripted Speech 26.0 — French of France (segment)
- Common Voice Scripted Speech 26.0 — Canadian French (segment)
- Common Voice Spontaneous Speech 5.0 — French
