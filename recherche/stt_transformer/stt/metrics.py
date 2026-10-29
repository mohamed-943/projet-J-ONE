"""Décodage de la sortie du modèle et métriques WER / CER.

Pipeline : log-probabilités du modèle (B, T', 50)
           -> greedy_decode -> phrases prédites
           -> error_rate(références, prédictions) -> WER ou CER en %

Validé par le test d'overfitting sur 10 clips (stt_03_overfit_10_clips.ipynb).
"""
from stt.text import decode


def greedy_decode(log_probs, lengths):
    """Sortie du modèle -> phrases, avec les règles du CTC."""
    best = log_probs.argmax(dim=-1)                 # (B, T') : le symbole le plus probable à chaque instant
    sentences = []
    for b in range(best.shape[0]):
        ids, previous = [], None
        for i in best[b, :lengths[b]].tolist():     # seulement les vrais instants (pas le padding)
            if i != previous and i != 0:            # 1) fusion des répétitions  2) suppression des blanks (indice 0)
                ids.append(i)
            previous = i
        sentences.append(decode(ids))
    return sentences


def edit_distance(ref, hyp):
    """Nombre minimum de substitutions + suppressions + insertions pour passer de ref à hyp.

    Programmation dynamique : on ne garde que la ligne précédente du tableau.
    """
    previous_row = list(range(len(hyp) + 1))        # passer de "" à hyp[:j] = j insertions
    for i, r in enumerate(ref, start=1):
        row = [i]                                   # passer de ref[:i] à "" = i suppressions
        for j, h in enumerate(hyp, start=1):
            row.append(min(previous_row[j] + 1,                 # suppression
                           row[j - 1] + 1,                      # insertion
                           previous_row[j - 1] + (r != h)))     # substitution (0 si identique)
        previous_row = row
    return previous_row[-1]


def error_rate(references, hypotheses, unit):
    """WER (unit="mot") ou CER (unit="caractère"), en %.

    Les erreurs sont additionnées sur toutes les phrases, puis divisées par le
    nombre total de mots (ou caractères) des références : c'est le calcul standard.
    """
    split = (lambda s: s.split()) if unit == "mot" else list
    errors = sum(edit_distance(split(r), split(h)) for r, h in zip(references, hypotheses))
    total = sum(len(split(r)) for r in references)
    return 100 * errors / total
