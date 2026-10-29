"""Outils d'entraînement : scheduler du learning rate et évaluation sur dev/test.

Pipeline d'une époque :
    batchs train -> forward -> loss CTC -> backward -> optimizer.step() -> scheduler.step()
    fin d'époque -> evaluate(dev) -> loss, WER, CER, exemples

Décisions : scheduler A (warmup 5 % puis cosinus, lr max 1e-3), 200 époques — voir JOURNAL_PROJET.md.
"""
import math

import torch

from stt.metrics import greedy_decode, error_rate
from stt.text import decode


def warmup_cosine(step, warmup_steps, total_steps):
    """Multiplicateur du learning rate (entre 0 et 1) à l'itération `step`."""
    if step < warmup_steps:
        return (step + 1) / warmup_steps                              # montée linéaire : ~0 -> 1 (+1 : le 1er pas n'est pas nul)
    progress = (step - warmup_steps) / (total_steps - warmup_steps)  # 0 -> 1 pendant la décroissance
    return 0.5 * (1 + math.cos(math.pi * progress))                   # demi-vague cosinus : 1 -> 0


@torch.no_grad()                                   # pas de gradients : évaluation seulement (moins de mémoire, plus rapide)
def evaluate(model, loader, ctc_loss, device):
    """Loss, WER et CER sur un jeu jamais utilisé pour apprendre (dev ou test)."""
    model.eval()                                   # dropout coupé : on mesure le vrai modèle
    total_loss, n_batches = 0.0, 0
    references, hypotheses = [], []
    for batch in loader:
        features, feature_lengths, targets, target_lengths = [t.to(device) for t in batch]
        log_probs, out_lengths = model(features, feature_lengths)
        total_loss += ctc_loss(log_probs.transpose(0, 1), targets, out_lengths, target_lengths).item()
        n_batches += 1
        hypotheses += greedy_decode(log_probs, out_lengths)                       # ce que le modèle a compris
        references += [decode(targets[b, :target_lengths[b]].tolist())           # ce qu'il fallait comprendre
                       for b in range(targets.shape[0])]
    model.train()                                  # on remet le mode entraînement pour la suite
    return {"loss": total_loss / n_batches,
            "wer": error_rate(references, hypotheses, "mot"),
            "cer": error_rate(references, hypotheses, "caractère"),
            "examples": list(zip(references, hypotheses))[:3]}   # 3 exemples à afficher
