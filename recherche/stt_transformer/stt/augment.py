"""SpecAugment : masques aléatoires sur le log-Mel, pour lutter contre l'overfitting.

Pipeline : log-Mel (T, 80) d'un clip du TRAIN -> spec_augment -> même forme, zones mises à 0
           (appliqué dans CommonVoiceDataset quand augment=True ; jamais sur le dev ni le test)

Décisions : masques écrits nous-mêmes (format (T, 80), sans transpose), taille « moyenne »
(2 x ≤ 27 bandes, 2 x ≤ 40 instants et ≤ 10 % du clip) — voir JOURNAL_PROJET.md.
"""
import torch


def spec_augment(features, n_freq_masks=2, max_freq_width=27,
                 n_time_masks=2, max_time_width=40, max_time_ratio=0.1):
    """Masque au hasard des bandes de fréquences et des morceaux de temps (entraînement seulement).

    features : (T, 80) log-Mel normalisé -> même forme, avec des zones mises à 0
    (0 = la moyenne de chaque bande, grâce à la normalisation : une valeur "neutre")
    """
    features = features.clone()                       # copie : on ne modifie jamais l'original
    n_frames, n_mels = features.shape                 # T instants, 80 bandes

    # Masques de FRÉQUENCES : des bandes Mel voisines effacées sur toute la durée
    for _ in range(n_freq_masks):
        width = torch.randint(0, max_freq_width + 1, (1,)).item()   # largeur tirée entre 0 et 27 bandes
        start = torch.randint(0, n_mels - width + 1, (1,)).item()   # position de départ tirée au hasard
        features[:, start:start + width] = 0                         # tous les instants, ces bandes

    # Masques de TEMPS : un court morceau du clip effacé sur toutes les fréquences
    time_limit = min(max_time_width, int(max_time_ratio * n_frames))  # ≤ 0,4 s ET ≤ 10 % du clip
    for _ in range(n_time_masks):
        width = torch.randint(0, time_limit + 1, (1,)).item()
        start = torch.randint(0, n_frames - width + 1, (1,)).item()
        features[start:start + width, :] = 0                         # ces instants, toutes les bandes

    return features
