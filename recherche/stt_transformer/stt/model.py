"""Modèle STT : Transformer + CTC.

D'où viennent les données, où elles vont
-----------------------------------------
    stt/audio.py      mp3 -> log-Mel normalisé (instants, 80)
    stt/text.py       phrase -> indices des caractères (cible)
    stt/dataset.py    regroupe les clips en batchs (padding) :
                          features (B, T, 80) + feature_lengths (B,)
                          targets  (B, L)     + target_lengths  (B,)
                              │
                              ▼
    stt/model.py      CE FICHIER : SpeechTransformer (classe en bas du fichier)
                              │
        features (B, T, 80)   │
          1. ConvSubsampling              -> (B, T/4, d_model)   réduit le temps par ~4
          2. SinusoidalPositionalEncoding -> (B, T/4, d_model)   ajoute l'ordre des instants
          3. n_layers × TransformerBlock  -> (B, T/4, d_model)   chaque instant prend du contexte
          4. LayerNorm finale + output head -> (B, T/4, 50)      un score par caractère + blank
          5. log_softmax                  -> log-probabilités     ce qu'attend la loss CTC
                              │
                              ▼
    (à venir) entraînement : nn.CTCLoss(log-probabilités, targets, longueurs)

Le fichier se lit de haut en bas dans l'ordre du pipeline : chaque section
correspond à une étape, et la classe SpeechTransformer (tout en bas) les assemble.
"""
import math

import torch
import torch.nn.functional as F
from torch import nn

from stt.text import VOCAB


# =============================================================================
# Étape 1 — Sous-échantillonnage : (B, T, 80) -> (B, T/4, d_model)
# =============================================================================

def conv_out_length(length):
    """Longueur de sortie d'une convolution de filtre 3, pas 2, sans padding.

    Formule générale : ⌊(entrée − taille du filtre) / pas⌋ + 1.
    Exemple : 466 instants -> 232. Fonctionne aussi sur un tenseur de longueurs.
    """
    return (length - 3) // 2 + 1


class ConvSubsampling(nn.Module):
    """Divise le nombre d'instants par ~4 avec deux convolutions 2D de pas 2.

    Pourquoi : le coût de l'attention croît avec le carré du nombre d'instants
    (466² ≈ 217 000 scores contre 115² ≈ 13 000 après réduction, soit 16 fois moins),
    et les convolutions apprennent des motifs locaux du spectrogramme
    (souffle d'un « s », transition consonne -> voyelle).
    """

    def __init__(self, n_mels, channels, d_model):
        # n_mels   : nombre de bandes Mel en entrée (80)
        # channels : nombre de filtres par convolution, donc de canaux en sortie
        # d_model  : taille des vecteurs à l'intérieur du Transformer
        super().__init__()  # obligatoire : initialise la partie nn.Module

        self.conv = nn.Sequential(
            # 1re convolution : le spectrogramme (1 canal) passe dans `channels` filtres 3×3.
            # Le pas de 2 divise le temps ET les fréquences par ~2 (466×80 -> 232×39).
            nn.Conv2d(1, channels, kernel_size=3, stride=2),
            # ReLU : remplace les valeurs négatives par 0 (non-linéarité).
            # Sans elle, deux convolutions de suite seraient équivalentes à UNE seule
            # opération linéaire : empiler des couches ne servirait à rien.
            nn.ReLU(),
            # 2e convolution : combine les motifs détectés par la 1re (232×39 -> 115×19).
            nn.Conv2d(channels, channels, kernel_size=3, stride=2),
            nn.ReLU(),
        )

        # Nombre de bandes de fréquence restant après les deux convolutions : 80 -> 39 -> 19
        freq_out = conv_out_length(conv_out_length(n_mels))

        # Chaque instant est décrit par channels × freq_out valeurs ;
        # on les projette vers la taille interne du modèle (d_model).
        self.linear = nn.Linear(channels * freq_out, d_model)

    def forward(self, x, lengths):
        # x       : (B, T, 80) features log-Mel paddées
        # lengths : (B,) vraies longueurs avant padding

        # Conv2d attend (batch, canaux, hauteur, largeur) : on ajoute la dimension « canal ».
        x = x.unsqueeze(1)                       # (B, 1, T, 80)

        x = self.conv(x)                         # (B, C, T', 19), avec T' ≈ T / 4

        b, c, t, f = x.shape                     # on récupère les 4 tailles

        # Le Transformer attend le temps en 2e position : (B, C, T', 19) -> (B, T', C, 19)
        x = x.permute(0, 2, 1, 3)

        # On met bout à bout les C canaux × 19 fréquences : chaque instant = un seul vecteur
        x = x.reshape(b, t, c * f)               # (B, T', C×19)

        x = self.linear(x)                       # (B, T', d_model)

        # Les vraies longueurs subissent la même réduction, sinon le CTC et le masque
        # d'attention utiliseraient des longueurs fausses.
        lengths = conv_out_length(conv_out_length(lengths))

        return x, lengths


# =============================================================================
# Étape 2 — Encodage des positions : (B, T', d_model) -> (B, T', d_model)
# =============================================================================

class SinusoidalPositionalEncoding(nn.Module):
    """Ajoute à chaque instant un vecteur qui encode sa position (Transformer original, 2017).

    Pourquoi : l'attention compare toutes les positions entre elles sans connaître leur ordre.
    Sans cette information, le modèle ne saurait pas quel son vient avant quel autre.
    """

    def __init__(self, d_model, max_len=10000):
        # max_len : nombre maximum d'instants (après sous-échantillonnage, 25 par seconde).
        # 10 000 instants = 400 s, largement au-dessus du clip le plus long (83 s ≈ 2 075 instants).
        super().__init__()

        # Numéros de position 0, 1, 2, ..., en colonne : (max_len, 1)
        position = torch.arange(max_len).unsqueeze(1)

        # Une fréquence par paire de dimensions : rapide pour les premières, lente pour les dernières.
        # C'est 1 / 10000^(2i / d_model), calculé avec exp et log pour éviter de trop grands nombres.
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)   # dimensions paires : sinus
        pe[:, 1::2] = torch.cos(position * div_term)   # dimensions impaires : cosinus

        # « buffer » : fait partie du module (suit .to("cuda")), mais n'est PAS appris.
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x : (B, T', d_model). On ajoute le motif des T' premières positions.
        # (T', d_model) s'additionne à (B, T', d_model) par broadcasting : même motif pour tout le batch.
        return x + self.pe[: x.shape[1]]


# =============================================================================
# Étape 3 — Blocs Transformer : (B, T', d_model) -> (B, T', d_model)
#   3a. masque du padding + attention à une tête (écrite à la main, vérifiée contre PyTorch)
#   3b. multi-head attention
#   3c. feed-forward
#   3d. bloc complet (pre-norm + résiduels)
# =============================================================================

def make_padding_mask(lengths, max_len):
    """Masque booléen (B, T) : True pour un vrai instant, False pour du padding.

    Exemple : lengths = [3, 1], max_len = 3  ->  [[True, True, True],
                                                   [True, False, False]]
    """
    positions = torch.arange(max_len, device=lengths.device)   # (T,)  : 0, 1, 2, ...
    # Comparaison (1, T) < (B, 1) -> (B, T) par broadcasting
    return positions[None, :] < lengths[:, None]


def scaled_dot_product_attention(q, k, v, mask):
    """Attention : softmax(Q·Kᵀ / √d_k) · V, en ignorant les instants de padding.

    Version écrite à la main pour comprendre le calcul ; vérifiée contre
    torch.nn.functional.scaled_dot_product_attention (version optimisée utilisée pour l'entraînement).

    q, k, v : (B, T, d_k)   mask : (B, T), True = vrai instant
    Renvoie la sortie (B, T, d_k) et les poids d'attention (B, T, T).
    """
    d_k = q.shape[-1]

    # Score de chaque paire (instant i, instant j) : à quel point i doit regarder j.
    # (B, T, d_k) @ (B, d_k, T) -> (B, T, T). Division par √d_k pour éviter des scores
    # trop grands, qui rendraient le softmax trop « pointu » et les gradients trop faibles.
    scores = q @ k.transpose(-2, -1) / math.sqrt(d_k)

    # Masque : aucun instant ne doit regarder le padding.
    # mask[:, None, :] a la forme (B, 1, T) : on masque des COLONNES (les instants regardés).
    # ~ inverse le masque : True là où il y a du padding -> score remplacé par -∞.
    scores = scores.masked_fill(~mask[:, None, :], float("-inf"))

    # Softmax sur la dernière dimension : pour chaque instant i, les poids sur tous
    # les instants j sont positifs et somment à 1 (le padding reçoit exactement 0).
    weights = torch.softmax(scores, dim=-1)

    # Moyenne pondérée des valeurs V : chaque instant reçoit un résumé du contexte.
    return weights @ v, weights


class MultiHeadSelfAttention(nn.Module):
    """Plusieurs attentions en parallèle (têtes), chacune sur un morceau du vecteur.

    Pourquoi : chaque tête peut se spécialiser (instants voisins, contexte plus lointain...)
    au lieu de faire un seul compromis.
    """

    def __init__(self, d_model, n_heads, use_pytorch_kernel=True):
        super().__init__()
        # Chaque tête reçoit d_model / n_heads dimensions : la division doit tomber juste.
        assert d_model % n_heads == 0, "d_model doit être divisible par n_heads"
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        # Une seule couche produit Q, K et V d'un coup (3 × d_model sorties) : plus rapide.
        self.qkv = nn.Linear(d_model, 3 * d_model)
        # Recombine les sorties des têtes
        self.out = nn.Linear(d_model, d_model)
        # True : version optimisée de PyTorch (entraînement). False : notre version (vérification).
        self.use_pytorch_kernel = use_pytorch_kernel

    def forward(self, x, mask):
        # x : (B, T, d_model)   mask : (B, T), True = vrai instant
        B, T, d_model = x.shape

        # Q, K, V : trois tenseurs (B, T, d_model)
        q, k, v = self.qkv(x).chunk(3, dim=-1)

        # Découpage en têtes : (B, T, d_model) -> (B, T, h, d_k) -> (B, h, T, d_k)
        # La dimension des têtes passe en 2e position : chaque tête est traitée comme
        # une séquence indépendante.
        q = q.view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_k).transpose(1, 2)

        # Masque de forme (B, 1, T) : la même règle s'applique à toutes les têtes.
        head_mask = mask[:, None, :]
        if self.use_pytorch_kernel:
            # Version PyTorch : masque attendu en (B, 1, 1, T)
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=head_mask[:, None, :, :])
        else:
            # Notre version (3a) : elle ajoute elle-même une dimension -> (B, 1, 1, T)
            out, _ = scaled_dot_product_attention(q, k, v, head_mask)

        # Recollage des têtes : (B, h, T, d_k) -> (B, T, h, d_k) -> (B, T, d_model)
        out = out.transpose(1, 2).reshape(B, T, d_model)
        return self.out(out)


class FeedForward(nn.Module):
    """Réseau appliqué à chaque instant séparément : Linear -> GELU -> Dropout -> Linear.

    Pourquoi : l'attention MÉLANGE l'information entre instants ; le feed-forward la
    TRANSFORME pour chaque instant (élargir à d_ff dimensions donne plus de capacité de calcul).
    """

    def __init__(self, d_model, d_ff, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),    # élargit : d_model -> d_ff (en général 4 × d_model)
            nn.GELU(),                   # non-linéarité (sinon deux Linear = un seul)
            nn.Dropout(dropout),         # met au hasard 10 % des valeurs à 0 (entraînement seulement)
            nn.Linear(d_ff, d_model),    # revient à d_model (indispensable pour le résiduel)
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    """Un bloc Transformer en version pre-norm : x + Attention(LayerNorm(x)), puis x + FFN(LayerNorm(x)).

    Pourquoi pre-norm : beaucoup plus stable à l'entraînement que la version d'origine (post-norm),
    précieux pour un premier modèle entraîné de zéro avec un budget GPU limité.
    """

    def __init__(self, d_model, n_heads, d_ff, dropout, use_pytorch_kernel=True):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attention = MultiHeadSelfAttention(d_model, n_heads, use_pytorch_kernel)
        self.norm2 = nn.LayerNorm(d_model)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        # 1. Attention : chaque instant puise de l'information dans les autres.
        #    Connexion résiduelle (x + ...) : l'information d'origine est conservée
        #    et les gradients circulent directement vers les couches précédentes.
        x = x + self.dropout(self.attention(self.norm1(x), mask))
        # 2. Feed-forward : chaque instant est transformé séparément.
        x = x + self.dropout(self.feed_forward(self.norm2(x)))
        return x


# =============================================================================
# Le modèle complet : assemble les étapes 1 à 5 dans l'ordre du pipeline
# =============================================================================

class SpeechTransformer(nn.Module):
    """Features log-Mel (B, T, 80) -> log-probabilités des caractères (B, T/4, 50).

    Valeurs par défaut = taille « S » (≈ 3,6 M paramètres), choisie pour nos 19 h de données :
    assez de capacité pour apprendre, sans trop de risque de mémoriser les 275 locuteurs.
    """

    def __init__(
        self,
        n_mels=80,               # entrée : bandes Mel (stt/audio.py)
        vocab_size=len(VOCAB),   # sortie : 49 caractères + blank = 50 (stt/text.py)
        d_model=144,             # taille des vecteurs dans le Transformer
        n_layers=12,             # nombre de blocs Transformer empilés
        n_heads=4,               # têtes d'attention (144 / 4 = 36 dimensions par tête)
        d_ff=576,                # taille interne du feed-forward (4 × d_model)
        conv_channels=144,       # filtres de chaque convolution du sous-échantillonnage
        dropout=0.1,
        use_pytorch_kernel=True, # attention optimisée de PyTorch (False : notre version)
    ):
        super().__init__()
        # Étape 1
        self.subsampling = ConvSubsampling(n_mels, conv_channels, d_model)
        # Étape 2
        self.positional_encoding = SinusoidalPositionalEncoding(d_model)
        self.dropout = nn.Dropout(dropout)
        # Étape 3 : ModuleList = liste de modules dont PyTorch suit les paramètres
        # (une liste Python ordinaire les cacherait à l'optimizer)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, d_ff, dropout, use_pytorch_kernel) for _ in range(n_layers)]
        )
        # Étape 4 : en pre-norm, la sortie du dernier bloc n'est jamais normalisée
        # (le chemin résiduel ne passe par aucune LayerNorm) : on ajoute donc une
        # LayerNorm finale avant l'output head, comme Whisper et les modèles récents.
        self.final_norm = nn.LayerNorm(d_model)
        self.output_head = nn.Linear(d_model, vocab_size)

    def forward(self, features, feature_lengths):
        # features        : (B, T, 80)  log-Mel paddés, venant de collate_batch (stt/dataset.py)
        # feature_lengths : (B,)        vraies longueurs avant padding

        # Étape 1 — sous-échantillonnage : le temps est divisé par ~4
        x, lengths = self.subsampling(features, feature_lengths)       # (B, T', d_model)

        # Étape 2 — ajout de la position de chaque instant
        x = self.dropout(self.positional_encoding(x))                  # (B, T', d_model)

        # Masque du padding, calculé une seule fois pour tous les blocs
        mask = make_padding_mask(lengths, x.shape[1])                  # (B, T')

        # Étape 3 — les blocs Transformer, l'un après l'autre
        for block in self.blocks:
            x = block(x, mask)                                         # (B, T', d_model)

        # Étape 4 — normalisation finale, puis un score par symbole du vocabulaire
        x = self.final_norm(x)
        logits = self.output_head(x)                                   # (B, T', 50)

        # Étape 5 — log-probabilités : nn.CTCLoss attend log(softmax), calculé ici
        # en une seule opération numériquement stable (évite log(0))
        log_probs = logits.log_softmax(dim=-1)                         # (B, T', 50)

        # Les longueurs réduites servent au CTC : combien d'instants valides par clip
        return log_probs, lengths
