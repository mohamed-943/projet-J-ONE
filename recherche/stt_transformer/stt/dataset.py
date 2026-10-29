"""Dataset PyTorch : un clip Common Voice -> (features log-Mel, cible encodée)."""
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from stt.audio import load_audio, log_mel
from stt.text import normalize, is_valid, encode


def load_split(folders, split):
    # Lit le fichier d'un découpage (train, dev ou test) pour plusieurs segments
    # (ex. fr-FR et fr-CA) et les assemble en un seul tableau
    tables = []
    for folder in folders:
        df = pd.read_csv(f"{folder}/{split}.tsv", sep="\t")
        df["audio_path"] = folder + "/clips/" + df["path"]   # chemin complet du mp3
        tables.append(df)
    df = pd.concat(tables, ignore_index=True)

    # Normalisation du texte et retrait des phrases non valides (EDA point 2)
    df["text"] = df["sentence"].map(normalize)
    df = df[df["text"].map(is_valid)].reset_index(drop=True)

    # On ne garde que les 3 colonnes utiles (EDA point 1)
    return df[["audio_path", "text", "client_id"]]


class CommonVoiceDataset(Dataset):
    def __init__(self, table):
        # Appelé une fois, à la création : on mémorise le tableau des clips
        self.table = table

    def __len__(self):
        # Nombre de clips
        return len(self.table)

    def __getitem__(self, i):
        # Prétraite le clip n° i à la demande et renvoie (features, cible)
        row = self.table.iloc[i]
        features = log_mel(load_audio(row["audio_path"]))   # (instants, 80)
        target = encode(row["text"])                          # (nombre de caractères,)
        return features, target


def collate_batch(batch):
    # batch = liste de paires (features, cible) renvoyées par __getitem__
    features_list, targets_list = zip(*batch)

    # Vraies longueurs, avant padding (nécessaires au CTC et au masque d'attention)
    feature_lengths = torch.tensor([f.shape[0] for f in features_list])
    target_lengths = torch.tensor([t.shape[0] for t in targets_list])

    # Complète avec des 0 jusqu'à la longueur du plus long élément du batch
    features = pad_sequence(features_list, batch_first=True)   # (batch, instants max, 80)
    targets = pad_sequence(targets_list, batch_first=True)     # (batch, caractères max)

    return features, feature_lengths, targets, target_lengths
