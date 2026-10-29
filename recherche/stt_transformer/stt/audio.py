"""Lecture de l'audio et calcul des features log-Mel.

Décisions : EDA point 4 (16 kHz), prétraitement étapes 1 à 4 (25 ms / 10 ms / 80 Mel, normalisation par clip).
"""
import soundfile as sf
import torch
import torchaudio.functional as AF
import torchaudio.transforms as T

TARGET_SR = 16000
N_FFT = 400        # fenêtre de 25 ms
HOP_LENGTH = 160   # pas de 10 ms
N_MELS = 80

# Créé une seule fois, réutilisé pour tous les clips
_mel_transform = T.MelSpectrogram(
    sample_rate=TARGET_SR, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS
)


def load_audio(path):
    # mp3 -> tenseur 1D à 16 kHz
    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim == 2:                  # par sécurité : stéréo -> mono
        audio = audio.mean(axis=1)
    waveform = torch.from_numpy(audio)
    if sr != TARGET_SR:
        waveform = AF.resample(waveform, orig_freq=sr, new_freq=TARGET_SR)
    return waveform


def log_mel(waveform):
    # Tenseur 1D à 16 kHz -> features normalisées (instants, 80)
    mel = _mel_transform(waveform)
    features = torch.log(mel + 1e-6).T
    mean = features.mean(dim=0, keepdim=True)
    std = features.std(dim=0, keepdim=True)
    return (features - mean) / (std + 1e-5)
