"""Conversions audio vers le format exigé par le haut-parleur du G1 (EF-09) :
PCM 16 kHz, mono, 16 bits signés little-endian.
"""

from __future__ import annotations

from math import gcd

import numpy as np

TARGET_RATE = 16000


def to_mono(samples: np.ndarray, channels: int = 1) -> np.ndarray:
    """Accepte un tableau (n,) entrelacé ou (n, canaux) et renvoie (n,)."""
    if samples.ndim == 2:
        return samples.mean(axis=1)
    if channels > 1:
        return samples.reshape(-1, channels).mean(axis=1)
    return samples


def resample(samples: np.ndarray, src_rate: int, dst_rate: int = TARGET_RATE) -> np.ndarray:
    if src_rate == dst_rate or samples.size == 0:
        return samples
    from scipy.signal import resample_poly

    g = gcd(src_rate, dst_rate)
    return resample_poly(samples, dst_rate // g, src_rate // g)


def to_float32(samples: np.ndarray) -> np.ndarray:
    if samples.dtype == np.int16:
        return samples.astype(np.float32) / 32768.0
    return samples.astype(np.float32)


def float_to_int16(samples: np.ndarray) -> np.ndarray:
    return (np.clip(samples, -1.0, 1.0) * 32767.0).astype(np.int16)


def to_pcm16k_mono(samples: np.ndarray, sample_rate: int, channels: int = 1) -> bytes:
    """Convertit n'importe quel signal (int16 ou float) en octets PCM 16 kHz mono 16 bits."""
    mono = to_mono(to_float32(np.asarray(samples)), channels)
    mono = resample(mono, sample_rate, TARGET_RATE)
    return float_to_int16(mono).astype("<i2").tobytes()


def pcm_bytes_to_int16(pcm: bytes) -> np.ndarray:
    return np.frombuffer(pcm, dtype="<i2")


def pcm_duration_s(pcm: bytes, sample_rate: int = TARGET_RATE) -> float:
    return len(pcm) / 2 / sample_rate
