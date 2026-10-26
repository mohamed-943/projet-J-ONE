"""Enregistre un corpus de test pour mesurer le WER (ENF-02).

    python tools/enregistrer_corpus.py --phrases tests/corpus_phrases.txt --out corpus/calme

Pour chaque phrase du fichier : Entrée pour démarrer, lire la phrase à voix haute,
Entrée pour arrêter. Produit NNN.wav (16 kHz mono) + NNN.txt (référence).
Utiliser --input g1_multicast pour enregistrer avec les micros du robot.
"""

from __future__ import annotations

import argparse
import threading
import wave
from pathlib import Path

import numpy as np

import _bootstrap  # noqa: F401
from jone.audio_input import create_mic
from jone.config import load_config


def record_until_enter(mic) -> np.ndarray:
    frames, stop = [], threading.Event()

    def reader():
        while not stop.is_set():
            f = mic.read(timeout=0.2)
            if f is not None:
                frames.append(f)

    mic.flush()
    t = threading.Thread(target=reader, daemon=True)
    t.start()
    input("  … enregistrement, Entrée pour arrêter ")
    stop.set()
    t.join()
    return np.concatenate(frames) if frames else np.zeros(0, dtype=np.int16)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--phrases", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--input", choices=["sounddevice", "g1_multicast"], default="sounddevice")
    args = p.parse_args()

    cfg = load_config(args.config)
    cfg["audio_input"]["backend"] = args.input
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    phrases = [l.strip() for l in Path(args.phrases).read_text(encoding="utf-8").splitlines() if l.strip()]

    mic = create_mic(cfg)
    mic.start()
    try:
        for i, phrase in enumerate(phrases, 1):
            input(f"\n[{i}/{len(phrases)}] « {phrase} »\n  Entrée pour démarrer ")
            audio = record_until_enter(mic)
            stem = out / f"{i:03d}"
            with wave.open(str(stem.with_suffix(".wav")), "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(cfg["audio_input"]["sample_rate"])
                w.writeframes(audio.astype("<i2").tobytes())
            stem.with_suffix(".txt").write_text(phrase + "\n", encoding="utf-8")
            print(f"  -> {stem}.wav ({len(audio) / cfg['audio_input']['sample_rate']:.1f} s)")
    finally:
        mic.close()


if __name__ == "__main__":
    main()
