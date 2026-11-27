"""Enregistre un corpus de test (WER, ENF-02 ; évaluation, docs/protocole_evaluation.md).

    python tools/enregistrer_corpus.py --phrases tests/corpus_phrases.txt --out corpus/calme
    python tools/enregistrer_corpus.py --protocole tests/protocole_evaluation.tsv         --out corpus/eval_local --input g1_multicast

--phrases : pour chaque phrase, Entrée pour démarrer, lire la phrase à voix haute,
Entrée pour arrêter. Produit NNN.wav (16 kHz mono) + NNN.txt (référence).
--protocole : même principe, fichiers nommés d'après l'identifiant (p01.wav…) ; les
lignes « bruit » s'enregistrent pendant --duree-bruit secondes, sans parler. Les
fichiers déjà présents sont sautés, ce qui permet de reprendre une séance.
Utiliser --input g1_multicast pour enregistrer avec les micros du robot.
"""

from __future__ import annotations

import argparse
import threading
import time
import wave
from pathlib import Path

import numpy as np

import _bootstrap  # noqa: F401
from jone.audio_input import create_mic
from jone.config import load_config
from jone.evaluation import load_protocol


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


def record_for(mic, seconds: float) -> np.ndarray:
    frames = []
    mic.flush()
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        f = mic.read(timeout=0.2)
        if f is not None:
            frames.append(f)
    return np.concatenate(frames) if frames else np.zeros(0, dtype=np.int16)


def save_wav(path: Path, audio: np.ndarray, rate: int) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(audio.astype("<i2").tobytes())
    print(f"  -> {path} ({len(audio) / rate:.1f} s)")


def record_phrases(mic, phrases: list[str], out: Path, rate: int) -> None:
    for i, phrase in enumerate(phrases, 1):
        input(f"
[{i}/{len(phrases)}] « {phrase} »
  Entrée pour démarrer ")
        stem = out / f"{i:03d}"
        save_wav(stem.with_suffix(".wav"), record_until_enter(mic), rate)
        stem.with_suffix(".txt").write_text(phrase + "
", encoding="utf-8")


def record_protocol(mic, items, out: Path, rate: int, noise_s: float) -> None:
    for i, item in enumerate(items, 1):
        path = out / f"{item.id}.wav"
        if path.exists():
            continue
        head = f"
[{i}/{len(items)}] {item.id} ({item.type}, {item.langue})"
        if item.type == "bruit":
            input(f"{head}
  {item.texte}
  Entrée pour démarrer ({noise_s:.0f} s, sans parler) ")
            audio = record_for(mic, noise_s)
        else:
            input(f"{head} « {item.texte} »
  Entrée pour démarrer ")
            audio = record_until_enter(mic)
        save_wav(path, audio, rate)


def main():
    p = argparse.ArgumentParser()
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--phrases", help="une phrase par ligne (corpus de WER)")
    src.add_argument("--protocole", help="protocole d'évaluation (.tsv)")
    p.add_argument("--out", required=True)
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--input", choices=["sounddevice", "g1_multicast"], default="sounddevice")
    p.add_argument("--duree-bruit", type=float, default=6.0, help="durée des enregistrements de bruit (s)")
    args = p.parse_args()

    cfg = load_config(args.config)
    cfg["audio_input"]["backend"] = args.input
    rate = cfg["audio_input"]["sample_rate"]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    mic = create_mic(cfg)
    mic.start()
    try:
        if args.protocole:
            record_protocol(mic, load_protocol(Path(args.protocole)), out, rate, args.duree_bruit)
        else:
            phrases = [l.strip() for l in Path(args.phrases).read_text(encoding="utf-8").splitlines()
                       if l.strip()]
            record_phrases(mic, phrases, out, rate)
    finally:
        mic.close()


if __name__ == "__main__":
    main()
