"""Phase 2 — mesure de Piper seul : temps de synthèse et facteur temps réel.

    python tools/bench_tts.py [--play] [--save logs/tts]

--play joue les phrases sur la sortie configurée (G1 ou PC) : sert aussi au test
d'intelligibilité ENF-05 (faire écouter à 5 testeurs, noter dans le rapport).
"""

from __future__ import annotations

import argparse
import time
import wave
from pathlib import Path

import _bootstrap  # noqa: F401
from jone.audio_utils import TARGET_RATE, pcm_duration_s
from jone.config import load_config
from jone.metrics import summarize
from jone.tts import PiperTTS

SENTENCES = [
    "Bonjour, je m'appelle J-ONE et je suis ravi de vous rencontrer.",
    "Il fait environ vingt degrés dans cette salle, c'est parfait pour discuter.",
    "Je suis un robot humanoïde, mais je ne peux pas me déplacer pendant notre conversation.",
    "Voulez-vous que je vous explique comment je comprends ce que vous dites ?",
    "Votre voix est captée par mes micros, transcrite en texte, puis je réfléchis à une réponse.",
    "D'accord.",
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--play", action="store_true")
    p.add_argument("--save")
    args = p.parse_args()

    cfg = load_config(args.config)
    tts = PiperTTS.from_config(cfg)
    tts.synthesize("Préchauffage.")
    speaker = None
    if args.play:
        from jone.audio_output import create_speaker

        speaker = create_speaker(cfg)

    times, rtfs = [], []
    for i, s in enumerate(SENTENCES):
        t0 = time.perf_counter()
        pcm = tts.synthesize(s)
        dt = time.perf_counter() - t0
        dur = pcm_duration_s(pcm)
        times.append(dt * 1000)
        rtfs.append(dt / dur)
        print(f"{dt * 1000:6.0f} ms pour {dur:4.1f} s d'audio (RTF {dt / dur:.2f}) : {s}")
        if args.save:
            out = Path(args.save)
            out.mkdir(parents=True, exist_ok=True)
            with wave.open(str(out / f"tts_{i:02d}.wav"), "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(TARGET_RATE)
                w.writeframes(pcm)
        if speaker:
            speaker.begin_response()
            speaker.enqueue(pcm)
            speaker.wait_until_done()

    print("\nLatence de synthèse (ms) :", summarize(times))
    print("Facteur temps réel        :", summarize(rtfs), "(< 1 = plus rapide que le temps réel)")
    if speaker:
        speaker.close()


if __name__ == "__main__":
    main()
