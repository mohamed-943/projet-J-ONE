"""Phase 1 — prise en main du SDK : vérifie PlayStream() et LedControl() sur le G1.

À lancer sur la Jetson, robot suspendu sur son cadre en mode développeur (L2+R2).

    python tools/test_g1_audio.py [--iface eth0] [--wav fichier.wav] [--say "Bonjour"]

1. fait défiler les quatre couleurs d'état de la LED (EF-11) ;
2. joue un bip de 440 Hz puis, au choix, un fichier WAV ou une phrase synthétisée
   par Piper, convertis en PCM 16 kHz mono 16 bits (EF-09) ;
3. écoute 3 s le micro multicast et affiche le niveau reçu (point à vérifier :
   groupe/port du réseau de micros).
"""

from __future__ import annotations

import argparse
import time
import wave

import numpy as np

import _bootstrap  # noqa: F401
from jone.audio_utils import TARGET_RATE, pcm_duration_s, to_pcm16k_mono
from jone.config import DEFAULTS, load_config


def tone(freq=440.0, seconds=1.0) -> bytes:
    t = np.arange(int(TARGET_RATE * seconds)) / TARGET_RATE
    return to_pcm16k_mono(0.3 * np.sin(2 * np.pi * freq * t), TARGET_RATE)


def read_wav(path: str) -> bytes:
    with wave.open(path, "rb") as w:
        if w.getsampwidth() != 2:
            raise SystemExit("WAV 16 bits attendu")
        frames = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
        return to_pcm16k_mono(frames, w.getframerate(), w.getnchannels())


def play(client, pcm: bytes, label: str):
    stream_id = str(int(time.time() * 1000))
    chunk = TARGET_RATE * 2  # 1 s
    print(f"PlayStream : {label} ({pcm_duration_s(pcm):.1f} s)")
    t0 = time.monotonic()
    for off in range(0, len(pcm), chunk):
        ret = client.PlayStream("jone_test", stream_id, pcm[off : off + chunk])
        print(f"  bloc {off // chunk}: code {ret}")
        time.sleep(max(0.0, (off + chunk) / 2 / TARGET_RATE - 0.5 - (time.monotonic() - t0)))
    time.sleep(max(0.0, pcm_duration_s(pcm) - (time.monotonic() - t0)) + 0.3)
    client.PlayStop("jone_test")


def check_mic(cfg, seconds=3.0):
    from jone.audio_input import create_mic

    cfg["audio_input"]["backend"] = "g1_multicast"
    mic = create_mic(cfg)
    mic.start()
    print(f"Écoute du micro pendant {seconds:.0f} s, parlez…")
    frames, t_end = [], time.monotonic() + seconds
    while time.monotonic() < t_end:
        f = mic.read(timeout=0.5)
        if f is not None:
            frames.append(f)
    mic.close()
    if not frames:
        print("  AUCUNE trame reçue : vérifier groupe/port multicast et g1_local_ip.")
        return
    audio = np.concatenate(frames).astype(np.float32) / 32768
    rms = float(np.sqrt(np.mean(audio**2)))
    print(f"  {len(frames)} trames reçues ({len(audio) / TARGET_RATE:.1f} s), niveau RMS {rms:.4f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--iface")
    p.add_argument("--wav")
    p.add_argument("--say")
    p.add_argument("--volume", type=int, default=70)
    p.add_argument("--skip-mic", action="store_true")
    args = p.parse_args()

    cfg = load_config(args.config)
    if args.iface:
        cfg["robot"]["network_interface"] = args.iface

    from jone.robot import get_audio_client

    client = get_audio_client(cfg["robot"])
    print("Volume actuel :", client.GetVolume())
    print("SetVolume :", client.SetVolume(args.volume))

    for state, rgb in DEFAULTS["led"]["colors"].items():
        if state == "idle":
            continue
        print(f"LedControl {state} {rgb} : code {client.LedControl(*rgb)}")
        time.sleep(1.5)
    client.LedControl(0, 0, 0)

    play(client, tone(), "bip 440 Hz")
    if args.wav:
        play(client, read_wav(args.wav), args.wav)
    if args.say:
        from jone.tts import PiperTTS

        play(client, PiperTTS.from_config(cfg).synthesize(args.say), f"Piper « {args.say} »")

    if not args.skip_mic:
        check_mic(cfg)


if __name__ == "__main__":
    main()
