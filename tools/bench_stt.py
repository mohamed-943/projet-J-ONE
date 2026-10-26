"""Phase 2 — mesure de Whisper seul sur la Jetson : latence, mémoire, WER (ENF-02).

    python tools/bench_stt.py --corpus corpus/calme [--model large-v3-turbo] [--beam 1]

Le corpus contient des paires NNN.wav / NNN.txt (voir tools/enregistrer_corpus.py).
Résultat détaillé enregistré dans logs/bench_stt-<date>.json.
"""

from __future__ import annotations

import argparse
import json
import time
import wave
from datetime import datetime
from pathlib import Path

import numpy as np

import _bootstrap  # noqa: F401
from jone.audio_utils import to_pcm16k_mono
from jone.config import load_config, resolve_path
from jone.metrics import corpus_wer, summarize, wer
from jone.stt import WhisperSTT


def load_wav_16k(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as w:
        frames = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
        pcm = to_pcm16k_mono(frames, w.getframerate(), w.getnchannels())
    return np.frombuffer(pcm, dtype="<i2")


def memory_report() -> dict:
    report = {}
    try:
        import resource

        report["rss_max_mo"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    except ImportError:
        pass
    try:
        meminfo = Path("/proc/meminfo").read_text().splitlines()
        values = {l.split(":")[0]: int(l.split()[1]) for l in meminfo}
        # Jetson : mémoire unifiée CPU/GPU, c'est donc la mémoire système qui compte.
        report["systeme_utilisee_mo"] = (values["MemTotal"] - values["MemAvailable"]) / 1024
    except (OSError, KeyError, ValueError):
        pass
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", required=True)
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--model")
    p.add_argument("--beam", type=int)
    p.add_argument("--device")
    args = p.parse_args()

    cfg = load_config(args.config)
    stt_cfg = cfg["stt"]
    for key, val in (("model", args.model), ("beam_size", args.beam), ("device", args.device)):
        if val is not None:
            stt_cfg[key] = val

    files = sorted(Path(args.corpus).glob("*.wav"))
    if not files:
        raise SystemExit(f"Aucun .wav dans {args.corpus}")

    mem_before = memory_report()
    t0 = time.perf_counter()
    stt = WhisperSTT(**stt_cfg)
    load_s = time.perf_counter() - t0
    stt.transcribe(np.zeros(16000, dtype=np.int16))  # préchauffage

    rows, pairs = [], []
    for wav in files:
        audio = load_wav_16k(wav)
        ref = wav.with_suffix(".txt").read_text(encoding="utf-8").strip()
        tr = stt.transcribe(audio)
        pairs.append((ref, tr.text))
        rows.append({"fichier": wav.name, "reference": ref, "hypothese": tr.text,
                     "wer": round(wer(ref, tr.text), 3), "audio_s": round(tr.audio_s, 2),
                     "latence_ms": round(tr.elapsed_ms, 1),
                     "rtf": round(tr.elapsed_ms / 1000 / max(tr.audio_s, 1e-6), 3)})
        print(f"{wav.name}: WER {rows[-1]['wer']:.0%}  {tr.elapsed_ms:.0f} ms  « {tr.text} »")

    result = {
        "config": stt_cfg,
        "chargement_s": round(load_s, 1),
        "memoire_avant": mem_before,
        "memoire_apres": memory_report(),
        "wer_global": round(corpus_wer(pairs), 4),
        "latence_ms": summarize([r["latence_ms"] for r in rows]),
        "rtf": summarize([r["rtf"] for r in rows]),
        "details": rows,
    }
    out = resolve_path(cfg["logging"]["dir"]) / f"bench_stt-{datetime.now():%Y%m%d-%H%M%S}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    verdict = "CONFORME" if result["wer_global"] <= 0.10 else "NON CONFORME"
    print(f"\nWER global : {result['wer_global']:.1%} (cible ≤ 10 %, {verdict})")
    print(f"Latence p50 {result['latence_ms']['p50']:.0f} ms, p90 {result['latence_ms']['p90']:.0f} ms")
    print(f"Mémoire : {result['memoire_apres']}")
    print(f"Détails : {out}")


if __name__ == "__main__":
    main()
