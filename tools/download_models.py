"""Télécharge la voix Piper fr_FR-siwis et rappelle les autres modèles à installer.

    python tools/download_models.py [--quality medium|low]

Whisper large-v3-turbo est téléchargé automatiquement par faster-whisper au
premier lancement (cache Hugging Face). Le LLM local s'installe avec Ollama :
    ollama pull qwen3:4b
"""

from __future__ import annotations

import argparse
import urllib.request

import _bootstrap  # noqa: F401
from _bootstrap import ROOT

BASE = "https://huggingface.co/rhasspy/piper-voices/resolve/main/fr/fr_FR/siwis/{q}/fr_FR-siwis-{q}.onnx"


def download(url: str, dest) -> None:
    if dest.exists():
        print(f"déjà présent : {dest.relative_to(ROOT)}")
        return
    print(f"téléchargement : {url}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(dest)
    print(f"  -> {dest.relative_to(ROOT)} ({dest.stat().st_size / 1e6:.1f} Mo)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--quality", default="medium", choices=["low", "medium"])
    args = p.parse_args()

    voices = ROOT / "voices"
    voices.mkdir(exist_ok=True)
    url = BASE.format(q=args.quality)
    name = f"fr_FR-siwis-{args.quality}.onnx"
    download(url, voices / name)
    download(url + ".json", voices / (name + ".json"))

    print("\nÀ faire aussi sur la Jetson :")
    print("  ollama pull qwen3:4b      # LLM de secours")
    print("  (Whisper large-v3-turbo se télécharge seul au premier lancement)")


if __name__ == "__main__":
    main()
