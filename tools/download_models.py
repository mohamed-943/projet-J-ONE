"""Télécharge les voix Piper (français fr_FR-siwis, anglais en_US-lessac) et
rappelle les autres modèles à installer.

    python tools/download_models.py [--quality medium|low] [--langs fr en]

Whisper est téléchargé automatiquement par faster-whisper au premier lancement
(cache Hugging Face). Le LLM local s'installe avec Ollama :
    ollama pull qwen3:4b
"""

from __future__ import annotations

import argparse
import urllib.request

import _bootstrap  # noqa: F401
from _bootstrap import ROOT

BASE = "https://huggingface.co/rhasspy/piper-voices/resolve/main"
VOICES = {  # langue : (dossier sur Hugging Face, nom de la voix)
    "fr": ("fr/fr_FR/siwis", "fr_FR-siwis"),
    "en": ("en/en_US/lessac", "en_US-lessac"),
}


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
    p.add_argument("--langs", nargs="+", default=list(VOICES), choices=list(VOICES))
    args = p.parse_args()

    voices = ROOT / "voices"
    voices.mkdir(exist_ok=True)
    for lang in args.langs:
        folder, voice = VOICES[lang]
        name = f"{voice}-{args.quality}.onnx"
        url = f"{BASE}/{folder}/{args.quality}/{name}"
        download(url, voices / name)
        download(url + ".json", voices / (name + ".json"))

    print("\nÀ faire aussi sur la Jetson :")
    print("  ollama pull qwen3:4b      # LLM de secours")
    print("  (Whisper se télécharge seul au premier lancement)")


if __name__ == "__main__":
    main()
