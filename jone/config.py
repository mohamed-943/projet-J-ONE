"""Chargement de la configuration (config.yaml + variables d'environnement .env).

Toutes les valeurs par défaut sont définies ici ; config.yaml ne surcharge que ce
qu'il précise. Ainsi une clé oubliée dans le fichier ne fait jamais planter le
pipeline (ENF-08).
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULTS: dict[str, Any] = {
    "audio_input": {
        "backend": "sounddevice",  # sounddevice (PC de dev) | g1_multicast (robot)
        "sample_rate": 16000,
        "frame_samples": 512,  # 32 ms à 16 kHz, taille attendue par Silero VAD
        "device": None,
        "g1_multicast_group": "239.168.123.161",
        "g1_multicast_port": 5555,
        "g1_local_ip": "192.168.123.164",
        "queue_max_frames": 500,
    },
    "vad": {
        "threshold": 0.5,
        "neg_threshold": 0.35,
        "min_speech_ms": 250,
        "min_silence_ms": 600,
        "pre_roll_ms": 300,
        "max_utterance_s": 15.0,
    },
    "stt": {
        "model": "large-v3-turbo",
        "device": "cuda",
        "compute_type": "int8_float16",
        "beam_size": 1,
        "language": "fr",
    },
    "filter": {
        "min_chars": 2,
        "max_no_speech_prob": 0.6,
        "min_avg_logprob": -1.0,
        "max_compression_ratio": 2.4,
        "blacklist": [],
    },
    "wake_word": {
        "enabled": False,
        "words": ["j-one", "j one", "jone", "djiwan", "ji wan", "gi one"],
        "session_timeout_s": 30.0,
    },
    "llm": {
        "system_prompt_file": "prompts/system_prompt_fr.txt",
        "history_max_turns": 10,
        "max_sentences": 3,
        "cloud": {
            "enabled": True,
            "provider": "claude",  # claude | groq (essais sans clé Anthropic)
            "model": "claude-haiku-5-5",
            "max_tokens": 300,
            "effort": "low",
            "disable_thinking": True,
            "connect_timeout_s": 2.0,
            "first_token_timeout_s": 3.0,
            "retry_cooldown_s": 30.0,
            "groq": {
                "model": "qwen/qwen3.8-27b",
                "reasoning_effort": "none",
            },
        },
        "local": {
            "enabled": True,
            "host": "http://127.0.0.1:11434",
            "model": "qwen3:4b",
            "max_tokens": 200,
            "temperature": 0.7,
            "keep_alive": "30m",
            "timeout_s": 20.0,
        },
    },
    "tts": {
        "voice_model": "voices/fr_FR-siwis-medium.onnx",
        "length_scale": 1.0,
        "use_cuda": False,
    },
    "audio_output": {
        "backend": "g1",  # g1 | sounddevice | none
        "volume": 80,
        "app_name": "jone",
        "chunk_seconds": 1.0,
        "lead_seconds": 0.5,
        "tail_margin_s": 0.4,
    },
    "robot": {
        "network_interface": "eth0",
        "sdk_timeout_s": 10.0,
    },
    "led": {
        "backend": "g1",  # g1 | console | none
        "colors": {
            "idle": [0, 0, 0],
            "listening": [0, 255, 0],
            "thinking": [0, 0, 255],
            "speaking": [255, 255, 255],
            "error": [255, 0, 0],
        },
        "error_hold_s": 2.0,
    },
    "logging": {
        "dir": "logs",
        "level": "INFO",
    },
}


def _deep_merge(base: dict, override: dict) -> dict:
    result = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def resolve_path(path: str | os.PathLike) -> Path:
    """Chemins relatifs du fichier de config = relatifs à la racine du projet."""
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def load_config(path: str | os.PathLike | None = None) -> dict[str, Any]:
    try:
        from dotenv import load_dotenv

        load_dotenv(PROJECT_ROOT / ".env")
    except ImportError:
        pass

    config_path = resolve_path(path or "config.yaml")
    user_cfg: dict = {}
    if config_path.exists():
        import yaml

        with open(config_path, encoding="utf-8") as f:
            user_cfg = yaml.safe_load(f) or {}
    return _deep_merge(DEFAULTS, user_cfg)
