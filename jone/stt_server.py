"""Transcription sur le GPU de la Jetson avec whisper.cpp (EF-02).

faster-whisper n'a pas de CUDA sous JetPack 5 : Whisper y tourne sur CPU (4 s par
énoncé avec le modèle small). whisper.cpp, compilé sur la Jetson avec CUDA, fournit
`whisper-server`, qui garde le modèle chargé sur le GPU et transcrit un fichier WAV
envoyé en HTTP. Si le serveur ne répond pas au démarrage, il est lancé ici et arrêté
avec le pipeline.

Vitesse : l'encodeur de Whisper analyse toujours une fenêtre de 30 s. `audio_ctx`
la réduit à la durée de l'énoncé plus une marge. Mesuré sur la Jetson Orin avec
large-v3-turbo, 48 phrases du protocole d'évaluation (tests/protocole_evaluation.tsv) :

    fenêtre de 30 s          WER 3,6 %, 2,3 s par phrase
    min 384, marge 2 s       WER 60 %, 0,6 s : phrases répétées ou déformées
    min 512, marge 3 s       WER 2,6 %, 0,8 s  ← valeurs par défaut

Langue : une requête en détection automatique renvoie aussi les probabilités de
chaque langue ; si la langue retenue parmi `languages` n'est pas celle que Whisper a
transcrite, une seconde requête force la bonne (quelques centaines de ms sur GPU).
"""

from __future__ import annotations

import atexit
import io
import logging
import math
import subprocess
import time
import wave
import zlib
from pathlib import Path
from urllib.parse import urlparse

import numpy as np

from .stt import Transcription, choose_language

log = logging.getLogger(__name__)


def _wav_bytes(samples: np.ndarray, sample_rate: int) -> bytes:
    pcm = samples if samples.dtype == np.int16 else np.clip(samples * 32768.0, -32768, 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


# whisper-server renvoie la langue transcrite en toutes lettres.
LANGUAGE_CODES = {"french": "fr", "english": "en", "spanish": "es", "german": "de",
                  "italian": "it", "portuguese": "pt", "dutch": "nl", "arabic": "ar"}
FULL_CTX = 1500  # 30 s d'audio, 50 positions par seconde


def audio_ctx_for(audio_s: float, margin_s: float, minimum: int) -> int:
    """Fenêtre de l'encodeur pour `audio_s` secondes, arrondie au multiple de 64."""
    ctx = math.ceil((audio_s + margin_s) * 50 / 64) * 64
    return min(FULL_CTX, max(minimum, ctx))


def _compression_ratio(text: str) -> float:
    """Même mesure que Whisper : un texte qui se répète se compresse beaucoup."""
    data = text.encode("utf-8")
    return len(data) / len(zlib.compress(data)) if data else 1.0


class WhisperServerSTT:
    def __init__(self, url="http://127.0.0.1:8080", binary=None, model=None, threads=4,
                 language=None, languages=("fr", "en"), default_language="fr",
                 language_margin=0.65, audio_ctx_margin_s=3.0, min_audio_ctx=512,
                 startup_timeout_s=60.0, request_timeout_s=30.0, log_dir="logs"):
        import httpx

        self.url = url.rstrip("/")
        self.language = language
        self.languages = list(languages or ([language] if language else []))
        self.default_language = default_language
        self.language_margin = language_margin
        self.audio_ctx_margin_s = audio_ctx_margin_s
        self.min_audio_ctx = min_audio_ctx  # 0 : fenêtre complète de 30 s
        self.client = httpx.Client(timeout=request_timeout_s)
        self._process = None
        if not self._alive():
            if not (binary and model):
                raise RuntimeError(f"whisper-server injoignable sur {self.url} "
                                   "(stt.server.binary et stt.server.model non renseignés)")
            self._start(binary, model, threads, log_dir, startup_timeout_s)

    @classmethod
    def from_config(cls, cfg: dict) -> "WhisperServerSTT":
        from .config import resolve_path

        s, server = cfg["stt"], cfg["stt"]["server"]
        return cls(
            url=server["url"],
            binary=server.get("binary") and str(Path(server["binary"]).expanduser()),
            model=server.get("model") and str(Path(server["model"]).expanduser()),
            threads=server.get("threads", 4),
            language=s.get("language"),
            languages=s.get("languages", ("fr", "en")),
            default_language=s.get("default_language", "fr"),
            language_margin=s.get("language_margin", 0.65),
            audio_ctx_margin_s=server.get("audio_ctx_margin_s", 3.0),
            min_audio_ctx=server.get("min_audio_ctx", 512),
            startup_timeout_s=server.get("startup_timeout_s", 60.0),
            log_dir=resolve_path(cfg["logging"]["dir"]),
        )

    # --- serveur ------------------------------------------------------------
    def _alive(self) -> bool:
        try:
            self.client.get(self.url + "/", timeout=1.0)
            return True
        except Exception:
            return False

    def _start(self, binary, model, threads, log_dir, timeout_s) -> None:
        port = str(urlparse(self.url).port or 8080)
        host = urlparse(self.url).hostname or "127.0.0.1"
        cmd = [binary, "-m", model, "--host", host, "--port", port, "-t", str(threads),
               "-l", "auto", "-nt",
               # pas de nouveaux décodages à température croissante (voir stt.py)
               "-nf"]
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        out = open(Path(log_dir) / "whisper-server.log", "ab")
        log.info("Lancement de whisper-server : %s", " ".join(cmd))
        self._process = subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT)
        atexit.register(self.close)
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < timeout_s:
            if self._process.poll() is not None:
                raise RuntimeError("whisper-server s'est arrêté au démarrage "
                                   f"(code {self._process.returncode}, voir whisper-server.log)")
            if self._alive():
                log.info("whisper-server prêt en %.1f s", time.perf_counter() - t0)
                return
            time.sleep(0.3)
        self.close()
        raise RuntimeError(f"whisper-server pas prêt après {timeout_s:.0f} s")

    def close(self) -> None:
        if self._process and self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=3)
            except (subprocess.TimeoutExpired, KeyboardInterrupt):  # 2e Ctrl+C : on force
                self._process.kill()

    # --- transcription ------------------------------------------------------
    def _request(self, wav: bytes, language: str, with_probs: bool, audio_s: float = 30.0) -> dict:
        data = {
            "language": language,
            "response_format": "verbose_json",
            "temperature": "0.0",
            "temperature_inc": "0.0",
            "no_timestamps": "true",
            "no_language_probabilities": "false" if with_probs else "true",
        }
        if self.min_audio_ctx:
            data["audio_ctx"] = str(audio_ctx_for(audio_s, self.audio_ctx_margin_s, self.min_audio_ctx))
        resp = self.client.post(self.url + "/inference", data=data,
                                files={"file": ("audio.wav", wav, "audio/wav")})
        resp.raise_for_status()
        return resp.json()

    def warmup(self) -> None:
        t0 = time.perf_counter()
        self._request(_wav_bytes(np.zeros(16000, dtype=np.int16), 16000),
                      self.language or self.default_language, with_probs=False, audio_s=1.0)
        log.info("Whisper (GPU) préchauffé en %.1f s", time.perf_counter() - t0)

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> Transcription:
        t0 = time.perf_counter()
        wav = _wav_bytes(audio, sample_rate)
        audio_s = len(audio) / sample_rate
        if self.language:
            language, language_prob = self.language, 1.0
            res = self._request(wav, language, with_probs=False, audio_s=audio_s)
        else:
            res = self._request(wav, "auto", with_probs=True, audio_s=audio_s)
            probs = res.get("language_probabilities") or {}
            language, language_prob = choose_language(probs, self.languages,
                                                      self.default_language, self.language_margin)
            # La langue que Whisper a réellement transcrite ; les probabilités viennent
            # d'une détection séparée et peuvent désigner une autre langue.
            if res.get("language"):
                transcribed = LANGUAGE_CODES.get(str(res["language"]).lower())
            else:
                transcribed = max(probs, key=probs.get) if probs else None
            if language != transcribed:
                log.info("Langue détectée %s, retranscription en %s", transcribed, language)
                res = self._request(wav, language, with_probs=False, audio_s=audio_s)

        elapsed = (time.perf_counter() - t0) * 1000
        segments = [s for s in res.get("segments", []) if s.get("text", "").strip()]
        text = " ".join(s["text"].strip() for s in segments).strip()
        log.info("Transcription GPU en %.0f ms (langue %s, %.0f %%)", elapsed, language, language_prob * 100)
        if not segments:
            return Transcription("", no_speech_prob=1.0, audio_s=audio_s, elapsed_ms=elapsed,
                                 language=language, language_prob=language_prob)
        return Transcription(
            text=text,
            avg_logprob=float(np.mean([s.get("avg_logprob", 0.0) for s in segments])),
            no_speech_prob=float(max(s.get("no_speech_prob", 0.0) for s in segments)),
            compression_ratio=_compression_ratio(text),
            audio_s=audio_s,
            elapsed_ms=elapsed,
            language=language,
            language_prob=language_prob,
        )
