"""LLM de secours : Qwen3 4B servi localement par Ollama sur la Jetson (ENF-03)."""

from __future__ import annotations

import logging
from typing import Iterator

from ..text_utils import ThinkStripper
from .base import LLMError

log = logging.getLogger(__name__)


def _field(obj, key):
    """Les réponses d'ollama-python sont des objets, des dicts selon la version."""
    return obj.get(key) if isinstance(obj, dict) else getattr(obj, key, None)


class OllamaBackend:
    name = "qwen3-local"

    def __init__(self, host="http://127.0.0.1:11434", model="qwen3:4b", max_tokens=200,
                 temperature=0.7, keep_alive="30m", timeout_s=20.0, **_ignored):
        import ollama

        self.client = ollama.Client(host=host, timeout=timeout_s)
        self.model = model
        self.keep_alive = keep_alive
        self.options = {"num_predict": max_tokens, "temperature": temperature}

    def stream(self, system: str, messages: list[dict[str, str]]) -> Iterator[str]:
        strip = ThinkStripper()
        try:
            chunks = self.client.chat(
                model=self.model,
                messages=[{"role": "system", "content": system}] + messages,
                stream=True,
                think=False,  # mode sans réflexion de Qwen3 : latence minimale
                options=self.options,
                keep_alive=self.keep_alive,
            )
            for chunk in chunks:
                content = _field(_field(chunk, "message"), "content") or ""
                text = strip.feed(content)
                if text:
                    yield text
        except Exception as exc:  # ResponseError, httpx, ConnectionError…
            raise LLMError(f"Ollama indisponible: {exc}") from exc

    def warmup(self) -> None:
        """Charge le modèle en mémoire GPU au démarrage pour éviter ~10 s au 1er appel."""
        try:
            self.client.generate(model=self.model, prompt="", keep_alive=self.keep_alive)
            log.info("Modèle local %s chargé", self.model)
        except Exception as exc:
            log.warning("Préchargement de %s impossible : %s", self.model, exc)
