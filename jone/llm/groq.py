"""LLM cloud de test : un modèle servi par Groq (API compatible OpenAI), en streaming.

Sert à essayer le pipeline avec un LLM cloud quand aucune clé Anthropic n'est
disponible (llm.cloud.provider: groq). Même contrat que ClaudeBackend : le délai
de lecture HTTP sert de seuil de bascule (ENF-04), pas de nouvelle tentative, et
toute erreur devient une LLMError pour que le routeur passe au LLM local.
Seul le texte transcrit est envoyé (ENF-07).
"""

from __future__ import annotations

import json
import logging
import os
from typing import Iterator

from .base import LLMError

log = logging.getLogger(__name__)

API_URL = "https://api.groq.com/openai/v1/chat/completions"


class GroqBackend:
    name = "groq"

    def __init__(self, model="qwen/qwen3.8-27b", max_tokens=300, reasoning_effort="none",
                 connect_timeout_s=2.0, first_token_timeout_s=3.0, **_ignored):
        import httpx  # installé avec ollama

        self._httpx = httpx
        self.model = model
        self.max_tokens = max_tokens
        # "none" coupe la réflexion des modèles Qwen3 : première phrase plus rapide.
        self.reasoning_effort = reasoning_effort
        self.client = httpx.Client(
            timeout=httpx.Timeout(first_token_timeout_s, connect=connect_timeout_s),
            headers={"Authorization": f"Bearer {os.environ.get('GROQ_API_KEY', '')}"},
        )

    @staticmethod
    def credentials_available() -> bool:
        return bool(os.environ.get("GROQ_API_KEY"))

    def _request(self, system: str, messages: list[dict[str, str]]) -> dict:
        body = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "stream": True,
            "messages": [{"role": "system", "content": system}] + messages,
        }
        if self.reasoning_effort:
            body["reasoning_effort"] = self.reasoning_effort
        return body

    def stream(self, system: str, messages: list[dict[str, str]]) -> Iterator[str]:
        httpx = self._httpx
        try:
            with self.client.stream("POST", API_URL, json=self._request(system, messages)) as resp:
                if resp.status_code != 200:
                    detail = resp.read().decode("utf-8", "replace")[:200]
                    if resp.status_code == 401:
                        raise LLMError("clé API invalide")
                    if resp.status_code == 429:
                        raise LLMError("limite de débit atteinte")
                    raise LLMError(f"erreur API {resp.status_code}: {detail}")
                for line in resp.iter_lines():
                    if not line.startswith("data: ") or line == "data: [DONE]":
                        continue
                    event = json.loads(line[6:])
                    if "error" in event:
                        raise LLMError(f"erreur en cours de flux : {event['error']}")
                    choices = event.get("choices") or []
                    text = choices[0].get("delta", {}).get("content") if choices else None
                    if text:
                        yield text
        except httpx.TimeoutException as exc:
            raise LLMError("délai dépassé") from exc
        except httpx.HTTPError as exc:
            raise LLMError("connexion impossible") from exc
        except (json.JSONDecodeError, KeyError, IndexError) as exc:
            raise LLMError(f"réponse illisible : {exc}") from exc

    def warmup(self) -> None:
        pass
