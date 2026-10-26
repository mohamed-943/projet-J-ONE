"""LLM principal : Claude Haiku 5.5 via l'API Anthropic, en streaming.

Le délai de lecture du client HTTP sert de seuil de bascule (ENF-04) : si aucun
fragment n'arrive dans `first_token_timeout_s` (premier jeton ou blocage en cours
de flux), le SDK lève APITimeoutError et le routeur passe au LLM local. Pas de
nouvelle tentative automatique : en conversation, mieux vaut basculer vite.
"""

from __future__ import annotations

import logging
import os
from typing import Iterator

from .base import LLMError, LLMRefusal

log = logging.getLogger(__name__)


class ClaudeBackend:
    name = "claude"

    def __init__(self, model="claude-haiku-5-5", max_tokens=300, effort="low",
                 disable_thinking=True, connect_timeout_s=2.0, first_token_timeout_s=3.0,
                 **_ignored):
        import anthropic

        self._anthropic = anthropic
        self.model = model
        self.max_tokens = max_tokens
        self.effort = effort
        self.disable_thinking = disable_thinking
        self.client = anthropic.Anthropic(
            timeout=anthropic.Timeout(first_token_timeout_s, connect=connect_timeout_s),
            max_retries=0,
        )

    @staticmethod
    def credentials_available() -> bool:
        return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))

    def _request(self, system: str, messages: list[dict[str, str]]) -> dict:
        kwargs = dict(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=messages,
            output_config={"effort": self.effort},
        )
        if self.disable_thinking:
            # Réponses courtes et conversationnelles : pas besoin de réflexion, et
            # chaque jeton de réflexion retarde la première phrase prononcée.
            kwargs["thinking"] = {"type": "disabled"}
        return kwargs

    def stream(self, system: str, messages: list[dict[str, str]]) -> Iterator[str]:
        anthropic = self._anthropic
        try:
            with self.client.messages.stream(**self._request(system, messages)) as stream:
                yield from stream.text_stream
                final = stream.get_final_message()
        except anthropic.APITimeoutError as exc:
            raise LLMError("délai dépassé") from exc
        except anthropic.APIConnectionError as exc:
            raise LLMError("connexion impossible") from exc
        except anthropic.AuthenticationError as exc:
            raise LLMError("clé API invalide") from exc
        except anthropic.RateLimitError as exc:
            raise LLMError("limite de débit atteinte") from exc
        except anthropic.APIStatusError as exc:
            raise LLMError(f"erreur API {exc.status_code}: {exc.message}") from exc
        if final.stop_reason == "refusal":
            raise LLMRefusal("réponse refusée par le modèle")

    def warmup(self) -> None:
        pass
