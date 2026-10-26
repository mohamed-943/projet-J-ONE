from __future__ import annotations

from typing import Iterator, Protocol


class LLMError(Exception):
    """Échec d'un backend (réseau, délai, serveur) : déclenche la bascule."""


class LLMRefusal(LLMError):
    """Le modèle a refusé de répondre (stop_reason « refusal »)."""


class LLMBackend(Protocol):
    name: str

    def stream(self, system: str, messages: list[dict[str, str]]) -> Iterator[str]:
        """Produit la réponse fragment par fragment ; lève LLMError en cas d'échec."""
        ...

    def warmup(self) -> None:
        ...
