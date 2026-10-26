"""Routeur LLM : Claude en priorité, Qwen3 local en secours (EF-07).

- échec du cloud avant le premier fragment : bascule immédiate vers le local ;
- échec en cours de réponse : on garde ce qui a déjà été dit, sans repartir de zéro
  sur l'autre modèle (le robot se répéterait) ;
- après un échec, le cloud est mis de côté `retry_cooldown_s` secondes pour ne pas
  payer le délai de bascule à chaque tour quand le Wi-Fi est coupé, puis retenté.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Iterator

from .base import LLMBackend, LLMError, LLMRefusal

log = logging.getLogger(__name__)

REFUSAL_REPLY = "Je préfère ne pas répondre à cette question. Parlons d'autre chose ?"


class NoBackendAvailable(LLMError):
    pass


@dataclass
class RouteInfo:
    """Rempli pendant le flux : quel modèle a répondu, et les échecs rencontrés."""

    backend: str | None = None
    fallback: bool = False
    errors: list[str] = field(default_factory=list)
    first_token_s: float | None = None


class LLMRouter:
    def __init__(self, cloud: LLMBackend | None, local: LLMBackend | None,
                 retry_cooldown_s: float = 30.0, clock: Callable[[], float] = time.monotonic):
        self.cloud = cloud
        self.local = local
        self.retry_cooldown_s = retry_cooldown_s
        self._clock = clock
        self._cloud_down_until = float("-inf")

    @property
    def cloud_available(self) -> bool:
        return self.cloud is not None and self._clock() >= self._cloud_down_until

    def _candidates(self) -> list[LLMBackend]:
        order = []
        if self.cloud_available:
            order.append(self.cloud)
        if self.local is not None:
            order.append(self.local)
        return order

    def stream(self, system: str, messages: list[dict[str, str]], info: RouteInfo) -> Iterator[str]:
        candidates = self._candidates()
        if not candidates:
            raise NoBackendAvailable("aucun LLM configuré")

        t0 = self._clock()
        for i, backend in enumerate(candidates):
            produced = False
            try:
                for delta in backend.stream(system, messages):
                    if not produced:
                        produced = True
                        info.backend = backend.name
                        info.fallback = i > 0
                        info.first_token_s = self._clock() - t0
                    yield delta
                return
            except LLMRefusal as exc:
                info.errors.append(f"{backend.name}: {exc}")
                log.warning("%s : %s", backend.name, exc)
                if not produced:
                    info.backend = backend.name
                    yield REFUSAL_REPLY
                return
            except LLMError as exc:
                info.errors.append(f"{backend.name}: {exc}")
                log.warning("Échec de %s : %s", backend.name, exc)
                if backend is self.cloud:
                    self._cloud_down_until = self._clock() + self.retry_cooldown_s
                if produced:
                    return
        raise NoBackendAvailable("; ".join(info.errors))

    def warmup(self) -> None:
        for backend in (self.cloud, self.local):
            if backend is not None:
                backend.warmup()


def create_router(cfg: dict) -> LLMRouter:
    llm_cfg = cfg["llm"]
    cloud = local = None
    if llm_cfg["cloud"]["enabled"]:
        from .claude import ClaudeBackend

        if ClaudeBackend.credentials_available():
            cloud = ClaudeBackend(**llm_cfg["cloud"])
        else:
            log.warning("ANTHROPIC_API_KEY absente : fonctionnement en LLM local uniquement")
    if llm_cfg["local"]["enabled"]:
        from .local import OllamaBackend

        local = OllamaBackend(**llm_cfg["local"])
    return LLMRouter(cloud, local, retry_cooldown_s=llm_cfg["cloud"]["retry_cooldown_s"])
