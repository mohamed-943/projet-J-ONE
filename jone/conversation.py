"""Historique de conversation partagé par les deux LLM (EF-05)."""

from __future__ import annotations

from pathlib import Path


def load_system_prompt(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


class Conversation:
    def __init__(self, max_turns: int = 10):
        self.max_turns = max_turns
        self.messages: list[dict[str, str]] = []

    def with_user(self, text: str) -> list[dict[str, str]]:
        """Messages à envoyer au LLM pour ce tour, sans modifier l'historique."""
        return self.messages + [{"role": "user", "content": text}]

    def commit(self, user_text: str, assistant_text: str) -> None:
        """N'enregistre que les échanges réellement prononcés par le robot."""
        if not assistant_text.strip():
            return
        self.messages.append({"role": "user", "content": user_text})
        self.messages.append({"role": "assistant", "content": assistant_text})
        if len(self.messages) > 2 * self.max_turns:
            # Coupe par blocs : on ne garde que la moitié la plus récente. Retirer un
            # seul échange à chaque tour changerait le début de la conversation à
            # chaque fois, et le LLM local devrait relire tout l'historique au lieu
            # de réutiliser son calcul (2 s au lieu de 0,2 s mesurées sur la Jetson).
            keep = max(1, self.max_turns // 2)
            self.messages = self.messages[-2 * keep :]

    def clear(self) -> None:
        self.messages.clear()
