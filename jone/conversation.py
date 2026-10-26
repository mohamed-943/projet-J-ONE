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
        excess = len(self.messages) - 2 * self.max_turns
        if excess > 0:
            self.messages = self.messages[excess:]

    def clear(self) -> None:
        self.messages.clear()
