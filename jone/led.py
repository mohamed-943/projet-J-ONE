"""Signalisation de l'état du pipeline par la LED de la tête (EF-11).

vert = écoute, bleu = réflexion, blanc = parole, rouge = erreur.
"""

from __future__ import annotations

import enum
import logging

log = logging.getLogger(__name__)


class State(str, enum.Enum):
    IDLE = "idle"
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"
    ERROR = "error"


class Led:
    """Interface commune ; `set_state` n'émet une commande que si l'état change."""

    def __init__(self, colors: dict):
        self.colors = {State(k): tuple(int(c) for c in v) for k, v in colors.items()}
        self.state: State | None = None

    def set_state(self, state: State) -> None:
        if state == self.state:
            return
        self.state = state
        r, g, b = self.colors.get(state, (0, 0, 0))
        try:
            self._apply(state, r, g, b)
        except Exception:  # une LED défaillante ne doit jamais arrêter le pipeline
            log.exception("Échec de la commande LED (%s)", state.value)

    def _apply(self, state: State, r: int, g: int, b: int) -> None:
        pass


class ConsoleLed(Led):
    _LABELS = {
        State.IDLE: "veille",
        State.LISTENING: "ÉCOUTE (vert)",
        State.THINKING: "RÉFLEXION (bleu)",
        State.SPEAKING: "PAROLE (blanc)",
        State.ERROR: "ERREUR (rouge)",
    }

    def _apply(self, state, r, g, b):
        print(f"[LED] {self._LABELS[state]}", flush=True)


class G1Led(Led):
    def __init__(self, colors: dict, robot_cfg: dict):
        super().__init__(colors)
        from .robot import get_audio_client

        self.client = get_audio_client(robot_cfg)

    def _apply(self, state, r, g, b):
        code = self.client.LedControl(r, g, b)
        if code not in (None, 0):
            log.warning("LedControl a renvoyé le code %s", code)


def create_led(cfg: dict) -> Led:
    led_cfg = cfg["led"]
    backend = led_cfg["backend"]
    if backend == "g1":
        return G1Led(led_cfg["colors"], cfg["robot"])
    if backend == "console":
        return ConsoleLed(led_cfg["colors"])
    return Led(led_cfg["colors"])
