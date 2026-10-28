"""Accès partagé au SDK Unitree (unitree_sdk2_python).

PlayStream() et LedControl() sont tous deux exposés par AudioClient : on n'en crée
qu'une instance, et ChannelFactoryInitialize n'est appelé qu'une fois par processus.
Le module vocal n'importe aucun client de locomotion : il ne peut pas commander de
mouvement ni interférer avec l'arrêt d'urgence (section 7 du cahier des charges).
"""

from __future__ import annotations

import logging
import threading

log = logging.getLogger(__name__)

_lock = threading.Lock()
_client = None


def sdk_ok(ret) -> bool:
    """Les appels du SDK renvoient 0, None ou un couple (code, données) : (0, '')."""
    code = ret[0] if isinstance(ret, tuple) and ret else ret
    return code in (None, 0)


def get_audio_client(robot_cfg: dict):
    global _client
    with _lock:
        if _client is None:
            from unitree_sdk2py.core.channel import ChannelFactoryInitialize
            from unitree_sdk2py.g1.audio.g1_audio_client import AudioClient

            iface = robot_cfg.get("network_interface")
            log.info("Initialisation du SDK Unitree sur l'interface %s", iface)
            ChannelFactoryInitialize(0, iface)
            client = AudioClient()
            client.SetTimeout(float(robot_cfg.get("sdk_timeout_s", 10.0)))
            client.Init()
            _client = client
        return _client
