"""Restitution audio. Les phrases sont mises en file et jouées dans l'ordre par un
fil dédié, pour que la synthèse de la phrase suivante se fasse pendant la lecture.

`G1Speaker` envoie le PCM 16 kHz mono au haut-parleur via PlayStream(). Le robot
met le flux en tampon : on cadence l'envoi par blocs d'environ une seconde, avec
`lead_seconds` d'avance, et on connaît ainsi l'heure de fin réelle de la lecture,
indispensable pour reprendre l'écoute au bon moment (EF-10).
"""

from __future__ import annotations

import logging
import queue
import threading
import time

from .audio_utils import TARGET_RATE, pcm_bytes_to_int16, pcm_duration_s

log = logging.getLogger(__name__)


class Speaker:
    def __init__(self):
        self._queue: queue.Queue[bytes | None] = queue.Queue()
        self._idle = threading.Event()
        self._idle.set()
        self._lock = threading.Lock()
        self._pending = 0  # phrases en file ou en cours de lecture
        self._first_audio_at: float | None = None
        self._thread = threading.Thread(target=self._run, name="speaker", daemon=True)
        self._thread.start()

    # --- API publique -----------------------------------------------------
    def begin_response(self) -> None:
        self._first_audio_at = None

    def enqueue(self, pcm: bytes) -> None:
        if pcm:
            with self._lock:
                self._pending += 1
                self._idle.clear()
            self._queue.put(pcm)

    def wait_until_done(self, timeout: float | None = None) -> bool:
        return self._idle.wait(timeout)

    @property
    def first_audio_at(self) -> float | None:
        """Instant (perf_counter) où le premier son de la réponse a été émis."""
        return self._first_audio_at

    def stop(self) -> None:
        """Interrompt la lecture et vide la file."""
        with self._lock:
            while True:
                try:
                    self._queue.get_nowait()
                    self._pending -= 1
                except queue.Empty:
                    break
            if self._pending <= 0:
                self._pending = 0
                self._idle.set()
        self._stop_playback()

    def close(self) -> None:
        self.stop()
        self._queue.put(None)
        self._thread.join(timeout=2.0)

    # --- à implémenter ----------------------------------------------------
    def _play_blocking(self, pcm: bytes) -> None:
        """Joue le PCM et ne rend la main qu'à la fin (ou presque) de la lecture."""
        raise NotImplementedError

    def _drain(self) -> None:
        """Attend la fin réelle du son déjà envoyé."""

    def _stop_playback(self) -> None:
        pass

    def _mark_first_audio(self) -> None:
        if self._first_audio_at is None:
            self._first_audio_at = time.perf_counter()

    # --- fil de lecture -----------------------------------------------------
    def _run(self) -> None:
        while True:
            pcm = self._queue.get()
            if pcm is None:
                return
            try:
                self._play_blocking(pcm)
                if self._queue.empty():
                    self._drain()
            except Exception:
                log.exception("Erreur de lecture audio")
            with self._lock:
                self._pending -= 1
                if self._pending <= 0:
                    self._pending = 0
                    self._idle.set()


class NullSpeaker(Speaker):
    """Pas de son : simule la durée de lecture (tests sans haut-parleur)."""

    def _play_blocking(self, pcm):
        self._mark_first_audio()
        time.sleep(pcm_duration_s(pcm))


class SoundDeviceSpeaker(Speaker):
    def _play_blocking(self, pcm):
        import sounddevice as sd

        self._mark_first_audio()
        sd.play(pcm_bytes_to_int16(pcm), TARGET_RATE, blocking=True)

    def _stop_playback(self):
        import sounddevice as sd

        sd.stop()


class G1Speaker(Speaker):
    def __init__(self, robot_cfg: dict, app_name="jone", volume=80, chunk_seconds=1.0,
                 lead_seconds=0.5):
        from .robot import get_audio_client

        self.client = get_audio_client(robot_cfg)
        self.app_name = app_name
        self.chunk_bytes = int(chunk_seconds * TARGET_RATE) * 2
        self.lead_seconds = lead_seconds
        self._play_end = 0.0
        self._stream_id = ""
        try:
            self.client.SetVolume(int(volume))
        except Exception:
            log.exception("SetVolume a échoué")
        super().__init__()

    def begin_response(self):
        super().begin_response()
        # Un identifiant de flux par réponse : les phrases s'enchaînent dans le même flux.
        self._stream_id = str(int(time.time() * 1000))

    def _play_blocking(self, pcm):
        for offset in range(0, len(pcm), self.chunk_bytes):
            chunk = pcm[offset : offset + self.chunk_bytes]
            now = time.monotonic()
            if self._play_end < now:  # le haut-parleur s'était tu : nouveau départ
                self._play_end = now
            ret = self.client.PlayStream(self.app_name, self._stream_id or "jone", chunk)
            if ret not in (None, 0):
                log.warning("PlayStream a renvoyé le code %s", ret)
            self._mark_first_audio()
            self._play_end += pcm_duration_s(chunk)
            # Ne pas prendre plus de `lead_seconds` d'avance sur la lecture.
            ahead = self._play_end - time.monotonic() - self.lead_seconds
            if ahead > 0:
                time.sleep(ahead)

    def _drain(self):
        remaining = self._play_end - time.monotonic()
        if remaining > 0:
            time.sleep(remaining)

    def _stop_playback(self):
        try:
            self.client.PlayStop(self.app_name)
        except Exception:
            log.exception("PlayStop a échoué")
        self._play_end = 0.0


def create_speaker(cfg: dict) -> Speaker:
    out = cfg["audio_output"]
    if out["backend"] == "g1":
        return G1Speaker(
            cfg["robot"],
            app_name=out["app_name"],
            volume=out["volume"],
            chunk_seconds=out["chunk_seconds"],
            lead_seconds=out["lead_seconds"],
        )
    if out["backend"] == "sounddevice":
        return SoundDeviceSpeaker()
    return NullSpeaker()
