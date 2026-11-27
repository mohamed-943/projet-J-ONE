"""Capture micro (objectif 1).

Deux sources, même interface :
- `SoundDeviceMic` : micro du PC de développement ou micro USB ;
- `G1MulticastMic` : réseau de micros du G1, diffusé en UDP multicast (PCM 16 kHz
  mono 16 bits) sur le réseau interne du robot. Groupe et port à confirmer dans la
  documentation développeur Unitree (voir README, « points à vérifier »).

Les trames sont redécoupées en blocs fixes de `frame_samples` échantillons (512 =
32 ms, taille attendue par Silero VAD). `pause()` coupe l'écoute pendant que le
robot parle (EF-10) : les trames reçues sont alors jetées.
"""

from __future__ import annotations

import logging
import queue
import socket
import struct
import threading
import time

import numpy as np

log = logging.getLogger(__name__)


class MicSource:
    def __init__(self, sample_rate: int, frame_samples: int, queue_max_frames: int = 500):
        self.sample_rate = sample_rate
        self.frame_samples = frame_samples
        self._queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=queue_max_frames)
        self._pending = np.zeros(0, dtype=np.int16)
        self._paused = threading.Event()
        self._lock = threading.Lock()

    # --- API publique -----------------------------------------------------
    def start(self) -> None:
        raise NotImplementedError

    def close(self) -> None:
        pass

    def read(self, timeout: float = 1.0) -> np.ndarray | None:
        """Renvoie une trame int16 de `frame_samples` échantillons, ou None."""
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def pause(self) -> None:
        self._paused.set()
        self.flush()

    def resume(self) -> None:
        self.flush()
        self._paused.clear()

    def flush(self) -> None:
        with self._lock:
            self._pending = np.zeros(0, dtype=np.int16)
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                return

    # --- côté producteur ----------------------------------------------------
    def _push(self, samples: np.ndarray) -> None:
        if self._paused.is_set():
            return
        with self._lock:
            buf = np.concatenate([self._pending, samples.astype(np.int16, copy=False)])
            n = self.frame_samples
            full = len(buf) // n
            for i in range(full):
                frame = buf[i * n : (i + 1) * n]
                try:
                    self._queue.put_nowait(frame)
                except queue.Full:
                    # Consommateur en retard : on jette la plus ancienne trame.
                    try:
                        self._queue.get_nowait()
                    except queue.Empty:
                        pass
                    self._queue.put_nowait(frame)
            self._pending = buf[full * n :].copy()


class SoundDeviceMic(MicSource):
    def __init__(self, sample_rate, frame_samples, device=None, **kw):
        super().__init__(sample_rate, frame_samples, **kw)
        self.device = device
        self._stream = None

    def start(self):
        import sounddevice as sd

        def callback(indata, frames, time_info, status):
            if status:
                log.debug("sounddevice: %s", status)
            self._push(indata[:, 0].copy())

        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="int16",
            blocksize=self.frame_samples,
            device=self.device,
            callback=callback,
        )
        self._stream.start()
        log.info("Micro sounddevice démarré (%s Hz)", self.sample_rate)

    def close(self):
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None


class G1MulticastMic(MicSource):
    def __init__(self, sample_rate, frame_samples, group, port, local_ip, **kw):
        super().__init__(sample_rate, frame_samples, **kw)
        self.group, self.port, self.local_ip = group, int(port), local_ip
        self._sock: socket.socket | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("", self.port))
        mreq = struct.pack("4s4s", socket.inet_aton(self.group), socket.inet_aton(self.local_ip))
        sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
        sock.settimeout(1.0)
        self._sock = sock
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="g1-mic", daemon=True)
        self._thread.start()
        log.info("Écoute du micro G1 sur %s:%s via %s", self.group, self.port, self.local_ip)

    def _run(self):
        silent_since = time.monotonic()
        warned = False
        while not self._stop.is_set():
            try:
                data, _ = self._sock.recvfrom(65536)
            except socket.timeout:
                continue
            except OSError:
                if not self._stop.is_set():
                    log.exception("Erreur de réception du micro G1")
                return
            if len(data) % 2:
                data = data[:-1]
            samples = np.frombuffer(data, dtype="<i2")
            # Constaté le 28/11 après un redémarrage du robot : paquets reçus mais
            # entièrement nuls, le micro est coupé côté robot.
            if samples.any():
                silent_since, warned = time.monotonic(), False
            elif not warned and time.monotonic() - silent_since > 5.0:
                warned = True
                log.warning("Le micro du G1 n'envoie que du silence (paquets nuls) : micro coupé côté "
                            "robot. Autres sources : --input udp (micro du PC, tools/micro_pc.py) ou "
                            "--input alsa (micro USB branché sur le robot).")
            self._push(samples)

    def close(self):
        self._stop.set()
        if self._sock is not None:
            self._sock.close()
            self._sock = None
        if self._thread is not None:
            self._thread.join(timeout=2.0)


class UdpMic(G1MulticastMic):
    """Micro d'une autre machine (le PC), reçu en UDP : PCM 16 kHz mono int16 brut,
    envoyé par tools/micro_pc.py. Sert quand le micro du robot ne fonctionne pas."""

    def __init__(self, sample_rate, frame_samples, port, **kw):
        super().__init__(sample_rate, frame_samples, group=None, port=port, local_ip=None, **kw)

    def start(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("", self.port))
        sock.settimeout(1.0)
        self._sock = sock
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="udp-mic", daemon=True)
        self._thread.start()
        log.info("Écoute du micro distant (UDP) sur le port %s : lancer tools/micro_pc.py sur le PC", self.port)


class AlsaMic(MicSource):
    """Micro USB branché sur la Jetson, lu par `arecord` (ALSA). PortAudio, dont a
    besoin sounddevice, ne s'installe pas sur la Jetson (apt cassé)."""

    def __init__(self, sample_rate, frame_samples, device="default", **kw):
        super().__init__(sample_rate, frame_samples, **kw)
        self.device = device
        self._proc = None
        self._thread: threading.Thread | None = None

    def start(self):
        import subprocess

        cmd = ["arecord", "-q", "-D", self.device, "-f", "S16_LE", "-r", str(self.sample_rate),
               "-c", "1", "-t", "raw"]
        self._proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self._thread = threading.Thread(target=self._run, name="alsa-mic", daemon=True)
        self._thread.start()
        log.info("Micro ALSA %s démarré (%s Hz)", self.device, self.sample_rate)

    def _run(self):
        chunk = self.frame_samples * 2
        while self._proc and self._proc.poll() is None:
            data = self._proc.stdout.read(chunk)
            if not data:
                break
            self._push(np.frombuffer(data[: len(data) // 2 * 2], dtype="<i2"))
        if self._proc and self._proc.returncode not in (None, 0):
            log.error("arecord s'est arrêté : %s", self._proc.stderr.read().decode(errors="replace").strip())

    def close(self):
        if self._proc is not None:
            self._proc.terminate()
            self._proc = None
        if self._thread is not None:
            self._thread.join(timeout=2.0)


def create_mic(cfg: dict) -> MicSource:
    a = cfg["audio_input"]
    common = dict(
        sample_rate=a["sample_rate"],
        frame_samples=a["frame_samples"],
        queue_max_frames=a["queue_max_frames"],
    )
    if a["backend"] == "g1_multicast":
        return G1MulticastMic(
            group=a["g1_multicast_group"],
            port=a["g1_multicast_port"],
            local_ip=a["g1_local_ip"],
            **common,
        )
    if a["backend"] == "udp":
        return UdpMic(port=a["udp_port"], **common)
    if a["backend"] == "alsa":
        return AlsaMic(device=a["alsa_device"], **common)
    return SoundDeviceMic(device=a["device"], **common)
