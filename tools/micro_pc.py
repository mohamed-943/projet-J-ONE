"""Envoie le micro du PC au robot, quand le micro du G1 ne fonctionne pas.

    Sur le PC     : python tools/micro_pc.py [--robot 192.168.123.164] [--port 5556] [--device N]
    Sur le robot  : python -m jone --input udp

PCM 16 kHz mono int16, une trame de 512 échantillons (32 ms) par paquet UDP. Le
robot répond toujours par son propre haut-parleur.
"""

from __future__ import annotations

import argparse
import socket


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--robot", default="192.168.123.164", help="adresse de la Jetson")
    p.add_argument("--port", type=int, default=5556)
    p.add_argument("--device", help="micro (index ou nom, voir --liste)")
    p.add_argument("--liste", action="store_true", help="afficher les micros disponibles")
    args = p.parse_args()

    import sounddevice as sd

    if args.liste:
        print(sd.query_devices())
        return
    device = int(args.device) if args.device and args.device.isdigit() else args.device
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    target = (args.robot, args.port)

    def callback(indata, frames, time_info, status):
        sock.sendto(indata[:, 0].tobytes(), target)

    with sd.InputStream(samplerate=16000, channels=1, dtype="int16", blocksize=512,
                        device=device, callback=callback):
        print(f"Micro du PC envoyé à {target[0]}:{target[1]} — Ctrl+C pour arrêter")
        try:
            while True:
                sd.sleep(1000)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
