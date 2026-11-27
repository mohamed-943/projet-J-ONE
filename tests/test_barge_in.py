import time

import numpy as np

from jone.audio_output import NullSpeaker
from jone.barge_in import BargeInMonitor, echo_ratio, is_interruption
from jone.stt import Transcription

ROBOT = "Bonjour, je suis Sibo, le robot de l'équipe. Je peux discuter avec vous."


def test_echo_of_robot_voice_is_not_an_interruption():
    assert is_interruption("je suis Sibo, le robot de l'équipe", -0.3, ROBOT) is None
    # « Sibo » est dit par le robot lui-même : seul, il ne suffit pas
    assert is_interruption("Sibo", -0.2, ROBOT) is None


def test_keywords_and_new_words_interrupt():
    assert is_interruption("Stop, stop !", -0.4, ROBOT) == "mot « stop »"
    assert is_interruption("attends une seconde", -0.4, ROBOT).startswith("mot")
    assert is_interruption("Sibo, quelle heure est-il ?", -0.3, "Il fait beau aujourd'hui.") == "mot « sibo »"
    assert is_interruption("quelle heure est-il maintenant", -0.3, ROBOT) == "4 mots nouveaux"
    # mots nouveaux mais confiance trop faible : bruit probable
    assert is_interruption("quelle heure est-il maintenant", -1.3, ROBOT) is None


def test_echo_ratio():
    assert echo_ratio("je peux discuter avec vous", ROBOT) == 1.0
    assert echo_ratio("quelle heure est-il", ROBOT) == 0.0


class StreamMic:
    """Micro factice : des trames sans fin, tant que l'écoute n'est pas en pause."""

    def __init__(self):
        self.paused = False

    def read(self, timeout=0.1):
        time.sleep(0.002)
        return np.zeros(512, dtype=np.int16)

    def flush(self): pass
    def pause(self): self.paused = True
    def resume(self): self.paused = False


class EchoThenUserSTT:
    """La 1re fenêtre ne contient que l'écho, la suivante l'utilisateur."""

    def __init__(self):
        self.calls = 0

    def transcribe(self, audio, prefer=None):
        self.calls += 1
        text = "Un. Deux." if self.calls == 1 else "Stop, attends !"
        return Transcription(text, avg_logprob=-0.3, language="fr")


def test_monitor_ignores_echo_then_detects_user():
    stt = EchoThenUserSTT()
    m = BargeInMonitor(StreamMic(), lambda f: 1.0, stt, window_s=0.1, hop_s=0.05)
    m.start(lambda: "Un. Deux. Trois.", "fr")
    assert m.triggered.wait(2.0)
    m.stop()
    assert stt.calls >= 2 and m.reason == "mot « stop »" and len(m.recent_audio(0.6)) > 0


def test_pipeline_stops_speaking_when_interrupted():
    from test_pipeline import FakeBackend, MemoryLog, RecordingLed

    from jone.conversation import Conversation
    from jone.led import State
    from jone.llm.router import LLMRouter
    from jone.pipeline import VoicePipeline

    class LongTTS:
        def synthesize(self, text, language=None):
            return b"\x00\x00" * 16000 * 2  # 2 s par phrase

    stt = EchoThenUserSTT()
    mic = StreamMic()
    log = MemoryLog()
    p = VoicePipeline(
        router=LLMRouter(FakeBackend("local", "Un. Deux. Trois."), None), conversation=Conversation(5),
        system_prompt="Tu es Sibo.", tts=LongTTS(), speaker=NullSpeaker(), led=RecordingLed(),
        exchange_log=log, tail_margin_s=0, error_hold_s=0, mic=mic,
        barge_in=BargeInMonitor(mic, lambda f: 1.0, stt, window_s=0.1, hop_s=0.05),
    )
    t0 = time.monotonic()
    p.respond("Raconte-moi quelque chose", language="fr")
    elapsed = time.monotonic() - t0
    rec = log.records[-1]
    assert elapsed < 3.0  # 3 phrases de 2 s : coupé bien avant les 6 s
    assert rec["statut"] == "interrompu" and rec["interruption"]["motif"] == "mot « stop »"
    assert p._pending_audio is not None and mic.paused
