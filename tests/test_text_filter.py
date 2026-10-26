from jone.stt import Transcription
from jone.text_filter import TranscriptFilter, WakeWordGate


def check(text, **kw):
    return TranscriptFilter().check(Transcription(text, **kw))


def test_accepts_normal_sentence():
    r = check("Bonjour, comment ça va ?", avg_logprob=-0.2, no_speech_prob=0.05)
    assert r.accepted and r.text == "Bonjour, comment ça va ?"


def test_rejects_empty_and_punctuation():
    assert not check("").accepted
    assert not check(" ... ").accepted
    assert check("♪").reason == "vide"


def test_rejects_known_hallucinations():
    r = check("Sous-titres réalisés par la communauté d’Amara.org")
    assert not r.accepted and r.reason == "hallucination connue"


def test_merci_accepted_only_when_confident():
    assert check("Merci !", avg_logprob=-0.2, no_speech_prob=0.05).accepted
    assert not check("Merci.", avg_logprob=-0.9, no_speech_prob=0.5).accepted


def test_rejects_low_confidence_noise():
    r = check("euh la porte", avg_logprob=-1.5, no_speech_prob=0.8)
    assert not r.accepted


def test_rejects_repetition_loops():
    assert not check("oui oui", compression_ratio=3.0).accepted
    assert not check("je pense que " * 6).accepted


def test_wake_word_disabled_passes_everything():
    assert WakeWordGate(enabled=False).admit("quelle heure", now=0) == "quelle heure"


def test_wake_word_required_then_session():
    gate = WakeWordGate(enabled=True, words=["j-one", "djiwan"], session_timeout_s=30)
    assert gate.admit("quelle heure est-il", now=0) is None
    assert gate.admit("J-ONE, quelle heure est-il ?", now=1) == "quelle heure est-il ?"
    gate.mark_exchange(now=2)
    assert gate.admit("et demain ?", now=20) == "et demain ?"
    assert gate.admit("et demain ?", now=40) is None


def test_wake_word_alone_becomes_greeting():
    gate = WakeWordGate(enabled=True, words=["djiwan"])
    assert gate.admit("Djiwan !", now=0) == "Bonjour"
