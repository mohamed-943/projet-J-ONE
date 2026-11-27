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
    assert check("♪").reason == "son non verbal"


def test_rejects_non_verbal_annotations_and_keeps_words_around_them():
    # vus sur le robot : « *cris* » envoyé au LLM
    for text in ("*cris*", "[Musique]", "(rires)", "*applaudissements* ♪"):
        assert check(text, avg_logprob=-0.3).reason == "son non verbal", text
    r = check("*rires* Bonjour Sarah", avg_logprob=-0.3)
    assert r.accepted and r.text == "Bonjour Sarah"


def test_rejects_multi_voice_dialogue():
    assert check("- Le déjeuner le matériel ? - Tout mal.", avg_logprob=-0.4).reason == "plusieurs voix"
    assert check("Bonjour - comment ça va ?", avg_logprob=-0.2).accepted


def test_confidence_levels():
    assert check("The octodino", avg_logprob=-1.49).reason == "confiance trop faible"
    r = check("Est-ce que les gombras se lèvés?", avg_logprob=-0.97)
    assert r.accepted and r.uncertain
    r = check("Comment tu t'appelles ?", avg_logprob=-0.3)
    assert r.accepted and not r.uncertain


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
