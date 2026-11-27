import numpy as np

from jone.audio_output import NullSpeaker
from jone.conversation import Conversation
from jone.led import Led, State
from jone.llm.base import LLMError
from jone.llm.router import LLMRouter
from jone.pipeline import EMPTY_REPLY, NO_LLM_REPLY, VoicePipeline
from jone.stt import Transcription
from jone.text_filter import TranscriptFilter, WakeWordGate
from rapport_tests import longest_ok_streak, report


class FakeBackend:
    def __init__(self, name, text="", fail=False):
        self.name, self.text, self.fail = name, text, fail
        self.seen = []

    def stream(self, system, messages):
        self.seen.append(messages)
        if self.fail:
            raise LLMError("hors ligne")
        for word in self.text.split(" "):
            yield word + " "

    def warmup(self):
        pass


class FakeTTS:
    def __init__(self):
        self.spoken = []
        self.languages = []

    def synthesize(self, text, language=None):
        self.spoken.append(text)
        self.languages.append(language)
        return b"\x00\x00" * 160  # 10 ms


class RecordingLed(Led):
    def __init__(self):
        super().__init__({s.value: [0, 0, 0] for s in State})
        self.history = []

    def _apply(self, state, r, g, b):
        self.history.append(state)


class MemoryLog:
    def __init__(self):
        self.records = []

    def write(self, record):
        self.records.append({"horodatage": "2026-10-26T10:00:00", **record})


class FakeMic:
    def __init__(self):
        self.events = []

    def pause(self):
        self.events.append("pause")

    def resume(self):
        self.events.append("resume")


class FakeSTT:
    def __init__(self, tr):
        self.tr = tr

    def transcribe(self, audio, prefer=None):
        self.prefer = prefer
        return self.tr


def make_pipeline(cloud=None, local=None, **kw):
    led, log, tts = RecordingLed(), MemoryLog(), FakeTTS()
    p = VoicePipeline(
        router=LLMRouter(cloud, local),
        conversation=Conversation(5),
        system_prompt="Tu es J-ONE.",
        tts=tts,
        speaker=NullSpeaker(),
        led=led,
        exchange_log=log,
        tail_margin_s=0,
        error_hold_s=0,
        **kw,
    )
    return p, led, log, tts


def test_reply_is_spoken_logged_and_remembered():
    from jone.pipeline import LANGUAGE_HINTS

    cloud = FakeBackend("claude", "Bonjour ! Je suis **J-ONE**. Ravi de vous voir.")
    p, led, log, tts = make_pipeline(cloud=cloud)
    reply = p.respond("Salut")
    assert tts.spoken == ["Bonjour !", "Je suis J-ONE.", "Ravi de vous voir."]
    assert reply == "Bonjour ! Je suis J-ONE. Ravi de vous voir."
    assert led.history == [State.THINKING, State.SPEAKING]
    rec = log.records[-1]
    assert rec["statut"] == "ok" and rec["llm"] == "claude" and rec["bascule"] is False
    assert rec["latences_ms"]["bout_en_bout"] is not None
    p.respond("Et toi ?")
    assert cloud.seen[-1][:2] == [{"role": "user", "content": "Salut" + LANGUAGE_HINTS["fr"]},
                                  {"role": "assistant", "content": reply}]


def test_reply_cut_after_three_sentences():
    cloud = FakeBackend("claude", "Un. Deux. Trois. Quatre. Cinq.")
    p, _, _, tts = make_pipeline(cloud=cloud)
    p.respond("Compte")
    assert tts.spoken == ["Un.", "Deux.", "Trois."]


def test_offline_fallback_to_local():
    p, _, log, _ = make_pipeline(cloud=FakeBackend("claude", fail=True),
                                 local=FakeBackend("local", "Je suis là."))
    assert p.respond("Allô ?") == "Je suis là."
    assert log.records[-1]["llm"] == "local" and log.records[-1]["bascule"] is True


def test_no_llm_says_apology_and_shows_error():
    p, led, log, tts = make_pipeline(local=FakeBackend("local", fail=True))
    p.respond("Allô ?")
    assert tts.spoken == [NO_LLM_REPLY]
    assert led.history[-1] == State.ERROR
    assert log.records[-1]["statut"] == "erreur_llm"
    assert p.conversation.messages == []


def test_empty_llm_answer():
    p, _, log, tts = make_pipeline(cloud=FakeBackend("claude", "😀"))
    p.respond("?")
    assert tts.spoken == [EMPTY_REPLY] and log.records[-1]["statut"] == "reponse_vide"


def test_utterance_rejected_by_filter_is_not_sent_to_llm():
    cloud = FakeBackend("claude", "Non.")
    mic = FakeMic()
    p, _, log, tts = make_pipeline(
        cloud=cloud, mic=mic, text_filter=TranscriptFilter(),
        stt=FakeSTT(Transcription("Sous-titres réalisés par la communauté d'Amara.org")),
    )
    p.handle_utterance(np.zeros(16000, dtype=np.int16), detected_at=0.0)
    assert cloud.seen == [] and tts.spoken == []
    assert log.records[-1]["statut"] == "rejete"
    assert mic.events == ["pause"]


def test_utterance_goes_through_full_chain_with_wake_word():
    cloud = FakeBackend("claude", "Il est midi.")
    p, _, log, tts = make_pipeline(
        cloud=cloud, mic=FakeMic(), text_filter=TranscriptFilter(),
        wake=WakeWordGate(enabled=True, words=["j-one"]),
        stt=FakeSTT(Transcription("J-ONE, quelle heure est-il ?", avg_logprob=-0.1)),
    )
    p.handle_utterance(np.zeros(16000, dtype=np.int16), detected_at=0.0)
    assert cloud.seen[0][-1]["content"].startswith("quelle heure est-il ?")
    assert tts.spoken == ["Il est midi."]
    assert log.records[-1]["transcription"] == "J-ONE, quelle heure est-il ?"


def test_report_from_logs():
    cloud = FakeBackend("claude", "Oui.")
    p, _, log, _ = make_pipeline(cloud=cloud)
    for _ in range(3):
        p.respond("Question")
    log.records.append({"horodatage": "2026-10-26T10:00:01", "statut": "rejete", "motif": "vide"})
    text = report(log.records)
    assert "| claude | 3 |" in text
    assert "Taux de bascule : 0/3" in text
    assert longest_ok_streak(log.records) == 3


def test_english_utterance_gets_english_voice_and_log():
    cloud = FakeBackend("groq", "Hello! I am Ultron.")
    p, _, log, tts = make_pipeline(
        cloud=cloud, mic=FakeMic(), text_filter=TranscriptFilter(),
        stt=FakeSTT(Transcription("Hello, who are you?", avg_logprob=-0.1, language="en")),
    )
    p.handle_utterance(np.zeros(16000, dtype=np.int16), detected_at=0.0)
    assert tts.spoken == ["Hello!", "I am Ultron."]
    assert tts.languages == ["en", "en"]
    assert log.records[-1]["langue"] == "en"


def test_fallback_messages_follow_language():
    from jone.pipeline import EMPTY_REPLIES, NO_LLM_REPLIES

    p, _, _, tts = make_pipeline(local=FakeBackend("local", fail=True))
    p.respond("Hello?", language="en")
    assert tts.spoken == [NO_LLM_REPLIES["en"]]
    p2, _, _, tts2 = make_pipeline(cloud=FakeBackend("claude", "😀"))
    p2.respond("?", language="en")
    assert tts2.spoken == [EMPTY_REPLIES["en"]]


def test_english_hallucinations_filtered():
    f = TranscriptFilter()
    assert not f.check(Transcription("Thanks for watching!")).accepted
    assert not f.check(Transcription("Thank you.", avg_logprob=-0.9, no_speech_prob=0.5)).accepted
    assert f.check(Transcription("Thank you!", avg_logprob=-0.2, no_speech_prob=0.05)).accepted


def test_voice_follows_reply_language():
    # Question en anglais, mais le LLM local répond en français : voix française.
    local = FakeBackend("local", "Je suis Sarah. Je suis là pour vous aider.")
    p, _, log, tts = make_pipeline(local=local)
    p.respond("Who are you?", language="en")
    assert tts.languages == ["fr", "fr"]
    assert log.records[-1]["voix"] == "fr" and log.records[-1]["langue"] == "en"


def test_language_hint_sent_to_llm():
    from jone.pipeline import LANGUAGE_HINTS

    cloud = FakeBackend("groq", "Hello.")
    p, _, _, _ = make_pipeline(cloud=cloud)
    p.respond("Hi", language="en")
    assert cloud.seen[0][-1]["content"] == "Hi" + LANGUAGE_HINTS["en"]
    # l'historique garde le message tel qu'envoyé, pour le cache du LLM local
    assert p.conversation.messages[0]["content"] == "Hi" + LANGUAGE_HINTS["en"]


def test_warmup_synthesizes_each_voice():
    p, _, _, tts = make_pipeline(cloud=FakeBackend("claude", "Oui."))
    tts.voices = {"fr": None, "en": None}
    p.warmup()
    assert tts.languages == ["fr", "en"]


def test_uncertain_utterance_is_clarified_then_sent_to_llm_with_warning():
    from jone.dialogue import CLARIFY_REPLIES
    from jone.pipeline import UNCERTAIN_HINTS

    cloud = FakeBackend("groq", "Pardon, pouvez-vous répéter ?")
    p, _, log, tts = make_pipeline(
        cloud=cloud, mic=FakeMic(), text_filter=TranscriptFilter(),
        stt=FakeSTT(Transcription("Est-ce que les gombras se lèvés?", avg_logprob=-0.97)),
    )
    audio = np.zeros(16000, dtype=np.int16)
    # 1re fois : le robot fait répéter lui-même, sans LLM
    p.handle_utterance(audio, detected_at=0.0)
    assert cloud.seen == [] and tts.spoken == [CLARIFY_REPLIES["fr"]]
    assert log.records[-1]["llm"] == "clarification" and log.records[-1]["incertain"] is True
    # 2e fois de suite : le LLM répond, prévenu de l'incertitude, hors historique
    p.handle_utterance(audio, detected_at=0.0)
    assert cloud.seen[0][-1]["content"].endswith(UNCERTAIN_HINTS["fr"])
    assert p.conversation.messages == []


def test_language_command_switches_without_llm():
    from jone.dialogue import LANGUAGE_SWITCH_REPLIES

    cloud = FakeBackend("groq", "Bonjour.")
    stt = FakeSTT(Transcription("Maintenant je voulais qu'on parle en anglais si possible.", avg_logprob=-0.2))
    p, _, log, tts = make_pipeline(cloud=cloud, mic=FakeMic(), text_filter=TranscriptFilter(), stt=stt)
    p.handle_utterance(np.zeros(16000, dtype=np.int16), detected_at=0.0)
    assert cloud.seen == [] and tts.spoken == [LANGUAGE_SWITCH_REPLIES["en"]] and tts.languages == ["en"]
    assert log.records[-1]["llm"] == "commande_langue"
    # la conversation continue en anglais : préférence transmise à Whisper
    stt.tr = Transcription("What is your name?", avg_logprob=-0.2, language="en")
    p.handle_utterance(np.zeros(16000, dtype=np.int16), detected_at=0.0)
    assert stt.prefer == "en"


def test_conversation_language_forgotten_after_idle(monkeypatch):
    import jone.pipeline as pipeline_mod

    now = [1000.0]
    monkeypatch.setattr(pipeline_mod.time, "monotonic", lambda: now[0])
    stt = FakeSTT(Transcription("Bonjour.", avg_logprob=-0.2, language="fr"))
    p, _, _, _ = make_pipeline(cloud=FakeBackend("groq", "Bonjour !"), mic=FakeMic(),
                               text_filter=TranscriptFilter(), stt=stt)
    p.handle_utterance(np.zeros(16000, dtype=np.int16), detected_at=0.0)
    assert p.conversation_language == "fr"
    now[0] += 121
    assert p.conversation_language is None
