import numpy as np

from jone.audio_utils import pcm_duration_s, to_pcm16k_mono
from jone.config import DEFAULTS, load_config
from jone.conversation import Conversation
from jone.metrics import corpus_wer, percentile, wer


def test_resample_piper_rate_to_16k_mono_int16():
    sr = 22050
    t = np.arange(sr) / sr
    stereo = np.stack([np.sin(2 * np.pi * 440 * t)] * 2, axis=1) * 0.5
    pcm = to_pcm16k_mono((stereo * 32767).astype(np.int16), sr)
    assert len(pcm) == 16000 * 2  # 1 s, 2 octets par échantillon
    assert abs(pcm_duration_s(pcm) - 1.0) < 1e-6
    samples = np.frombuffer(pcm, dtype="<i2")
    assert 0.4 * 32767 < np.abs(samples).max() < 0.6 * 32767


def test_already_16k_is_untouched_in_length():
    pcm = to_pcm16k_mono(np.zeros(1600, dtype=np.int16), 16000)
    assert len(pcm) == 3200


def test_conversation_keeps_last_turns_and_only_spoken_replies():
    conv = Conversation(max_turns=2)
    for i in range(3):
        conv.commit(f"q{i}", f"r{i}")
    conv.commit("q-vide", "  ")
    assert [m["content"] for m in conv.messages] == ["q1", "r1", "q2", "r2"]
    assert conv.with_user("q3")[-1] == {"role": "user", "content": "q3"}
    assert len(conv.messages) == 4


def test_wer():
    assert wer("Bonjour, comment ça va ?", "bonjour comment ça va") == 0
    assert wer("le chat dort", "le chien dort") == 1 / 3
    assert wer("l'école est fermée", "l école est fermé") == 1 / 4
    assert corpus_wer([("a b c d", "a b c d"), ("e f", "e")]) == 1 / 6


def test_percentile():
    assert percentile([1, 2, 3, 4], 50) == 2.5
    assert percentile([], 90) is None


def test_config_file_matches_defaults_schema():
    cfg = load_config("config.yaml")
    for section, values in DEFAULTS.items():
        assert section in cfg
        if isinstance(values, dict):
            assert set(values) <= set(cfg[section])
    assert cfg["llm"]["cloud"]["model"] == "claude-haiku-5-5"
    assert cfg["led"]["colors"]["listening"] == [0, 255, 0]


def test_pc_config_overrides_only_what_it_sets():
    cfg = load_config("config.pc.yaml")
    assert cfg["audio_output"]["backend"] == "sounddevice"
    assert cfg["llm"]["local"]["model"] == "qwen3:4b"


def test_sdk_return_codes_as_seen_on_g1():
    from jone.robot import sdk_ok

    assert sdk_ok(0) and sdk_ok(None) and sdk_ok((0, "")) and sdk_ok((0, {"volume": 100}))
    assert not sdk_ok(3104) and not sdk_ok((3104, ""))


def test_clean_for_speech_english_symbols():
    from jone.text_utils import clean_for_speech

    assert clean_for_speech("It's 20°C and 50% humid", "en") == "It's 20 degrees Celsius and 50 percent humid"


def test_splitter_keeps_english_titles():
    from jone.text_utils import SentenceSplitter

    s = SentenceSplitter()
    assert s.feed("Ask Mr. Smith now. Ok") == ["Ask Mr. Smith now."]


def test_multivoice_falls_back_to_default_voice():
    from jone.tts import MultiVoiceTTS

    class Voice:
        def __init__(self, tag):
            self.tag = tag

        def synthesize(self, text):
            return self.tag

    tts = MultiVoiceTTS({"fr": Voice(b"fr"), "en": Voice(b"en")}, default="fr")
    assert tts.synthesize("x", language="en") == b"en"
    assert tts.synthesize("x", language="es") == b"fr"
    assert tts.synthesize("x") == b"fr"


def test_whisper_language_restricted_to_allowed(monkeypatch):
    import types

    from jone.stt import WhisperSTT

    calls = []

    class FakeModel:
        def transcribe(self, samples, language=None, **kw):
            calls.append(language)
            seg = types.SimpleNamespace(text=" Hola", avg_logprob=-0.2, no_speech_prob=0.1,
                                        compression_ratio=1.0)
            info = types.SimpleNamespace(language=language or "es", language_probability=0.6,
                                         all_language_probs=[("es", 0.6), ("en", 0.3), ("fr", 0.1)])
            return iter([seg]), info

    stt = _bare_stt(FakeModel())
    tr = stt.transcribe(np.zeros(1600, dtype=np.int16))
    assert calls == [None, "en"] and tr.language == "en"


def _bare_stt(model, detector=None):
    from jone.stt import WhisperSTT

    stt = WhisperSTT.__new__(WhisperSTT)
    stt.model, stt.detector, stt.beam_size, stt.language = model, detector, 1, None
    stt.languages, stt.default_language, stt.language_margin = ["fr", "en"], "fr", 0.65
    return stt


class _Model:
    """Whisper factice : renvoie les probabilités de langue données."""

    def __init__(self, probs):
        self.probs, self.calls = probs, []

    def transcribe(self, samples, language=None, **kw):
        import types

        self.calls.append(language)
        seg = types.SimpleNamespace(text=" Bonjour", avg_logprob=-0.2, no_speech_prob=0.1,
                                    compression_ratio=1.0)
        best = max(self.probs, key=self.probs.get)
        info = types.SimpleNamespace(language=language or best, language_probability=self.probs[best],
                                     all_language_probs=sorted(self.probs.items(), key=lambda kv: -kv[1]))
        return iter([seg]), info


def test_whisper_detector_then_single_pass():
    big, tiny = _Model({"en": 0.9, "fr": 0.1}), _Model({"en": 0.8, "fr": 0.1, "de": 0.1})
    tr = _bare_stt(big, detector=tiny).transcribe(np.zeros(1600, dtype=np.int16))
    assert tiny.calls == [None] and big.calls == ["en"] and tr.language == "en"


def test_whisper_hesitation_defaults_to_french():
    # en 0.33 contre fr 0.25 : part de l'anglais 0.57 < 0.65 → français
    big, tiny = _Model({"fr": 0.1}), _Model({"en": 0.33, "fr": 0.25, "nn": 0.42})
    tr = _bare_stt(big, detector=tiny).transcribe(np.zeros(1600, dtype=np.int16))
    assert big.calls == ["fr"] and tr.language == "fr"


def test_local_config_overrides(tmp_path, monkeypatch):
    import jone.config as config

    (tmp_path / "config.yaml").write_text("stt:\n  model: large-v3-turbo\n", encoding="utf-8")
    (tmp_path / "config.local.yaml").write_text("stt:\n  device: cpu\n", encoding="utf-8")
    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    cfg = config.load_config(tmp_path / "config.yaml")
    assert cfg["stt"]["model"] == "large-v3-turbo" and cfg["stt"]["device"] == "cpu"
    assert cfg["stt"]["languages"] == ["fr", "en"]


def _server_stt(responses):
    from jone.stt_server import WhisperServerSTT

    stt = WhisperServerSTT.__new__(WhisperServerSTT)
    stt.language, stt.languages, stt.default_language, stt.language_margin = None, ["fr", "en"], "fr", 0.65
    calls = []

    def fake_request(wav, language, with_probs, audio_s=30.0):
        calls.append(language)
        return responses.pop(0)

    stt._request = fake_request
    return stt, calls


def test_whisper_server_keeps_auto_transcription_when_language_allowed():
    res = {"language_probabilities": {"en": 0.9, "fr": 0.05},
           "segments": [{"text": " Hello there.", "avg_logprob": -0.2, "no_speech_prob": 0.05}]}
    stt, calls = _server_stt([res])
    tr = stt.transcribe(np.zeros(1600, dtype=np.int16))
    assert calls == ["auto"] and tr.language == "en" and tr.text == "Hello there."
    assert tr.avg_logprob == -0.2 and tr.compression_ratio < 2.4


def test_whisper_server_retranscribes_in_allowed_language():
    first = {"language_probabilities": {"nn": 0.42, "en": 0.33, "fr": 0.25},
             "segments": [{"text": " Hei.", "avg_logprob": -0.9, "no_speech_prob": 0.3}]}
    second = {"segments": [{"text": " Bonjour.", "avg_logprob": -0.3, "no_speech_prob": 0.1}]}
    stt, calls = _server_stt([first, second])
    tr = stt.transcribe(np.zeros(1600, dtype=np.int16))
    assert calls == ["auto", "fr"] and tr.text == "Bonjour." and tr.language == "fr"


def test_whisper_server_empty_result():
    stt, _ = _server_stt([{"language_probabilities": {"fr": 0.9}, "segments": []}])
    tr = stt.transcribe(np.zeros(1600, dtype=np.int16))
    assert tr.text == "" and tr.no_speech_prob == 1.0


def test_audio_ctx_has_margin_and_bounds():
    from jone.stt_server import audio_ctx_for

    assert audio_ctx_for(2.85, 2.0, 384) == 384      # court : minimum
    assert audio_ctx_for(10.0, 2.0, 384) == 640      # (10 + 2) s × 50, arrondi à 64
    assert audio_ctx_for(40.0, 2.0, 384) == 1500     # jamais plus de 30 s
