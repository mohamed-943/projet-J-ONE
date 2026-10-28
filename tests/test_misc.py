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
