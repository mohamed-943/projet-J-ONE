import numpy as np

from jone.vad import SpeechSegmenter

FRAME = 512  # 32 ms


class ScriptedVAD:
    """Probabilité de parole lue dans l'amplitude de la trame (doublure de Silero)."""

    resets = 0

    def __call__(self, frame):
        return float(frame[0]) / 100.0

    def reset(self):
        self.resets += 1


def frames(prob, n):
    return [np.full(FRAME, int(prob * 100), dtype=np.int16) for _ in range(n)]


def run(seg, seq):
    out = []
    for f in seq:
        u = seg.process(f)
        if u is not None:
            out.append(u)
    return out


def make(**kw):
    params = dict(min_speech_ms=250, min_silence_ms=320, pre_roll_ms=96, max_utterance_s=5)
    params.update(kw)
    return SpeechSegmenter(ScriptedVAD(), **params)


def test_detects_utterance_with_pre_roll_and_trims_silence():
    seq = frames(0.0, 10) + frames(0.9, 20) + frames(0.0, 10)
    utts = run(make(), seq)
    assert len(utts) == 1
    # 3 trames de pré-enregistrement + 20 trames de parole, silence final retiré
    assert len(utts[0]) == (3 + 20) * FRAME


def test_ignores_short_noise_bursts():
    seq = frames(0.0, 5) + frames(0.9, 3) + frames(0.0, 15)  # 96 ms < 250 ms
    assert run(make(), seq) == []


def test_short_pauses_do_not_split_utterance():
    seq = frames(0.9, 10) + frames(0.1, 5) + frames(0.9, 10) + frames(0.0, 12)
    assert len(run(make(), seq)) == 1


def test_hysteresis_keeps_mid_probabilities_as_speech():
    seq = frames(0.9, 10) + frames(0.4, 30) + frames(0.0, 12)
    utts = run(make(), seq)
    assert len(utts) == 1 and len(utts[0]) >= 40 * FRAME


def test_max_duration_forces_end():
    seg = make(max_utterance_s=1.0)  # ~31 trames
    utts = run(seg, frames(0.9, 70))
    assert len(utts) == 2


def test_reset_clears_state():
    seg = make()
    run(seg, frames(0.9, 5))
    assert seg.in_speech
    seg.reset()
    assert not seg.in_speech
