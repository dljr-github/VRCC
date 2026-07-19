"""Gain processor: fixed dB boost with soft clipping, and a smoothed auto-level."""

from __future__ import annotations

import numpy as np

from vrcc.audio.gain import GainProcessor


def _rms(x):
    return float(np.sqrt(np.mean(x ** 2)))


def test_zero_db_is_identity():
    g = GainProcessor()
    g.configure(0.0, auto=False)
    frame = np.full(512, 0.1, dtype=np.float32)
    out = g.process(frame)
    assert np.allclose(out, frame)


def test_fixed_gain_scales_by_decibels():
    g = GainProcessor()
    g.configure(6.0206, auto=False)  # ~2x
    frame = np.full(512, 0.1, dtype=np.float32)
    out = g.process(frame)
    assert _rms(out) > 0.19 and _rms(out) < 0.21


def test_output_never_exceeds_unity():
    g = GainProcessor()
    g.configure(30.0, auto=False)
    frame = np.full(512, 0.5, dtype=np.float32)
    out = g.process(frame)
    assert np.max(np.abs(out)) <= 1.0 + 1e-6


def test_auto_gain_raises_a_quiet_signal_over_time():
    g = GainProcessor()
    g.configure(0.0, auto=True)
    quiet = np.full(512, 0.02, dtype=np.float32)
    last = quiet
    for _ in range(200):
        last = g.process(quiet)
    assert _rms(last) > _rms(quiet) * 2


def test_auto_gain_holds_on_silence():
    g = GainProcessor()
    g.configure(0.0, auto=True)
    near_silence = np.full(512, 0.0005, dtype=np.float32)
    out = g.process(near_silence)
    for _ in range(200):
        out = g.process(near_silence)
    # Below the noise floor, gain must not pump silence up toward the target.
    assert _rms(out) < 0.05
