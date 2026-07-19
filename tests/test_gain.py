"""Gain processor: fixed dB boost with soft clipping, and a smoothed auto-level."""

from __future__ import annotations

import math

import numpy as np
import pytest

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


def test_empty_frame_does_not_change_auto_gain():
    g = GainProcessor()
    g.configure(0.0, auto=True)
    quiet = np.full(512, 0.02, dtype=np.float32)
    for _ in range(50):
        g.process(quiet)
    gain_before = g._auto_gain
    empty = np.zeros(0, dtype=np.float32)
    out = g.process(empty)
    assert out.size == 0
    assert g._auto_gain == gain_before
    assert math.isfinite(g._auto_gain)


def test_configure_publishes_auto_and_fixed_linear_as_one_atomic_read():
    # configure() must publish (auto, fixed_linear) as a single reference so
    # process() never reads a mix of the old auto flag and the new gain (or
    # vice versa). A stored tuple identity swap is the simplest thing that
    # guarantees this under the GIL.
    g = GainProcessor()
    g.configure(6.0206, auto=False)  # ~2x
    params_after_first = g._params
    g.configure(0.0, auto=True)
    assert g._params is not params_after_first
    auto, fixed_linear = g._params
    assert auto is True
    assert fixed_linear == pytest.approx(1.0)


def test_empty_frame_then_normal_frame_still_works():
    g = GainProcessor()
    g.configure(0.0, auto=True)
    empty = np.zeros(0, dtype=np.float32)
    g.process(empty)
    assert math.isfinite(g._auto_gain)
    quiet = np.full(512, 0.02, dtype=np.float32)
    last = quiet
    for _ in range(200):
        last = g.process(quiet)
    assert math.isfinite(g._auto_gain)
    assert not np.any(np.isnan(last))
    assert _rms(last) > _rms(quiet) * 2
