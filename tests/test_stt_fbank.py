"""Tests for :mod:`vrcc.stt.fbank`, the numpy SenseVoice front-end.

Every constant in that module (Kaldi's frame geometry, hamming window,
preemphasis, mel triangles, log floor; FunASR's LFR padding) is a
silent-failure knob: get one wrong and the model still runs, it just
transcribes worse. So this pins the numbers rather than only the shapes.

The pinned values are self-generated, and what anchors them to *correct* is
tests/integration/test_sensevoice_reference.py: it runs this front-end into
the real model and asserts sherpa-onnx's own published transcripts come back.
Treat a diff here as "the front-end changed" and re-run that integration test
before accepting it.
"""

from __future__ import annotations

import numpy as np
import pytest

from vrcc.stt.fbank import apply_cmvn, apply_lfr, fbank

_SAMPLE_RATE = 16000


def _signal(seconds: float = 1.0) -> np.ndarray:
    """A deterministic 440 Hz tone plus seeded noise, in int16 amplitude (the
    scale the SenseVoice export's ``normalize_samples=0`` asks for)."""
    rng = np.random.default_rng(20240717)
    n = int(_SAMPLE_RATE * seconds)
    t = np.arange(n) / _SAMPLE_RATE
    tone = 0.3 * np.sin(2 * np.pi * 440 * t)
    noise = 0.05 * rng.standard_normal(n)
    return ((tone + noise) * 32768.0).astype(np.float32)


# -- fbank -----------------------------------------------------------------


def test_fbank_frame_geometry_is_snip_edges():
    # 25 ms window / 10 ms hop at 16 kHz: 1 + (16000 - 400) // 160 == 98.
    assert fbank(_signal()).shape == (98, 80)


def test_fbank_shorter_than_one_window_yields_no_frames():
    # snip_edges=True drops a partial trailing window rather than padding it,
    # so audio under 25 ms has nothing to transcribe.
    assert fbank(np.zeros(399, dtype=np.float32)).shape == (0, 80)
    assert fbank(np.zeros(400, dtype=np.float32)).shape == (1, 80)


def test_fbank_silence_lands_on_the_log_floor():
    # Digital silence has zero energy in every bin; Kaldi clamps to
    # finfo(float32).eps before the log rather than returning -inf.
    silent = fbank(np.zeros(_SAMPLE_RATE, dtype=np.float32))
    assert silent.shape == (98, 80)
    assert np.allclose(silent, np.log(np.float32(1.1920928955078125e-07)))


def test_fbank_values_are_pinned():
    feats = fbank(_signal())
    assert feats.dtype == np.float32
    assert feats[0, :4] == pytest.approx(
        [13.865433, 13.352506, 13.624375, 14.272738], abs=1e-4
    )
    assert float(feats.sum()) == pytest.approx(154388.14, abs=0.5)


def test_fbank_is_deterministic():
    signal = _signal()
    assert np.array_equal(fbank(signal), fbank(signal))


def test_fbank_amplitude_scale_changes_the_features():
    """The normalize_samples footgun: feeding [-1, 1] audio where the export
    wants int16 scale does not raise, it just shifts every energy. Pinned so a
    refactor that drops the scaling is caught here and not in the field."""
    quiet = fbank(_signal() / 32768.0)
    loud = fbank(_signal())
    assert not np.allclose(quiet, loud)


# -- LFR -------------------------------------------------------------------


def test_apply_lfr_stacks_and_subsamples():
    feats = fbank(_signal())  # (98, 80)
    stacked = apply_lfr(feats, 7, 6)
    # (7 - 1) // 2 == 3 leading pad frames, then ceil(101 / 6) == 17 steps.
    assert stacked.shape == (17, 80 * 7)


def test_apply_lfr_left_pads_with_the_first_frame():
    feats = np.arange(20, dtype=np.float32).reshape(5, 4)
    stacked = apply_lfr(feats, 7, 6)
    # First step is [f0, f0, f0, f0, f1, f2, f3]: three repeats of the first
    # frame, then the real frames.
    assert np.array_equal(stacked[0, 0:4], feats[0])
    assert np.array_equal(stacked[0, 4:8], feats[0])
    assert np.array_equal(stacked[0, 12:16], feats[0])
    assert np.array_equal(stacked[0, 16:20], feats[1])


def test_apply_lfr_right_pads_a_short_final_chunk_with_the_last_frame():
    feats = np.arange(20, dtype=np.float32).reshape(5, 4)
    stacked = apply_lfr(feats, 7, 6)
    assert stacked.shape[0] == 2  # ceil((5 + 3) / 6)
    # The final step runs off the end and repeats the last real frame.
    assert np.array_equal(stacked[-1, -4:], feats[-1])


def test_apply_lfr_of_empty_features_keeps_the_stacked_width():
    empty = apply_lfr(np.zeros((0, 80), dtype=np.float32), 7, 6)
    assert empty.shape == (0, 560)


def test_apply_lfr_values_are_pinned():
    stacked = apply_lfr(fbank(_signal()), 7, 6)
    assert stacked[0, :3] == pytest.approx([13.865433, 13.352506, 13.624375], abs=1e-4)
    assert float(stacked.sum()) == pytest.approx(187499.09, abs=0.5)


# -- CMVN ------------------------------------------------------------------


def test_apply_cmvn_adds_the_negated_mean_then_scales():
    feats = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    neg_mean = np.array([-1.0, -2.0], dtype=np.float32)
    inv_stddev = np.array([2.0, 0.5], dtype=np.float32)
    assert np.array_equal(
        apply_cmvn(feats, neg_mean, inv_stddev),
        np.array([[0.0, 0.0], [4.0, 1.0]], dtype=np.float32),
    )
