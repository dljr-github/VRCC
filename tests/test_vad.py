"""Tests for the streaming Silero VAD wrapper against deterministic synthetic
audio (onnxruntime and the bundled silero_vad_v6.onnx are always available).
"""

from pathlib import Path

import numpy as np
import pytest

from vrcc.audio.vad import StreamingVad, find_silero_onnx

FRAME = 512


def _zeros() -> np.ndarray:
    return np.zeros(FRAME, dtype=np.float32)


def _noise(seed: int = 0) -> np.ndarray:
    # Loud deterministic white noise -- reliably drives the RNN state away
    # from its zero baseline without depending on a real speech sample.
    rng = np.random.default_rng(seed)
    return rng.standard_normal(FRAME).astype(np.float32)


class TestFindSileroOnnx:
    def test_returns_existing_path(self):
        path = find_silero_onnx()
        assert isinstance(path, Path)
        assert path.is_file()
        assert path.suffix == ".onnx"


class TestProbContract:
    def test_prob_returns_python_float_in_unit_interval(self):
        vad = StreamingVad()
        p = vad.prob(_noise())
        assert type(p) is float
        assert 0.0 <= p <= 1.0

    def test_wrong_frame_length_raises_value_error(self):
        vad = StreamingVad()
        with pytest.raises(ValueError):
            vad.prob(np.zeros(FRAME + 1, dtype=np.float32))

    def test_wrong_frame_ndim_raises_value_error(self):
        vad = StreamingVad()
        with pytest.raises(ValueError):
            vad.prob(np.zeros((1, FRAME), dtype=np.float32))

    def test_non_float_input_is_accepted(self):
        # int16-style PCM values should convert cleanly to float32.
        vad = StreamingVad()
        frame = np.zeros(FRAME, dtype=np.int16)
        p = vad.prob(frame)
        assert 0.0 <= p <= 1.0

    def test_frame_constant_is_512(self):
        assert StreamingVad.FRAME == 512


class TestSilence:
    def test_zeros_after_warmup_is_low_probability(self):
        vad = StreamingVad()
        # A few warmup frames to settle the recurrent state.
        for _ in range(5):
            p = vad.prob(_zeros())
        assert p < 0.2


class TestStatefulness:
    def test_state_changes_after_nonzero_frame(self):
        vad = StreamingVad()
        baseline = vad._state.copy()
        vad.prob(_noise())
        assert not np.array_equal(vad._state, baseline)

    def test_context_persists_across_calls(self):
        vad = StreamingVad()
        vad.prob(_noise(1))
        assert not np.array_equal(vad._context, np.zeros_like(vad._context))

    def test_reset_restores_baseline_behavior(self):
        fresh = StreamingVad()
        fresh_prob = fresh.prob(_zeros())

        used = StreamingVad()
        for _ in range(3):
            used.prob(_noise())
        used.reset()
        after_reset = used.prob(_zeros())

        assert abs(after_reset - fresh_prob) < 1e-6

    def test_reset_zeros_all_state(self):
        vad = StreamingVad()
        for _ in range(3):
            vad.prob(_noise())
        vad.reset()
        assert np.array_equal(vad._state, np.zeros_like(vad._state))
        assert np.array_equal(vad._context, np.zeros_like(vad._context))

    def test_history_affects_probability(self):
        # The same frame should generally produce a different probability
        # depending on preceding audio, proving state actually feeds forward.
        primed = StreamingVad()
        for _ in range(3):
            primed.prob(_noise())
        primed_prob = primed.prob(_zeros())

        cold = StreamingVad()
        cold_prob = cold.prob(_zeros())

        assert primed_prob != cold_prob

    def test_caller_buffer_mutation_does_not_corrupt_context(self):
        # Regression: audio capture loops reuse their buffer in place. The
        # wrapper must copy the frame tail it keeps as context; if it stored
        # a view, mutating the buffer after prob() would silently change the
        # "previous context" seen by the next inference.
        frame1 = _noise(0)
        frame2 = _noise(1)

        control = StreamingVad()
        control.prob(frame1.copy())
        expected = control.prob(frame2.copy())

        vad = StreamingVad()
        buffer = frame1.copy()
        vad.prob(buffer)  # float32 input: wrapper sees this exact array
        buffer[:] = 999.0  # capture loop overwrites the buffer in place
        result = vad.prob(frame2.copy())

        assert result == expected


class TestBatchEquivalence:
    def test_streaming_matches_faster_whisper_batch(self):
        # The wrapper must reproduce faster-whisper's SileroVADModel contract
        # exactly. Caveats handled below, verified against the installed
        # faster_whisper/vad.py (1.2.1):
        #   * upstream zeros the batch's FINAL frame's 64-sample tail in
        #     place (`context[-1] = 0` before np.roll), so its final frame's
        #     probability is NOT a true streaming value -- we append a
        #     throwaway trailing frame so every compared frame is non-final;
        #   * that same in-place zeroing mutates the caller's audio array,
        #     so the batch model gets its own copy.
        from faster_whisper.vad import SileroVADModel

        from vrcc.audio.vad import find_silero_onnx

        n_frames = 8
        rng = np.random.default_rng(42)
        audio = (rng.standard_normal(FRAME * n_frames) * 0.3).astype(np.float32)

        batch_input = np.concatenate([audio, np.zeros(FRAME, dtype=np.float32)])
        batch = SileroVADModel(str(find_silero_onnx()))
        batch_probs = batch(batch_input.copy()).reshape(-1)[:n_frames]

        vad = StreamingVad()
        stream_probs = np.array(
            [
                vad.prob(audio[i * FRAME : (i + 1) * FRAME].copy())
                for i in range(n_frames)
            ]
        )

        np.testing.assert_allclose(stream_probs, batch_probs, atol=1e-6)
