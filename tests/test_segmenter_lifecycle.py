"""Tests for the Segmenter utterance-boundary state machine (min/max
utterance guards, frame copy-safety, hysteresis dead-band, config counts).
"""

from __future__ import annotations

import numpy as np
import pytest

from vrcc.audio.segmenter import (
    FRAME,
    HYSTERESIS_GAP,
    SegDiscard,
    SegFinal,
    SegLevel,
    SegSpeculative,
    SegSpeechStart,
    Segmenter,
)
from vrcc.core.config import VadConfig


class ScriptedVad:
    """Pops one scripted probability per call; errors loudly if exhausted."""

    def __init__(self, probs: list[float]) -> None:
        self._probs = list(probs)
        self.calls = 0

    def __call__(self, frame: np.ndarray) -> float:
        self.calls += 1
        if not self._probs:
            raise AssertionError(
                f"ScriptedVad exhausted after {self.calls} calls but process() "
                "was called again"
            )
        return self._probs.pop(0)


def _frame(value: float = 0.1) -> np.ndarray:
    return np.full(FRAME, value, dtype=np.float32)


def _by_type(events: list[object], cls: type) -> list[object]:
    return [e for e in events if isinstance(e, cls)]


class TestMinUtteranceRejection:
    def test_too_short_utterance_emits_no_final_but_resets(self):
        # Deliberately tiny speculative/finalize thresholds (1 and 2 frames)
        # with the default min_utterance_ms (16 frames) so the finalize
        # threshold is reached long before the utterance is "long enough",
        # isolating the min-utterance guard in behavior 5.
        cfg = VadConfig(
            speculative_silence_ms=32,   # 1 frame
            finalize_silence_ms=64,      # 2 frames
            min_utterance_ms=500,        # 16 frames
            pre_roll_ms=0,
        )
        vad = ScriptedVad([0.9, 0.1, 0.1, 0.9])
        seg = Segmenter(cfg, vad)

        e1 = seg.process(_frame())  # speech start, frames_since_start=1
        e2 = seg.process(_frame())  # silence #1 -> speculative (threshold=1)
        e3 = seg.process(_frame())  # silence #2 -> finalize check: 3 < 16 -> no Final, reset
        e4 = seg.process(_frame())  # speech again -> new SpeechStart, utterance_id=2

        assert _by_type(e1, SegSpeechStart)[0].utterance_id == 1
        assert len(_by_type(e2, SegSpeculative)) == 1
        assert _by_type(e3, SegFinal) == []
        # The pending speculative must not be orphaned: the too-short reset
        # resolves it with a SegDiscard.
        assert len(_by_type(e3, SegDiscard)) == 1
        starts4 = _by_type(e4, SegSpeechStart)
        assert len(starts4) == 1
        assert starts4[0].utterance_id == 2

    def test_long_enough_utterance_does_emit_final(self):
        cfg = VadConfig(
            speculative_silence_ms=32,
            finalize_silence_ms=64,
            min_utterance_ms=32,  # 1 frame -- trivially satisfied
            pre_roll_ms=0,
        )
        vad = ScriptedVad([0.9, 0.1, 0.1])
        seg = Segmenter(cfg, vad)
        seg.process(_frame())
        seg.process(_frame())
        events = seg.process(_frame())
        assert len(_by_type(events, SegFinal)) == 1

    def test_too_short_finalize_discards_pending_speculative(self):
        # Invariant: every SegSpeculative is resolved by exactly one
        # SegFinal or SegDiscard. With min_utterance_ms > finalize_silence_ms
        # (min 1000ms -> 32 frames; finalize 400ms -> 13 frames; speculative
        # 350ms -> 11 frames), one speech frame plus 13 silence frames emits
        # a speculative at the 11th silence frame, then hits the finalize
        # threshold while still too short: no SegFinal may fire, but the
        # in-flight speculative MUST be resolved with a SegDiscard so the
        # downstream STT worker drops the job.
        cfg = VadConfig(
            min_utterance_ms=1000,
            finalize_silence_ms=400,
            speculative_silence_ms=350,
        )
        vad = ScriptedVad([0.9] + [0.1] * 13)
        seg = Segmenter(cfg, vad)

        timeline = []  # (frame_index, event_type_name)
        for i in range(14):
            events = seg.process(_frame())
            for e in events:
                if not isinstance(e, SegLevel):
                    timeline.append((i, type(e).__name__))

        assert timeline == [
            (0, "SegSpeechStart"),
            (11, "SegSpeculative"),   # 11th silence frame
            (13, "SegDiscard"),       # finalize threshold, too short
        ]
        assert seg._active is False
        assert seg._utterance_id == 2  # reset still increments


class TestMaxUtteranceForceFinal:
    def test_force_final_at_max_utterance_frames(self):
        # max_utterance_s tuned small (11 frames) with continuous speech
        # (no silence at all) so the only possible trigger is the max-
        # duration guard, not the silence-based finalize path.
        cfg = VadConfig(pre_roll_ms=0, max_utterance_s=11 * 32 / 1000)
        vad = ScriptedVad([0.9] * 11)
        seg = Segmenter(cfg, vad)

        finals = []
        for _ in range(11):
            events = seg.process(_frame())
            finals.extend(_by_type(events, SegFinal))

        assert len(finals) == 1
        assert finals[0].samples.shape[0] == 11 * FRAME

    def test_no_force_final_one_frame_before_max(self):
        cfg = VadConfig(pre_roll_ms=0, max_utterance_s=11 * 32 / 1000)
        vad = ScriptedVad([0.9] * 10)
        seg = Segmenter(cfg, vad)
        finals = []
        for _ in range(10):
            events = seg.process(_frame())
            finals.extend(_by_type(events, SegFinal))
        assert finals == []

    def test_force_final_on_speech_start_frame_when_preroll_exceeds_max(self):
        # Degenerate config: pre-roll (6 frames) larger than the max
        # utterance cap (5 frames). The buffer is already over the cap on
        # the very IDLE->ACTIVE transition frame, so behavior 6 must force
        # SegFinal in the SAME process() call as SegSpeechStart -- not one
        # frame later (which would overshoot the configured cap).
        cfg = VadConfig(
            pre_roll_ms=192,                 # 6 frames
            max_utterance_s=5 * 32 / 1000,   # 5 frames
        )
        vad = ScriptedVad([0.1] * 6 + [0.9])
        seg = Segmenter(cfg, vad)
        for _ in range(6):
            seg.process(_frame())
        events = seg.process(_frame())

        assert len(_by_type(events, SegSpeechStart)) == 1
        finals = _by_type(events, SegFinal)
        assert len(finals) == 1
        assert finals[0].samples.shape[0] == 7 * FRAME  # 6 pre-roll + trigger
        assert seg._active is False
        assert seg._utterance_id == 2

    def test_force_final_reuses_pending_speculative_identity(self):
        # Behavior 8 must hold on the force-final path too: a speculative
        # fires (1 silence frame with speculative_silence_ms=32), then only
        # dead-band frames (0.4 -- NOT speech) pad the buffer to the max;
        # no speech frame occurred between speculative and final, so the
        # forced SegFinal must reuse the speculative array by identity.
        cfg = VadConfig(
            pre_roll_ms=0,
            speculative_silence_ms=32,   # 1 frame
            finalize_silence_ms=64_000,  # unreachably large
            min_utterance_ms=32,
            max_utterance_s=6 * 32 / 1000,  # 6 frames
        )
        vad = ScriptedVad([0.9, 0.9, 0.9, 0.1, 0.4, 0.4])
        seg = Segmenter(cfg, vad)

        spec = None
        final = None
        for _ in range(6):
            events = seg.process(_frame())
            for e in _by_type(events, SegSpeculative):
                spec = e
            for e in _by_type(events, SegFinal):
                final = e

        assert spec is not None
        assert final is not None
        assert final.samples is spec.samples


class TestFrameCopySafety:
    def test_caller_buffer_reuse_does_not_corrupt_stored_audio(self):
        # Regression guard for the Task 6 aliasing class of bug: audio
        # capture loops reuse one buffer in place. The segmenter must copy
        # each frame it stores, so mutating the caller's buffer after
        # process() returns must not change previously buffered audio.
        cfg = VadConfig(
            pre_roll_ms=0,
            # speculative disabled (unreachably large) so SegFinal is built
            # from the full buffer, not an earlier speculative snapshot.
            speculative_silence_ms=64_000,
            finalize_silence_ms=64,
            min_utterance_ms=32,
        )
        vad = ScriptedVad([0.9, 0.9, 0.1, 0.1])
        seg = Segmenter(cfg, vad)

        buffer = np.full(FRAME, 0.5, dtype=np.float32)
        expected_values = [0.5, 0.25, 0.125, 0.0625]

        final = None
        for value in expected_values[1:] + [None]:
            events = seg.process(buffer)
            for e in _by_type(events, SegFinal):
                final = e
            if value is not None:
                buffer[:] = value  # capture loop overwrites its buffer

        assert final is not None
        assert final.samples.shape[0] == 4 * FRAME
        for i, value in enumerate(expected_values):
            chunk = final.samples[i * FRAME : (i + 1) * FRAME]
            assert np.all(chunk == np.float32(value)), f"frame {i} corrupted"

    def test_final_samples_content_is_preroll_plus_utterance(self):
        # Content check: with pre_roll_ms=64 (2 frames), feed 2 distinct
        # idle frames, then speech/silence frames with distinct values; the
        # final buffer must be exactly [idle1, idle2, speech..., silence...]
        # in order.
        cfg = VadConfig(
            pre_roll_ms=64,               # 2 frames
            speculative_silence_ms=64_000,  # disabled: final = full buffer
            finalize_silence_ms=64,       # 2 frames
            min_utterance_ms=32,
        )
        vad = ScriptedVad([0.1, 0.1, 0.9, 0.9, 0.2, 0.2])
        seg = Segmenter(cfg, vad)

        values = [0.01, 0.02, 0.6, 0.7, 0.03, 0.04]
        final = None
        for v in values:
            events = seg.process(_frame(v))
            for e in _by_type(events, SegFinal):
                final = e

        assert final is not None
        assert final.samples.shape[0] == 6 * FRAME
        for i, v in enumerate(values):
            chunk = final.samples[i * FRAME : (i + 1) * FRAME]
            assert np.all(chunk == np.float32(v)), f"frame {i} out of order"


class TestHysteresisDeadBand:
    def test_dead_band_frames_do_not_move_silence_run_while_active(self):
        cfg = VadConfig()
        # speech start, 5 silence frames, 3 dead-band frames (0.4, between
        # 0.35 and 0.5), then 6 more silence frames -> speculative should
        # fire at the 11th *classified-silence* frame (5 + 6), unaffected
        # by the 3 dead-band frames in between.
        probs = [0.9] + [0.1] * 5 + [0.4] * 3 + [0.1] * 6
        vad = ScriptedVad(probs)
        seg = Segmenter(cfg, vad)

        for _ in range(1 + 5):
            seg.process(_frame())
        assert seg._silence_run == 5

        for _ in range(3):
            events = seg.process(_frame())
            assert _by_type(events, SegSpeculative) == []
            assert _by_type(events, SegDiscard) == []
        assert seg._silence_run == 5  # untouched by dead-band frames

        spec_at = None
        for i in range(6):
            events = seg.process(_frame())
            if _by_type(events, SegSpeculative):
                spec_at = i
        assert spec_at == 5  # 6th of these frames (0-based index 5)

    def test_dead_band_frames_do_not_trigger_speech_start_while_idle(self):
        cfg = VadConfig()
        vad = ScriptedVad([0.4] * 4)
        seg = Segmenter(cfg, vad)
        for _ in range(4):
            events = seg.process(_frame())
            assert _by_type(events, SegSpeechStart) == []
        assert seg._active is False

    def test_vad_exactly_at_threshold_is_speech(self):
        # Boundary: speech is `vad >= threshold`, inclusive. 0.5 is exactly
        # representable in binary floating point, so this is a stable check.
        cfg = VadConfig()
        vad = ScriptedVad([cfg.threshold])
        seg = Segmenter(cfg, vad)
        events = seg.process(_frame())
        assert len(_by_type(events, SegSpeechStart)) == 1

    def test_vad_exactly_at_lower_bound_is_dead_band(self):
        # Boundary: silence is STRICTLY below `threshold - 0.15`. Feed the
        # exact same float expression the implementation computes, so the
        # comparison is x < x == False regardless of float representation.
        cfg = VadConfig()
        boundary = cfg.threshold - HYSTERESIS_GAP
        vad = ScriptedVad([0.9, boundary, boundary, boundary - 1e-6])
        seg = Segmenter(cfg, vad)
        seg.process(_frame())  # speech start
        seg.process(_frame())  # exactly at bound -> dead band
        seg.process(_frame())  # exactly at bound -> dead band
        assert seg._silence_run == 0
        seg.process(_frame())  # just below bound -> silence
        assert seg._silence_run == 1


class TestConfigFrameCounts:
    def test_default_config_frame_counts_match_spec(self):
        seg = Segmenter(VadConfig(), ScriptedVad([]))
        assert seg._speculative_frames == 11
        assert seg._finalize_frames == 19
        assert seg._min_utterance_frames == 16
        assert seg._preroll_frames == 5
        assert seg._max_utterance_frames == 875


class TestReconfigure:
    # frame_ms = 32 at 16 kHz/512; these ms values round to distinct frame
    # counts so a stale threshold would be caught.
    _NEW = dict(
        speculative_silence_ms=64,   # 2 frames
        finalize_silence_ms=128,     # 4 frames
        min_utterance_ms=96,         # 3 frames
        pre_roll_ms=96,              # 3 frames
        max_utterance_s=1.6,         # 50 frames
    )

    def test_reconfigure_recomputes_every_frame_count(self):
        seg = Segmenter(VadConfig(), ScriptedVad([]))
        seg.reconfigure(VadConfig(**self._NEW))
        assert seg._speculative_frames == 2
        assert seg._finalize_frames == 4
        assert seg._min_utterance_frames == 3
        assert seg._preroll_frames == 3
        assert seg._max_utterance_frames == 50

    def test_reconfigure_updates_cfg_so_threshold_applies(self):
        # Idle before/after; the new (higher) threshold must gate a start.
        seg = Segmenter(VadConfig(threshold=0.5), ScriptedVad([0.6, 0.9]))
        seg.reconfigure(VadConfig(threshold=0.8))
        assert seg.cfg.threshold == 0.8
        assert _by_type(seg.process(_frame()), SegSpeechStart) == []  # 0.6 < 0.8
        assert len(_by_type(seg.process(_frame()), SegSpeechStart)) == 1  # 0.9

    def test_reconfigure_does_not_disturb_in_flight_utterance(self):
        seg = Segmenter(VadConfig(pre_roll_ms=0), ScriptedVad([0.9, 0.9]))
        seg.process(_frame())  # speech start: ACTIVE, utterance 1
        seg.process(_frame())  # still speaking
        assert seg._active is True
        seg.reconfigure(VadConfig(**self._NEW))
        # In-flight state is untouched: same utterance, same buffered audio.
        assert seg._active is True
        assert seg._utterance_id == 1
        assert seg._frames_since_start == 2
        assert len(seg._buffer) == 2

    def test_reconfigure_resizes_preroll_ring_only_on_change(self):
        seg = Segmenter(VadConfig(pre_roll_ms=150), ScriptedVad([]))
        assert seg._preroll.maxlen == 5
        ring = seg._preroll
        # Same pre_roll_ms: ring object is preserved (accumulated frames kept).
        seg.reconfigure(VadConfig(pre_roll_ms=150, finalize_silence_ms=800))
        assert seg._preroll is ring
        # Changed pre_roll_ms: ring resized to the new maxlen.
        seg.reconfigure(VadConfig(pre_roll_ms=96))
        assert seg._preroll.maxlen == 3

    def test_reconfigured_timings_take_effect_next_utterance(self):
        # Default finalize is 19 frames; drop it to 2 so one speech + two
        # silence frames finalizes, proving the new timing is live.
        seg = Segmenter(VadConfig(pre_roll_ms=0), ScriptedVad([0.9, 0.1, 0.1]))
        seg.reconfigure(
            VadConfig(
                pre_roll_ms=0,
                speculative_silence_ms=32,
                finalize_silence_ms=64,
                min_utterance_ms=32,
            )
        )
        finals = []
        for _ in range(3):
            finals.extend(_by_type(seg.process(_frame()), SegFinal))
        assert len(finals) == 1
