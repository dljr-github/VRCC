"""Tests for the Segmenter's periodic live-partial emit (SegPartial), split out
from test_segmenter.py to stay under the 500-line cap.
"""

from __future__ import annotations

import numpy as np

from vrcc.audio.segmenter import (
    FRAME,
    SegDiscard,
    SegFinal,
    SegPartial,
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


class TestLivePartials:
    def test_sustained_speech_emits_partial_about_every_partial_frames(self):
        # partial_interval_ms=64 -> 2 frames at 32ms/frame. Sustained speech
        # (never entering silence, so speculative/final never fire) should
        # emit a SegPartial roughly every 2 frames.
        cfg = VadConfig(live_partials=True, partial_interval_ms=64)
        vad = ScriptedVad([0.9] * 10)
        seg = Segmenter(cfg, vad)
        assert seg._partial_frames == 2

        partial_frames = []
        for i in range(10):
            events = seg.process(_frame())
            if _by_type(events, SegPartial):
                partial_frames.append(i)

        # Speech starts on frame 0 (index 0); buffer len must exceed preroll
        # frames (5, default) before a partial can fire, so the first partial
        # cannot land before the buffer has grown past that. With a 2-frame
        # interval and preroll=5, partials land at indices 5, 7, 9 (buffer
        # lengths 6, 8, 10 -- all > 5).
        assert partial_frames == [5, 7, 9]

    def test_partial_events_carry_utterance_id_and_snapshot_samples(self):
        cfg = VadConfig(live_partials=True, partial_interval_ms=64, pre_roll_ms=0)
        vad = ScriptedVad([0.9] * 6)
        seg = Segmenter(cfg, vad)
        assert seg._preroll_frames == 0

        partials = []
        for _ in range(6):
            events = seg.process(_frame())
            partials.extend(_by_type(events, SegPartial))

        assert len(partials) >= 1
        p = partials[0]
        assert p.utterance_id == 1
        assert isinstance(p.samples, np.ndarray)
        assert p.samples.dtype == np.float32
        assert p.samples.ndim == 1

    def test_live_partials_false_never_emits(self):
        cfg = VadConfig(live_partials=False, partial_interval_ms=64)
        vad = ScriptedVad([0.9] * 20)
        seg = Segmenter(cfg, vad)

        partials = []
        for _ in range(20):
            events = seg.process(_frame())
            partials.extend(_by_type(events, SegPartial))

        assert partials == []

    def test_partial_counter_restarts_after_reset_to_idle(self):
        # Utterance 1 is too short for its silence-run counter to ever reach
        # partial_frames (3) before finalize fires (2 silence frames), so it
        # never emits a partial. Utterance 2 then runs 5 sustained speech
        # frames; if the counter carried over stale state instead of being
        # reset by _reset_to_idle, the first partial would land earlier than
        # 3 ACTIVE frames after utterance 2's speech start.
        cfg = VadConfig(
            live_partials=True,
            partial_interval_ms=96,   # 3 frames
            speculative_silence_ms=64_000,  # disabled
            finalize_silence_ms=64,   # 2 frames
            min_utterance_ms=32,
            pre_roll_ms=0,
        )
        probs = [0.9, 0.1, 0.1] + [0.9] * 5
        vad = ScriptedVad(probs)
        seg = Segmenter(cfg, vad)
        assert seg._partial_frames == 3

        starts = []
        partial_frames = []
        for i in range(len(probs)):
            events = seg.process(_frame())
            partial_frames.extend([i] * len(_by_type(events, SegPartial)))
            starts.extend(_by_type(events, SegSpeechStart))

        assert len(starts) == 2  # utterance 1 and utterance 2 both started
        # Utterance 2 starts at index 3; the sole partial lands 3 ACTIVE
        # frames later (index 6), not immediately.
        assert partial_frames == [6]


class TestPartialDiscard:
    def test_abort_with_partial_emitted_returns_discard(self):
        # A live partial was emitted (a LISTENING row exists downstream) but no
        # speculative is pending; abort must still return a SegDiscard so the
        # row is cleared, not left stuck listening.
        cfg = VadConfig(
            live_partials=True,
            partial_interval_ms=64,          # 2 frames
            speculative_silence_ms=64_000,   # disabled
            finalize_silence_ms=64_000,      # disabled
            pre_roll_ms=0,
        )
        vad = ScriptedVad([0.9] * 6)
        seg = Segmenter(cfg, vad)
        partials = []
        for _ in range(6):
            partials.extend(_by_type(seg.process(_frame()), SegPartial))

        assert partials  # a partial was emitted
        assert seg._pending_spec_samples is None  # yet no speculative pending
        # Abort ends the utterance: the discard must be terminal so the
        # LISTENING row is cleared, not left waiting for a SegPartial that
        # will never come.
        assert seg.abort() == [SegDiscard(utterance_id=1, terminal=True)]

    def test_too_short_finalize_with_partial_emitted_returns_discard(self):
        # A partial was emitted, then silence finalizes while the utterance is
        # still under the minimum length. No SegFinal fires, so the LISTENING
        # row is cleared with a SegDiscard instead.
        cfg = VadConfig(
            live_partials=True,
            partial_interval_ms=64,          # 2 frames
            speculative_silence_ms=64_000,   # disabled
            finalize_silence_ms=64,          # 2 frames
            min_utterance_ms=64_000,         # never long enough for a final
            pre_roll_ms=0,
        )
        vad = ScriptedVad([0.9] * 4 + [0.1] * 2)
        seg = Segmenter(cfg, vad)
        events = []
        for _ in range(6):
            events.extend(seg.process(_frame()))

        assert _by_type(events, SegPartial)   # a partial was emitted
        assert not _by_type(events, SegFinal)  # too short: no final
        # The utterance ends here (too short to finalize): terminal, so the
        # LISTENING row is cleared.
        assert _by_type(events, SegDiscard) == [SegDiscard(utterance_id=1, terminal=True)]


class TestDiscardTerminalFlag:
    def test_speech_resume_discard_is_not_terminal(self):
        # Speculative fires during a pause, then speech resumes mid-utterance:
        # the SAME utterance continues (more SegPartials will follow and keep
        # updating the same row), so this discard must NOT be terminal.
        cfg = VadConfig()
        probs = [0.9] * 5 + [0.1] * 12 + [0.9] * 5
        vad = ScriptedVad(probs)
        seg = Segmenter(cfg, vad)

        discards = []
        for _ in range(22):
            discards.extend(_by_type(seg.process(_frame()), SegDiscard))

        assert discards == [SegDiscard(utterance_id=1, terminal=False)]
