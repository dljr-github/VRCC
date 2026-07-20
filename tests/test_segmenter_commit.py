"""Tests for the Segmenter's commit-path pre-roll handling: the retained ring
is trimmed to the commit window after a commit (M4), the commit-consuming
frame is kept so the seam into the next utterance is contiguous, and a normal
finalize does not trim. Split from test_segmenter.py to stay under the line cap.
"""

from __future__ import annotations

import numpy as np

from vrcc.audio.segmenter import FRAME, SegFinal, Segmenter
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


class TestCommitPreroll:
    def test_commit_trims_retained_preroll_to_commit_window(self):
        # pre_roll 400ms (13 frames) > speculative 250ms (8). After a commit
        # the retained ring must be trimmed to the commit window (<=
        # speculative), so the just-committed sentence's tail cannot prepend
        # onto the next utterance (M4), while the most recent onset is kept.
        cfg = VadConfig(pre_roll_ms=400, speculative_silence_ms=250)
        vad = ScriptedVad([0.9] * 16)
        seg = Segmenter(cfg, vad)
        for _ in range(15):
            seg.process(_frame(0.3))  # fill the ring past 13 frames
        assert len(seg._preroll) == 13
        seg.request_commit(1)
        seg.process(_frame(0.77))  # commit consumed on this frame
        assert not seg.active
        assert seg._commit_preroll_frames == 8
        assert len(seg._preroll) == 8  # trimmed to the commit window
        # The trim keeps the MOST RECENT frames: the commit frame is last.
        assert np.all(list(seg._preroll)[-1] == np.float32(0.77))

    def test_commit_frame_is_kept_in_preroll(self):
        # The commit-consuming frame must join the ring so the seam into the
        # next utterance is contiguous (no 32ms hole at the sentence boundary).
        cfg = VadConfig()
        vad = ScriptedVad([0.9, 0.9])
        seg = Segmenter(cfg, vad)
        seg.process(_frame(0.4))  # speech start, utterance 1 active
        seg.request_commit(1)
        seg.process(_frame(0.55))  # commit consumed here
        assert not seg.active
        assert any(np.all(f == np.float32(0.55)) for f in seg._preroll)

    def test_committed_resume_inherits_only_the_commit_window(self):
        # End to end: after a commit the next utterance's onset seed must not
        # inherit more than the commit window of prior audio (M4 preserved).
        cfg = VadConfig(pre_roll_ms=400, speculative_silence_ms=250)
        vad = ScriptedVad([0.9] * 16 + [0.9])
        seg = Segmenter(cfg, vad)
        for _ in range(15):
            seg.process(_frame(0.3))
        seg.request_commit(1)
        seg.process(_frame(0.77))  # commit consumed, ring trimmed to 8
        events = seg.process(_frame(0.9))  # next utterance's speech start
        assert _by_type(events, SegFinal) == []
        assert seg.active
        # 8 pre-roll frames (commit window) + the triggering frame.
        assert len(seg._buffer) == 9

    def test_normal_finalize_does_not_trim_preroll(self):
        # A normal (silence) finalize must NOT trim the ring to the commit
        # window; only a commit trims. The speculative window (8) is >= the
        # finalize window (2), so no speculative fires and the ring stays full.
        cfg = VadConfig(
            pre_roll_ms=400,               # 13 frames
            speculative_silence_ms=250,    # 8 frames -> commit window 8
            finalize_silence_ms=64,        # 2 frames
            min_utterance_ms=32,
        )
        vad = ScriptedVad([0.9] * 15 + [0.1] * 2)
        seg = Segmenter(cfg, vad)
        finals = []
        for _ in range(17):
            finals.extend(_by_type(seg.process(_frame()), SegFinal))
        assert len(finals) == 1  # a normal finalize fired
        assert seg._commit_preroll_frames == 8
        assert len(seg._preroll) > seg._commit_preroll_frames  # not trimmed
