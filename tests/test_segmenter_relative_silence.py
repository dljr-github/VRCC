"""Tests for VadConfig.relative_silence_ratio: a sustained fall from the
in-utterance VAD peak counts as silence even while the absolute reading
stays above both threshold and silence_threshold (a loud room that never
dips, e.g. background babble). Gated by Segmenter._update_room_floor's
running estimate of the room's own VAD reading (see TestRoomFloorGate), so
every scenario here meant to exercise the relative test first earns that
gate open with a room preamble: idle frames always count as room evidence,
and once ACTIVE a frame counts unless it is within _PEAK_MARGIN of the
in-utterance peak -- so a preamble that stays below its own leading peak
opens the gate the same way real fluctuating babble does. Split out of
test_segmenter.py, which was at the 500-line cap.
"""

from __future__ import annotations

import numpy as np

from vrcc.audio.segmenter import (
    FRAME,
    _FLOOR_WARM_FRAMES,
    SegDiscard,
    SegFinal,
    SegSpeculative,
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


def _run(seg: Segmenter, count: int) -> list[object]:
    events: list[object] = []
    for _ in range(count):
        events += seg.process(_frame())
    return events


def _room_preamble(vad_probs: list[float], count: int = _FLOOR_WARM_FRAMES) -> list[float]:
    """A leading 0.9 frame (sets the peak; also the very first, idle, frame,
    so it counts as room evidence unconditionally) followed by ``count``
    frames at 0.6 -- below that peak by more than _PEAK_MARGIN (0.9*0.85 =
    0.765), so every one of them counts too, the same way a real room's own
    fluctuation (babble never reads perfectly flat) keeps most of its frames
    under its own peak. Prepended to ``vad_probs``."""
    return [0.9] + [0.6] * count + vad_probs


class TestRelativeSilence:
    def test_sustained_relative_drop_finalizes_where_absolute_alone_would_not(self):
        # Room preamble opens the gate; peak re-established at 0.9 by 20 real-
        # speech frames, then 19 frames (finalize_frames) at 0.6: above
        # threshold(0.35) and silence_threshold(0.25) throughout, so the
        # absolute-only path (ratio 0, or the gate never opening) never
        # finalizes. relative_silence_ratio=0.7 puts the bar at 0.63, so 0.6
        # reads as a sustained fall from peak and closes the utterance.
        probs = _room_preamble([0.9] * 20 + [0.6] * 19)

        cfg_off = VadConfig(relative_silence_ratio=0.0)
        seg_off = Segmenter(cfg_off, ScriptedVad(list(probs)))
        events_off = _run(seg_off, len(probs))
        assert not _by_type(events_off, SegFinal), "ratio 0.0 must reproduce the absolute-only behavior"

        cfg_on = VadConfig(relative_silence_ratio=0.7)
        seg_on = Segmenter(cfg_on, ScriptedVad(list(probs)))
        events_on = _run(seg_on, len(probs))
        finals = _by_type(events_on, SegFinal)
        assert len(finals) == 1
        assert finals[0].utterance_id == 1

    def test_short_relative_dip_does_not_finalize(self):
        # A 5-frame dip to 0.6 (below finalize_frames=19) between two runs of
        # real 0.9 speech must not close the utterance: the dwell requirement
        # protects a momentary dip exactly as it does for absolute silence.
        probs = _room_preamble([0.9] * 10 + [0.6] * 5 + [0.9] * 10)
        cfg = VadConfig(relative_silence_ratio=0.7)
        seg = Segmenter(cfg, ScriptedVad(probs))
        events = _run(seg, len(probs))
        assert not _by_type(events, SegFinal)
        assert not _by_type(events, SegDiscard)

    def test_relative_speculative_is_discarded_by_a_real_resume(self):
        # A relative-only drop (never crosses the absolute bars) still reaches
        # the speculative threshold and gets resolved by a Discard when real
        # speech (not a drop) resumes, same contract as an absolute dip.
        probs = _room_preamble([0.9] * 20 + [0.6] * 10 + [0.9] * 5)
        cfg = VadConfig(relative_silence_ratio=0.7)
        seg = Segmenter(cfg, ScriptedVad(probs))
        events = _run(seg, len(probs))
        assert len(_by_type(events, SegSpeculative)) == 1
        assert len(_by_type(events, SegDiscard)) == 1
        assert not _by_type(events, SegFinal)


class TestRoomFloorGate:
    """Segmenter._update_room_floor is the mechanism that keeps a clean room
    on the absolute-only path by default (VadConfig.relative_silence_ratio
    ships nonzero): these are its two ways of staying closed, each isolated
    from the other."""

    def test_gate_stays_closed_before_enough_room_evidence(self):
        # No preamble: a cold start. The dip itself eventually supplies room
        # evidence too -- a sustained below-peak reading looks like a room to
        # the same test that gates it -- but warming up costs
        # _FLOOR_WARM_FRAMES of the dip before is_drop can fire at all, so
        # fewer than finalize_frames dip frames remain to actually close the
        # utterance: a dip exactly finalize_frames long, with no prior
        # evidence, still does not finalize. Compare
        # test_sustained_relative_drop_finalizes_where_absolute_alone_would_
        # not, the same dip length with a room preamble already warm.
        probs = [0.9] * 20 + [0.6] * 19
        cfg = VadConfig(relative_silence_ratio=0.7)
        seg = Segmenter(cfg, ScriptedVad(probs))
        events = _run(seg, len(probs))
        assert not _by_type(events, SegFinal)

    def test_gate_stays_closed_behind_a_true_quiet_lead_in(self):
        # A genuinely quiet, below-threshold lead-in (real idle: no false
        # SpeechStart) still counts as room evidence -- idle frames always do
        # -- but its own low VAD reading keeps the estimate under
        # silence_bar, so the gate never opens even once warm: the accuracy
        # case this default has to hold (see relative_silence_ratio's
        # docstring in vad_config.py).
        lead_in = [0.05] * _FLOOR_WARM_FRAMES
        probs = lead_in + [0.9] * 20 + [0.6] * 19
        cfg = VadConfig(relative_silence_ratio=0.7)
        seg = Segmenter(cfg, ScriptedVad(probs))
        events = _run(seg, len(probs))
        assert not _by_type(events, SegFinal)
