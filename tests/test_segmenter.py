"""Tests for the Segmenter utterance-boundary state machine (VAD/level
gating, speech-start/preroll, speculative caching, discard-on-resume).
"""

from __future__ import annotations

import numpy as np
import pytest

from vrcc.audio.segmenter import (
    FRAME,
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


class TestSegLevel:
    def test_emitted_every_frame(self):
        vad = ScriptedVad([0.1, 0.2, 0.9, 0.1, 0.05])
        seg = Segmenter(VadConfig(), vad)
        all_events = []
        for _ in range(5):
            all_events.append(seg.process(_frame()))
        for events in all_events:
            levels = _by_type(events, SegLevel)
            assert len(levels) == 1

    def test_rms_and_vad_prob_values(self):
        vad = ScriptedVad([0.42])
        seg = Segmenter(VadConfig(), vad)
        events = seg.process(_frame(0.1))
        level = events[0]
        assert isinstance(level, SegLevel)
        assert level.rms == pytest.approx(0.1, abs=1e-6)
        assert level.vad_prob == pytest.approx(0.42)

    def test_level_is_first_event_even_when_others_fire(self):
        vad = ScriptedVad([0.9])
        seg = Segmenter(VadConfig(), vad)
        events = seg.process(_frame())
        assert isinstance(events[0], SegLevel)
        assert isinstance(events[1], SegSpeechStart)


class TestSpeechStartAndPreroll:
    def test_speech_start_fires_on_first_speech_frame(self):
        cfg = VadConfig()
        vad = ScriptedVad([0.1, 0.1, 0.1, 0.9])
        seg = Segmenter(cfg, vad)
        e1 = seg.process(_frame())
        e2 = seg.process(_frame())
        e3 = seg.process(_frame())
        e4 = seg.process(_frame())
        assert not _by_type(e1, SegSpeechStart)
        assert not _by_type(e2, SegSpeechStart)
        assert not _by_type(e3, SegSpeechStart)
        starts = _by_type(e4, SegSpeechStart)
        assert len(starts) == 1
        assert starts[0].utterance_id == 1

    def test_buffer_seeded_with_preroll(self):
        # pre_roll_ms=150 -> 5 frames. Feed 5 silent IDLE frames to fill the
        # ring, then a speech frame; the buffer should start life holding
        # the 5 preroll frames plus the triggering frame itself (6 total).
        cfg = VadConfig()
        vad = ScriptedVad([0.1] * 5 + [0.9])
        seg = Segmenter(cfg, vad)
        for _ in range(5):
            seg.process(_frame())
        seg.process(_frame())
        assert seg._preroll_frames == 5
        assert len(seg._buffer) == 6

    def test_preroll_ring_caps_at_configured_frames(self):
        # Feed far more than 5 silent frames before speech starts; only the
        # most recent 5 should seed the buffer.
        cfg = VadConfig()
        vad = ScriptedVad([0.1] * 20 + [0.9])
        seg = Segmenter(cfg, vad)
        for _ in range(20):
            seg.process(_frame())
        seg.process(_frame())
        assert len(seg._buffer) == 5 + 1

    def test_no_speech_start_while_never_crossing_threshold(self):
        vad = ScriptedVad([0.1] * 10)
        seg = Segmenter(VadConfig(), vad)
        for _ in range(10):
            events = seg.process(_frame())
            assert not _by_type(events, SegSpeechStart)

    def test_preroll_ring_stays_warm_across_utterances(self):
        # The pre-roll ring keeps updating while ACTIVE, so a second
        # utterance starting right after a finalize is seeded with the
        # trailing frames of the FIRST utterance (real recent audio), not
        # stale pre-first-utterance frames or an empty ring.
        cfg = VadConfig(
            pre_roll_ms=64,                 # 2 frames
            speculative_silence_ms=64_000,  # disabled
            finalize_silence_ms=64,         # 2 frames
            min_utterance_ms=32,
        )
        vad = ScriptedVad([0.9, 0.2, 0.2, 0.9, 0.2, 0.2])
        seg = Segmenter(cfg, vad)

        values = [0.6, 0.03, 0.04, 0.7, 0.05, 0.06]
        finals = []
        for v in values:
            events = seg.process(_frame(v))
            finals.extend(_by_type(events, SegFinal))

        assert len(finals) == 2
        assert finals[0].utterance_id == 1
        assert finals[1].utterance_id == 2
        # Utterance 1: ring was empty at start -> just [0.6, 0.03, 0.04].
        assert finals[0].samples.shape[0] == 3 * FRAME
        # Utterance 2: pre-rolled with utterance 1's trailing silence
        # frames -> [0.03, 0.04, 0.7, 0.05, 0.06].
        expected2 = [0.03, 0.04, 0.7, 0.05, 0.06]
        assert finals[1].samples.shape[0] == 5 * FRAME
        for i, v in enumerate(expected2):
            chunk = finals[1].samples[i * FRAME : (i + 1) * FRAME]
            assert np.all(chunk == np.float32(v)), f"utterance 2 frame {i}"

    def test_idle_preroll_frames_are_copied(self):
        # Copy-safety for the IDLE pre-roll path: the caller reuses one
        # buffer for the idle frames and mutates it between calls; the
        # pre-roll audio seeded into the utterance must keep the values
        # each frame had AT process() time.
        cfg = VadConfig(
            pre_roll_ms=64,                 # 2 frames
            speculative_silence_ms=64_000,  # disabled
            finalize_silence_ms=64,         # 2 frames
            min_utterance_ms=32,
        )
        vad = ScriptedVad([0.1, 0.1, 0.9, 0.2, 0.2])
        seg = Segmenter(cfg, vad)

        buffer = np.full(FRAME, 0.01, dtype=np.float32)
        seg.process(buffer)        # idle pre-roll frame #1 (0.01)
        buffer[:] = 0.02
        seg.process(buffer)        # idle pre-roll frame #2 (0.02)
        buffer[:] = 999.0          # capture loop clobbers its buffer

        final = None
        for v in (0.6, 0.03, 0.04):
            events = seg.process(_frame(v))
            for e in _by_type(events, SegFinal):
                final = e

        assert final is not None
        expected = [0.01, 0.02, 0.6, 0.03, 0.04]
        assert final.samples.shape[0] == 5 * FRAME
        for i, v in enumerate(expected):
            chunk = final.samples[i * FRAME : (i + 1) * FRAME]
            assert np.all(chunk == np.float32(v)), f"frame {i} corrupted"


class TestSpeculativeAndFinalWithIdentity:
    def test_speculative_fires_once_at_11_silence_frames(self):
        # Brief scenario: 20 speech frames, then 11 silence frames ->
        # Speculative at overall frame 31 (350ms ~= 11 frames).
        cfg = VadConfig()
        probs = [0.9] * 20 + [0.1] * 11
        vad = ScriptedVad(probs)
        seg = Segmenter(cfg, vad)

        spec_events = []
        for _ in range(31):
            events = seg.process(_frame())
            spec_events.append(_by_type(events, SegSpeculative))

        # Speculative must not fire before the 11th silence frame (index 30,
        # 0-based -> overall frame 31).
        for i in range(30):
            assert spec_events[i] == [], f"unexpected speculative at frame {i + 1}"
        assert len(spec_events[30]) == 1
        spec = spec_events[30][0]
        assert spec.utterance_id == 1
        assert isinstance(spec.samples, np.ndarray)
        assert spec.samples.dtype == np.float32
        assert spec.samples.ndim == 1
        # No pre-roll frames were ever idle-buffered before this run (speech
        # started on frame 1), so buffer == 20 speech + 11 silence frames.
        assert spec.samples.shape[0] == (20 + 11) * FRAME

    def test_final_at_19_silence_frames_reuses_speculative_object(self):
        # Continue: after the speculative at 11 silence frames, 8 more
        # silence frames (total 19) should force SegFinal, reusing the
        # exact speculative array object (behavior 8).
        cfg = VadConfig()
        probs = [0.9] * 20 + [0.1] * 19
        vad = ScriptedVad(probs)
        seg = Segmenter(cfg, vad)

        spec = None
        final = None
        for i in range(39):
            events = seg.process(_frame())
            specs = _by_type(events, SegSpeculative)
            finals = _by_type(events, SegFinal)
            if specs:
                assert spec is None, "speculative fired more than once"
                spec = specs[0]
            if finals:
                assert final is None, "final fired more than once"
                final = finals[0]

        assert spec is not None
        assert final is not None
        assert final.utterance_id == 1
        # Identity, not equality: same object reused because no speech frame
        # occurred between the speculative emit and finalization.
        assert final.samples is spec.samples

    def test_equal_thresholds_emit_only_final_not_speculative(self):
        # When speculative and finalize thresholds trip on the SAME frame
        # (equal frame counts), a speculative would be pointless -- the
        # final is already here. Only SegFinal must be emitted, so the
        # every-speculative-is-resolved invariant holds trivially.
        cfg = VadConfig(
            speculative_silence_ms=64,  # 2 frames
            finalize_silence_ms=64,     # 2 frames
            min_utterance_ms=32,
            pre_roll_ms=0,
        )
        vad = ScriptedVad([0.9, 0.1, 0.1])
        seg = Segmenter(cfg, vad)

        specs = []
        finals = []
        for _ in range(3):
            events = seg.process(_frame())
            specs.extend(_by_type(events, SegSpeculative))
            finals.extend(_by_type(events, SegFinal))

        assert specs == []
        assert len(finals) == 1

    def test_utterance_id_increments_after_final(self):
        cfg = VadConfig()
        probs = [0.9] * 20 + [0.1] * 19 + [0.9]
        vad = ScriptedVad(probs)
        seg = Segmenter(cfg, vad)
        starts = []
        for _ in range(40):
            events = seg.process(_frame())
            starts.extend(_by_type(events, SegSpeechStart))
        assert len(starts) == 2
        assert starts[0].utterance_id == 1
        assert starts[1].utterance_id == 2


class TestDiscardOnResume:
    def test_discard_fires_once_on_speech_resume_after_speculative(self):
        # Brief scenario: speech -> 12 silence (speculative fires at 11th,
        # 12th is just more silence) -> 5 speech frames -> Discard on the
        # FIRST of those 5 (not repeated).
        cfg = VadConfig()
        probs = [0.9] * 5 + [0.1] * 12 + [0.9] * 5
        vad = ScriptedVad(probs)
        seg = Segmenter(cfg, vad)

        discard_frames = []
        discards = []
        spec_frame = None
        for i in range(22):
            events = seg.process(_frame())
            if _by_type(events, SegSpeculative):
                spec_frame = i
            for e in _by_type(events, SegDiscard):
                discard_frames.append(i)
                discards.append(e)

        assert spec_frame == 5 + 11 - 1  # 0-based index of the 11th silence frame
        assert discard_frames == [5 + 12]  # first of the 5 resumed speech frames
        assert discards[0].utterance_id == 1  # discard belongs to the SAME utterance

    def test_no_discard_if_speech_resumes_before_speculative_threshold(self):
        # Only 5 silence frames (< 11) then speech resumes: no speculative
        # was ever emitted, so no Discard should fire either.
        cfg = VadConfig()
        probs = [0.9] * 5 + [0.1] * 5 + [0.9] * 5
        vad = ScriptedVad(probs)
        seg = Segmenter(cfg, vad)
        discards = []
        specs = []
        for _ in range(15):
            events = seg.process(_frame())
            discards.extend(_by_type(events, SegDiscard))
            specs.extend(_by_type(events, SegSpeculative))
        assert specs == []
        assert discards == []

    def test_new_speculative_after_discard_is_a_new_object(self):
        # After a discard, a fresh silence run reaching the speculative
        # threshold must build a NEW array (not reuse the pre-discard one),
        # and a subsequent finalize reuses THAT new object.
        cfg = VadConfig()
        probs = (
            [0.9] * 5      # speech start
            + [0.1] * 12   # speculative #1 at 11th, then one more silence
            + [0.9] * 5    # resume -> discard
            + [0.1] * 19   # new silence run -> speculative #2 at 11, final at 19
        )
        vad = ScriptedVad(probs)
        seg = Segmenter(cfg, vad)

        specs = []
        final = None
        for _ in range(5 + 12 + 5 + 19):
            events = seg.process(_frame())
            specs.extend(_by_type(events, SegSpeculative))
            finals = _by_type(events, SegFinal)
            if finals:
                final = finals[0]

        assert len(specs) == 2
        assert final is not None
        assert specs[0].samples is not specs[1].samples
        assert final.samples is specs[1].samples
        assert final.samples is not specs[0].samples
        # Discard continues the SAME utterance: every event in this run
        # (both speculatives and the final) carries utterance_id 1.
        assert specs[0].utterance_id == 1
        assert specs[1].utterance_id == 1
        assert final.utterance_id == 1
