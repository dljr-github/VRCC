"""Microphone-level independent onset confirmation and short replies."""

import numpy as np
import pytest

from vrcc.audio.segmenter import FRAME, SegFinal, SegSpeculative, SegSpeechStart, Segmenter
from vrcc.core.config import VadConfig


def feed(probs, cfg=None, level=0.01):
    sequence = iter(probs)
    seg = Segmenter(cfg or VadConfig(), lambda _: next(sequence))
    events = []
    for _ in probs:
        events.extend(seg.process(np.full(FRAME, level, dtype=np.float32)))
    return seg, events


@pytest.mark.parametrize("level", [0.0001, 0.001, 0.01, 0.1, 0.9])
def test_brief_spike_never_starts_recognition(level):
    seg, events = feed([0.1] * 20 + [0.9] + [0.1] * 30, level=level)
    assert not any(isinstance(e, (SegSpeechStart, SegSpeculative, SegFinal)) for e in events)
    assert not seg.active


@pytest.mark.parametrize("level", [0.0001, 0.001, 0.01, 0.1, 0.9])
def test_short_soft_reply_is_not_rejected_by_half_second_minimum(level):
    _, events = feed([0.4] * 4 + [0.1] * 19, level=level)
    assert len([e for e in events if isinstance(e, SegSpeechStart)]) == 1
    assert len([e for e in events if isinstance(e, SegFinal)]) == 1


def test_confirmation_keeps_onset_even_without_preroll():
    probs = iter([0.9] * 4 + [0.1] * 19)
    seg = Segmenter(VadConfig(pre_roll_ms=0), lambda _: next(probs))
    events = []
    for i in range(23):
        events.extend(seg.process(np.full(FRAME, i / 100, dtype=np.float32)))
        if i == 0:
            assert not any(isinstance(e, SegSpeechStart) for e in events)
            # Optional energy gating must keep feeding a pending onset.
            assert seg.active
    final = next(e for e in events if isinstance(e, SegFinal))
    np.testing.assert_array_equal(final.samples[:4 * FRAME:FRAME],
                                  np.array([0, .01, .02, .03], dtype=np.float32))


@pytest.mark.parametrize("gap", [0.1, 0.3])
def test_separate_spikes_cannot_accumulate_confirmation(gap):
    _, events = feed([0.9, gap] * 30 + [0.1] * 19)
    assert not any(isinstance(e, (SegSpeechStart, SegSpeculative, SegFinal)) for e in events)


@pytest.mark.parametrize("method", ["reset", "abort"])
def test_device_change_or_mute_drops_pending_onset(method):
    seg = Segmenter(VadConfig(), lambda _: 0.9)
    frame = np.ones(FRAME, dtype=np.float32)
    seg.process(frame)
    getattr(seg, method)()
    assert not seg.active
    assert not any(isinstance(e, SegSpeechStart) for e in seg.process(frame))


def test_preroll_cap_cannot_publish_an_unconfirmed_spike():
    _, events = feed([0.1] * 6 + [0.9] + [0.1] * 19,
                     VadConfig(pre_roll_ms=192, max_utterance_s=0.16))
    assert not any(isinstance(e, (SegSpeechStart, SegFinal)) for e in events)


def test_configured_confirmation_delay_applies_and_keeps_copied_audio():
    sequence = iter([0.9] * 4 + [0.1] * 19)
    seg = Segmenter(VadConfig(speech_start_ms=128, pre_roll_ms=0),
                    lambda _: next(sequence))
    frame = np.zeros(FRAME, dtype=np.float32)
    events = []
    for i in range(23):
        frame[:] = i / 100
        current = seg.process(frame)
        if i < 3:
            assert not any(isinstance(e, SegSpeechStart) for e in current)
        elif i == 3:
            assert sum(isinstance(e, SegSpeechStart) for e in current) == 1
        events.extend(current)
    frame[:] = 1
    final = next(e for e in events if isinstance(e, SegFinal))
    np.testing.assert_array_equal(final.samples[:4 * FRAME:FRAME],
                                  np.array([0, .01, .02, .03], dtype=np.float32))


@pytest.mark.parametrize("minimum, uncertain_frames", [(96, 1), (500, 16)])
def test_trailing_uncertainty_cannot_extend_short_speech(minimum, uncertain_frames):
    _, events = feed([0.9] * 2 + [0.1] * 8 + [0.3] * uncertain_frames + [0.1] * 11,
                     VadConfig(min_utterance_ms=minimum))
    assert not any(isinstance(e, SegFinal) for e in events)
