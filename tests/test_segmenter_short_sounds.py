"""Silence padding must not turn a brief trigger into a valid utterance."""

import numpy as np
import pytest

from vrcc.audio.segmenter import (
    FRAME, SegDiscard, SegFinal, SegSpeculative, Segmenter,
)
from vrcc.core.config import VadConfig


@pytest.mark.parametrize("speech_frames", [1, 4, 15])
@pytest.mark.parametrize("finalize_ms", [600, 1000])
def test_trailing_silence_cannot_satisfy_minimum(speech_frames, finalize_ms):
    # 15 frames are 480 ms, below a configured 500 ms minimum even when
    # pre-roll and the wait for finalization make the buffer much longer.
    probabilities = iter([0.1] * 5 + [0.9] * speech_frames + [0.1] * 32)
    segmenter = Segmenter(
        VadConfig(finalize_silence_ms=finalize_ms, min_utterance_ms=500), lambda _: next(probabilities)
    )
    events = []
    for _ in range(5 + speech_frames + 32):
        events.extend(segmenter.process(np.zeros(FRAME, dtype=np.float32)))

    assert not any(isinstance(event, SegFinal) for event in events)
    speculative = [e.utterance_id for e in events if isinstance(e, SegSpeculative)]
    discarded = [e.utterance_id for e in events if isinstance(e, SegDiscard)]
    assert discarded == speculative
    assert not segmenter.active


def test_minimum_length_soft_speech_still_finalizes():
    # The duration check must not raise the probability or volume threshold.
    probabilities = iter([0.4] * 16 + [0.1] * 19)
    segmenter = Segmenter(VadConfig(), lambda _: next(probabilities))
    events = []
    for _ in range(35):
        events.extend(segmenter.process(np.full(FRAME, 0.001, dtype=np.float32)))

    finals = [e for e in events if isinstance(e, SegFinal)]
    speculative = [e for e in events if isinstance(e, SegSpeculative)]
    assert len(finals) == len(speculative) == 1
    assert finals[0].samples is speculative[0].samples
