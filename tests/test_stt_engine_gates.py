"""Tests for :meth:`SttEngine.transcribe`'s result assembly and quality
gates: split out of test_stt_engine.py to stay under the 500-line cap.
"""

from __future__ import annotations

import numpy as np
import pytest

from vrcc.core.bus import EventBus
from vrcc.stt.engine import SttEngine, SttResult

from .stt_fakes import MODEL_DIR, _cfg, _RecordingFactory, _seg


def test_result_joins_segment_texts_and_reports_detected_language():
    factory = _RecordingFactory(
        segments=[_seg(" hello", 0.0, 1.0, -0.1, 0.05), _seg("world ", 1.0, 2.0, -0.1, 0.05)],
        language="fr",
    )
    cfg = _cfg(source_language="auto")
    eng = SttEngine(cfg, MODEL_DIR, EventBus(), model_factory=factory)
    eng.load()

    result = eng.transcribe(np.zeros(1600, dtype=np.float32))

    assert result == SttResult(text="hello world", language="fr", avg_logprob=-0.1, no_speech_prob=0.05)


def test_weighted_mean_avg_logprob_across_differing_durations():
    # seg1: duration 1, logprob -0.2 ; seg2: duration 3, logprob -0.5
    # weighted mean = (-0.2*1 + -0.5*3) / 4 = -0.425
    factory = _RecordingFactory(
        segments=[
            _seg("hi", 0.0, 1.0, -0.2, 0.1),
            _seg("there", 1.0, 4.0, -0.5, 0.2),
        ]
    )
    cfg = _cfg(avg_logprob_gate=-0.8, no_speech_gate=0.6)
    eng = SttEngine(cfg, MODEL_DIR, EventBus(), model_factory=factory)
    eng.load()

    result = eng.transcribe(np.zeros(1600, dtype=np.float32))

    assert result is not None
    assert result.avg_logprob == pytest.approx(-0.425)
    assert result.no_speech_prob == pytest.approx(0.2)  # max across segments


def test_gate_drops_empty_text():
    factory = _RecordingFactory(segments=[])
    eng = SttEngine(_cfg(), MODEL_DIR, EventBus(), model_factory=factory)
    eng.load()

    assert eng.transcribe(np.zeros(1600, dtype=np.float32)) is None


def test_gate_drops_low_avg_logprob(caplog):
    factory = _RecordingFactory(segments=[_seg("hello", 0.0, 1.0, -0.9, 0.05)])
    eng = SttEngine(_cfg(avg_logprob_gate=-0.8), MODEL_DIR, EventBus(), model_factory=factory)
    eng.load()

    with caplog.at_level("DEBUG", logger="vrcc.stt.engine"):
        assert eng.transcribe(np.zeros(1600, dtype=np.float32)) is None

    assert "avg_logprob" in caplog.text


def test_gate_drops_high_no_speech_prob(caplog):
    factory = _RecordingFactory(segments=[_seg("hello", 0.0, 1.0, -0.1, 0.9)])
    eng = SttEngine(_cfg(no_speech_gate=0.6), MODEL_DIR, EventBus(), model_factory=factory)
    eng.load()

    with caplog.at_level("DEBUG", logger="vrcc.stt.engine"):
        assert eng.transcribe(np.zeros(1600, dtype=np.float32)) is None

    assert "no_speech_prob" in caplog.text


def test_gate_drops_high_compression_ratio(caplog):
    factory = _RecordingFactory(
        segments=[_seg("ha ha ha ha", 0.0, 1.0, -0.1, 0.05, compression_ratio=37.3)]
    )
    eng = SttEngine(
        _cfg(compression_ratio_gate=2.5), MODEL_DIR, EventBus(), model_factory=factory
    )
    eng.load()
    with caplog.at_level("DEBUG", logger="vrcc.stt.engine"):
        assert eng.transcribe(np.zeros(1600, dtype=np.float32)) is None

    assert "compression_ratio" in caplog.text


def test_gate_uses_max_compression_ratio_across_segments():
    factory = _RecordingFactory(
        segments=[
            _seg("hello", 0.0, 1.0, -0.1, 0.05, compression_ratio=1.2),
            _seg("na na na", 1.0, 2.0, -0.1, 0.05, compression_ratio=9.0),
        ]
    )
    eng = SttEngine(
        _cfg(compression_ratio_gate=2.5), MODEL_DIR, EventBus(), model_factory=factory
    )
    eng.load()
    assert eng.transcribe(np.zeros(1600, dtype=np.float32)) is None


def test_normal_compression_ratio_passes_the_gate():
    factory = _RecordingFactory(
        segments=[_seg("hello world", 0.0, 1.0, -0.1, 0.05, compression_ratio=1.5)]
    )
    eng = SttEngine(
        _cfg(compression_ratio_gate=2.5), MODEL_DIR, EventBus(), model_factory=factory
    )
    eng.load()
    assert eng.transcribe(np.zeros(1600, dtype=np.float32)) is not None
