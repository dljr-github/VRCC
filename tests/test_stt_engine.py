"""Tests for :mod:`vrcc.stt.engine` with a fake model factory (no
faster-whisper): ctor device/compute/threads, transcribe kwargs, result
text/logprob aggregation, quality gates, and the CUDA-unusable CPU fallback.
"""

from __future__ import annotations

import numpy as np
import pytest

from vrcc.core.bus import EventBus
from vrcc.stt.engine import SttEngine, SttResult

from .stt_fakes import (
    _CUDA_UNUSABLE_TEXTS,
    MODEL_DIR,
    _cfg,
    _collect,
    _RecordingFactory,
    _seg,
)


# --------------------------------------------------------------------------
# load(): ctor kwargs + event sequence
# --------------------------------------------------------------------------

def test_load_records_ctor_kwargs_and_publishes_loading_then_ready():
    bus = EventBus()
    events = _collect(bus)
    factory = _RecordingFactory()
    cfg = _cfg(cpu_threads=4, num_workers=2)
    eng = SttEngine(cfg, MODEL_DIR, bus, model_factory=factory)

    eng.load()

    assert len(factory.calls) == 1
    call = factory.calls[0]
    assert call.model_path == str(MODEL_DIR)
    assert call.device == "cpu"
    assert call.device_index == 0
    assert call.compute_type == "int8"
    assert call.cpu_threads == 4
    assert call.num_workers == 2
    assert call.local_files_only is True
    assert [(e.engine, e.state) for e in events] == [("stt", "loading"), ("stt", "ready")]
    assert events[-1].detail == "cpu:int8"


def test_load_passes_config_device_settings_to_resolve(monkeypatch):
    seen = []

    def fake_resolve(device_cfg, device_index, compute_cfg):
        seen.append((device_cfg, device_index, compute_cfg))
        return ("cpu", 0, "int8")

    monkeypatch.setattr("vrcc.stt.engine.resolve", fake_resolve)
    factory = _RecordingFactory()
    cfg = _cfg(device="cuda", device_index=2, compute_type="auto")
    eng = SttEngine(cfg, MODEL_DIR, EventBus(), model_factory=factory)

    eng.load()

    assert seen == [("cuda", 2, "auto")]


# --------------------------------------------------------------------------
# transcribe(): kwargs passthrough
# --------------------------------------------------------------------------

def test_transcribe_kwarg_passthrough_and_no_repeat_ngram_omitted_when_zero():
    factory = _RecordingFactory(segments=[_seg("hello", 0.0, 1.0, -0.1, 0.05)])
    cfg = _cfg(
        source_language="English",
        beam_size=3,
        temperature=0.2,
        condition_on_previous_text=True,
        without_timestamps=False,
        initial_prompt="",
        no_repeat_ngram_size=0,
    )
    eng = SttEngine(cfg, MODEL_DIR, EventBus(), model_factory=factory)
    eng.load()

    eng.transcribe(np.zeros(1600, dtype=np.float32))

    kwargs = factory.built[0].calls[0].kwargs
    assert kwargs["language"] == "en"
    assert kwargs["beam_size"] == 3
    assert kwargs["temperature"] == 0.2
    assert kwargs["condition_on_previous_text"] is True
    assert kwargs["without_timestamps"] is False
    assert kwargs["word_timestamps"] is False
    assert kwargs["vad_filter"] is False
    assert kwargs["initial_prompt"] is None
    assert "no_repeat_ngram_size" not in kwargs


def test_transcribe_includes_no_repeat_ngram_size_when_positive():
    factory = _RecordingFactory(segments=[_seg("hello", 0.0, 1.0, -0.1, 0.05)])
    cfg = _cfg(no_repeat_ngram_size=3)
    eng = SttEngine(cfg, MODEL_DIR, EventBus(), model_factory=factory)
    eng.load()

    eng.transcribe(np.zeros(1600, dtype=np.float32))

    assert factory.built[0].calls[0].kwargs["no_repeat_ngram_size"] == 3


def test_transcribe_default_includes_no_repeat_ngram_size_three():
    factory = _RecordingFactory(segments=[_seg("hello", 0.0, 1.0, -0.1, 0.05)])
    eng = SttEngine(_cfg(), MODEL_DIR, EventBus(), model_factory=factory)
    eng.load()
    eng.transcribe(np.zeros(1600, dtype=np.float32))
    assert factory.built[0].calls[0].kwargs["no_repeat_ngram_size"] == 3


def test_transcribe_passes_nonempty_initial_prompt():
    factory = _RecordingFactory(segments=[_seg("hello", 0.0, 1.0, -0.1, 0.05)])
    cfg = _cfg(initial_prompt="VRChat, avatar, world")
    eng = SttEngine(cfg, MODEL_DIR, EventBus(), model_factory=factory)
    eng.load()

    eng.transcribe(np.zeros(1600, dtype=np.float32))

    assert factory.built[0].calls[0].kwargs["initial_prompt"] == "VRChat, avatar, world"


def test_extra_kwargs_merge_overrides_beam_size_and_adds_new_key():
    factory = _RecordingFactory(segments=[_seg("hello", 0.0, 1.0, -0.1, 0.05)])
    cfg = _cfg(beam_size=1, extra_transcribe_kwargs={"beam_size": 5, "patience": 2.0})
    eng = SttEngine(cfg, MODEL_DIR, EventBus(), model_factory=factory)
    eng.load()

    eng.transcribe(np.zeros(1600, dtype=np.float32))

    kwargs = factory.built[0].calls[0].kwargs
    assert kwargs["beam_size"] == 5  # user override wins
    assert kwargs["patience"] == 2.0


def test_language_none_for_auto_source_language():
    factory = _RecordingFactory(segments=[_seg("hello", 0.0, 1.0, -0.1, 0.05)])
    cfg = _cfg(source_language="auto")
    eng = SttEngine(cfg, MODEL_DIR, EventBus(), model_factory=factory)
    eng.load()

    eng.transcribe(np.zeros(1600, dtype=np.float32))

    assert factory.built[0].calls[0].kwargs["language"] is None


def test_language_code_for_japanese_source_language():
    factory = _RecordingFactory(segments=[_seg("hello", 0.0, 1.0, -0.1, 0.05)])
    cfg = _cfg(source_language="Japanese")
    eng = SttEngine(cfg, MODEL_DIR, EventBus(), model_factory=factory)
    eng.load()

    eng.transcribe(np.zeros(1600, dtype=np.float32))

    assert factory.built[0].calls[0].kwargs["language"] == "ja"


# --------------------------------------------------------------------------
# _build_kwargs(): Traditional Chinese initial_prompt bias
# --------------------------------------------------------------------------

def test_traditional_chinese_gets_nonempty_seed_initial_prompt():
    eng = SttEngine(_cfg(source_language="Chinese Traditional"), MODEL_DIR, EventBus())

    kwargs = eng._build_kwargs()

    assert kwargs["initial_prompt"]


def test_simplified_chinese_initial_prompt_left_unchanged():
    eng = SttEngine(_cfg(source_language="Chinese Simplified"), MODEL_DIR, EventBus())

    kwargs = eng._build_kwargs()

    assert kwargs["initial_prompt"] is None


def test_english_initial_prompt_left_unchanged():
    eng = SttEngine(_cfg(source_language="English"), MODEL_DIR, EventBus())

    kwargs = eng._build_kwargs()

    assert kwargs["initial_prompt"] is None


def test_traditional_seed_prompt_differs_from_simplified_and_empty():
    trad = SttEngine(
        _cfg(source_language="Chinese Traditional"), MODEL_DIR, EventBus()
    )._build_kwargs()["initial_prompt"]
    simp = SttEngine(
        _cfg(source_language="Chinese Simplified"), MODEL_DIR, EventBus()
    )._build_kwargs()["initial_prompt"]

    assert trad not in (None, "")
    assert trad != simp


def test_user_initial_prompt_preserved_verbatim_for_traditional_chinese():
    eng = SttEngine(
        _cfg(source_language="Chinese Traditional", initial_prompt="VRChat, avatar"),
        MODEL_DIR,
        EventBus(),
    )

    kwargs = eng._build_kwargs()

    assert kwargs["initial_prompt"] == "VRChat, avatar"


# --------------------------------------------------------------------------
# transcribe(): result assembly + quality gates
# --------------------------------------------------------------------------

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


def test_gate_drops_low_avg_logprob():
    factory = _RecordingFactory(segments=[_seg("hello", 0.0, 1.0, -0.9, 0.05)])
    eng = SttEngine(_cfg(avg_logprob_gate=-0.8), MODEL_DIR, EventBus(), model_factory=factory)
    eng.load()

    assert eng.transcribe(np.zeros(1600, dtype=np.float32)) is None


def test_gate_drops_high_no_speech_prob():
    factory = _RecordingFactory(segments=[_seg("hello", 0.0, 1.0, -0.1, 0.9)])
    eng = SttEngine(_cfg(no_speech_gate=0.6), MODEL_DIR, EventBus(), model_factory=factory)
    eng.load()

    assert eng.transcribe(np.zeros(1600, dtype=np.float32)) is None


def test_gate_drops_high_compression_ratio():
    factory = _RecordingFactory(
        segments=[_seg("ha ha ha ha", 0.0, 1.0, -0.1, 0.05, compression_ratio=37.3)]
    )
    eng = SttEngine(
        _cfg(compression_ratio_gate=2.5), MODEL_DIR, EventBus(), model_factory=factory
    )
    eng.load()
    assert eng.transcribe(np.zeros(1600, dtype=np.float32)) is None


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


# --------------------------------------------------------------------------
# CUDA-unusable fallback (VRAM OOM or missing runtime library)
# --------------------------------------------------------------------------

@pytest.mark.parametrize("error_text", _CUDA_UNUSABLE_TEXTS)
def test_cuda_unusable_in_ctor_rebuilds_on_cpu_int8_with_fallback_event(
    monkeypatch, error_text
):
    monkeypatch.setattr("vrcc.stt.engine.resolve", lambda *a: ("cuda", 0, "int8_float16"))
    bus = EventBus()
    events = _collect(bus)
    factory = _RecordingFactory(
        ctor_fail_at=[0], segments=[_seg("hi", 0.0, 1.0, -0.1, 0.05)],
        error_text=error_text,
    )
    cfg = _cfg(device="cuda", compute_type="int8_float16", cpu_threads=4, num_workers=2)
    eng = SttEngine(cfg, MODEL_DIR, bus, model_factory=factory)

    eng.load()

    assert len(factory.calls) == 2
    assert factory.calls[0].device == "cuda"
    second = factory.calls[1]
    assert (second.device, second.device_index, second.compute_type) == ("cpu", 0, "int8")
    assert second.cpu_threads == 4 and second.num_workers == 2  # threads preserved
    assert [(e.engine, e.state) for e in events] == [
        ("stt", "loading"),
        ("stt", "fallback_cpu"),
        ("stt", "ready"),
    ]
    assert events[-1].detail == "cpu:int8"
    # engine is usable afterwards on the cpu model
    result = eng.transcribe(np.zeros(1600, dtype=np.float32))
    assert result is not None


@pytest.mark.parametrize("error_text", _CUDA_UNUSABLE_TEXTS)
def test_cuda_unusable_in_transcribe_rebuilds_cpu_and_retries_once(
    monkeypatch, error_text
):
    monkeypatch.setattr("vrcc.stt.engine.resolve", lambda *a: ("cuda", 0, "int8_float16"))
    bus = EventBus()
    events = _collect(bus)
    factory = _RecordingFactory(
        transcribe_fail_at=[0], segments=[_seg("hi", 0.0, 1.0, -0.1, 0.05)],
        error_text=error_text,
    )
    cfg = _cfg(device="cuda", compute_type="int8_float16")
    eng = SttEngine(cfg, MODEL_DIR, bus, model_factory=factory)
    eng.load()
    assert factory.calls[0].device == "cuda"
    events.clear()  # drop the load events; focus on the transcribe path

    result = eng.transcribe(np.zeros(1600, dtype=np.float32))

    assert len(factory.calls) == 2
    assert (factory.calls[1].device, factory.calls[1].compute_type) == ("cpu", "int8")
    assert [(e.engine, e.state) for e in events] == [
        ("stt", "fallback_cpu"),
        ("stt", "ready"),
    ]
    assert events[-1].detail == "cpu:int8"
    assert result is not None
    assert len(factory.built[0].calls) == 1  # failed attempt
    assert len(factory.built[1].calls) == 1  # successful retry


def test_second_oom_after_fallback_propagates(monkeypatch):
    monkeypatch.setattr("vrcc.stt.engine.resolve", lambda *a: ("cuda", 0, "int8_float16"))
    factory = _RecordingFactory(transcribe_fail_at=[0, 1])
    cfg = _cfg(device="cuda", compute_type="int8_float16")
    eng = SttEngine(cfg, MODEL_DIR, EventBus(), model_factory=factory)
    eng.load()

    with pytest.raises(RuntimeError, match="out of memory"):
        eng.transcribe(np.zeros(1600, dtype=np.float32))
    assert len(factory.calls) == 2  # rebuilt exactly once, no infinite retry


def test_unrecognized_error_in_ctor_publishes_failed_and_reraises():
    bus = EventBus()
    events = _collect(bus)

    class _BoomFactory:
        def __call__(self, *a, **k):
            raise RuntimeError("model.bin is corrupt")

    eng = SttEngine(_cfg(), MODEL_DIR, bus, model_factory=_BoomFactory())

    with pytest.raises(RuntimeError, match="corrupt"):
        eng.load()
    assert [(e.engine, e.state) for e in events] == [("stt", "loading"), ("stt", "failed")]
    assert events[-1].detail == "model.bin is corrupt"


# --------------------------------------------------------------------------
# guards: transcribe before / after a failed load
# --------------------------------------------------------------------------

def test_transcribe_before_load_raises():
    eng = SttEngine(_cfg(), MODEL_DIR, EventBus(), model_factory=_RecordingFactory())
    with pytest.raises(RuntimeError):
        eng.transcribe(np.zeros(1600, dtype=np.float32))


def test_transcribe_after_failed_load_raises():
    class _BoomFactory:
        def __call__(self, *a, **k):
            raise RuntimeError("boom")

    eng = SttEngine(_cfg(), MODEL_DIR, EventBus(), model_factory=_BoomFactory())
    with pytest.raises(RuntimeError):
        eng.load()
    with pytest.raises(RuntimeError):
        eng.transcribe(np.zeros(1600, dtype=np.float32))


# --------------------------------------------------------------------------
# warm_up / unload
# --------------------------------------------------------------------------

def test_warm_up_transcribes_half_second_of_zeros():
    factory = _RecordingFactory(segments=[_seg("", 0.0, 0.5, -0.1, 0.05)])
    eng = SttEngine(_cfg(), MODEL_DIR, EventBus(), model_factory=factory)
    eng.load()

    eng.warm_up()

    call = factory.built[0].calls[0]
    assert call.samples.dtype == np.float32
    assert call.samples.shape == (8000,)
    assert np.all(call.samples == 0.0)


def test_warm_up_errors_propagate():
    class _BoomModel:
        def transcribe(self, samples, **kwargs):
            raise RuntimeError("boom")

    class _BoomFactory:
        def __call__(self, *a, **k):
            return _BoomModel()

    eng = SttEngine(_cfg(), MODEL_DIR, EventBus(), model_factory=_BoomFactory())
    eng.load()

    with pytest.raises(RuntimeError, match="boom"):
        eng.warm_up()


def test_unload_is_safe_before_and_after_load():
    factory = _RecordingFactory()
    eng = SttEngine(_cfg(), MODEL_DIR, EventBus(), model_factory=factory)

    eng.unload()  # safe with nothing loaded

    eng.load()
    eng.unload()
    with pytest.raises(RuntimeError):
        eng.transcribe(np.zeros(1600, dtype=np.float32))


def test_module_does_not_import_faster_whisper_eagerly():
    import vrcc.stt.engine as engine_mod

    # faster_whisper is imported lazily inside load(), never at module top level.
    assert not hasattr(engine_mod, "faster_whisper")
