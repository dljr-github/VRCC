"""Tests for :mod:`vrcc.stt.engine` with a fake model factory (no
faster-whisper): ctor device/compute/threads, transcribe kwargs, result
text/logprob aggregation, quality gates, and the CUDA-unusable CPU fallback.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from vrcc.core.bus import EventBus
from vrcc.core.config import SttConfig
from vrcc.core.events import EngineStateChanged
from vrcc.stt.engine import SttEngine, SttResult

_OOM_TEXT = "CUDA failed with error out of memory (CUBLAS_STATUS_ALLOC_FAILED)"
# Verbatim CTranslate2 dynamic-loader error from a CPU-only install driving a
# visible GPU: cudart is statically linked, so the device enumerates, and the
# load dies at model build or the first CUDA op instead.
_MISSING_LIBRARY_TEXT = "Library cublas64_12.dll is not found or cannot be loaded"

# Both shapes of CUDA-unusable RuntimeError must take the one-shot CPU fallback.
_CUDA_UNUSABLE_TEXTS = [_OOM_TEXT, _MISSING_LIBRARY_TEXT]


def _seg(text: str, start: float, end: float, avg_logprob: float, no_speech_prob: float):
    return SimpleNamespace(
        text=text, start=start, end=end, avg_logprob=avg_logprob, no_speech_prob=no_speech_prob
    )


class _FakeModel:
    """Records transcribe() calls; returns a canned (segments, info) pair."""

    def __init__(
        self, segments=None, language="en", fail_on_transcribe: bool = False,
        error_text: str = _OOM_TEXT,
    ) -> None:
        self.segments = segments if segments is not None else []
        self.language = language
        self.fail_on_transcribe = fail_on_transcribe
        self.error_text = error_text
        self.calls: list[SimpleNamespace] = []

    def transcribe(self, samples, **kwargs):
        self.calls.append(SimpleNamespace(samples=samples, kwargs=kwargs))
        if self.fail_on_transcribe:
            raise RuntimeError(self.error_text)
        return iter(self.segments), SimpleNamespace(language=self.language)


class _RecordingFactory:
    """Fake ``faster_whisper.WhisperModel`` factory recording every ctor call."""

    def __init__(
        self, ctor_fail_at=(), transcribe_fail_at=(), segments=None, language="en",
        error_text=_OOM_TEXT,
    ) -> None:
        self.calls: list[SimpleNamespace] = []
        self.built: list[_FakeModel] = []
        self._ctor_fail_at = set(ctor_fail_at)
        self._transcribe_fail_at = set(transcribe_fail_at)
        self._segments = segments if segments is not None else []
        self._language = language
        self._error_text = error_text

    def __call__(
        self,
        model_path,
        *,
        device,
        device_index,
        compute_type,
        cpu_threads,
        num_workers,
        local_files_only,
    ):
        idx = len(self.calls)
        self.calls.append(
            SimpleNamespace(
                model_path=model_path,
                device=device,
                device_index=device_index,
                compute_type=compute_type,
                cpu_threads=cpu_threads,
                num_workers=num_workers,
                local_files_only=local_files_only,
            )
        )
        if idx in self._ctor_fail_at:
            raise RuntimeError(self._error_text)
        m = _FakeModel(
            segments=list(self._segments),
            language=self._language,
            fail_on_transcribe=idx in self._transcribe_fail_at,
            error_text=self._error_text,
        )
        self.built.append(m)
        return m


def _cfg(**over) -> SttConfig:
    base = dict(device="cpu", device_index=0, compute_type="int8")
    base.update(over)
    return SttConfig(**base)


def _collect(bus: EventBus) -> list[EngineStateChanged]:
    events: list[EngineStateChanged] = []
    bus.subscribe(EngineStateChanged, events.append)
    return events


MODEL_DIR = Path("C:/fake/model/dir")


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
