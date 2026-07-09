"""Tests for :mod:`vrcc.stt.parakeet` with a fake onnx-asr factory (no
onnx-asr/onnxruntime model load): load events, provider selection + CPU
fallback, transcribe result contract, and the create_stt_engine dispatch.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from vrcc.core.bus import EventBus
from vrcc.core.config import SttConfig
from vrcc.core.events import EngineStateChanged
from vrcc.stt import create_stt_engine
from vrcc.stt.engine import SttEngine, SttResult
from vrcc.stt.parakeet import ParakeetEngine
from vrcc.stt.registry import WHISPER_MODELS

PARAKEET_ID = "parakeet-tdt-0.6b-v3"
SPEC = WHISPER_MODELS[PARAKEET_ID]


class _FakeModel:
    """Records recognize() calls; returns a canned text."""

    def __init__(self, text: str = "hello there") -> None:
        self.text = text
        self.calls: list[SimpleNamespace] = []

    def recognize(self, samples, sample_rate=16000):
        self.calls.append(SimpleNamespace(samples=samples, sample_rate=sample_rate))
        return self.text


class _RecordingFactory:
    """Fake ``onnx_asr.load_model`` recording every call."""

    def __init__(self, text: str = "hello there", fail_at=()) -> None:
        self.calls: list[SimpleNamespace] = []
        self.built: list[_FakeModel] = []
        self._text = text
        self._fail_at = set(fail_at)

    def __call__(self, model, path, *, quantization, providers):
        idx = len(self.calls)
        self.calls.append(
            SimpleNamespace(
                model=model, path=path, quantization=quantization, providers=providers
            )
        )
        if idx in self._fail_at:
            raise RuntimeError("CUDA provider unavailable")
        m = _FakeModel(self._text)
        self.built.append(m)
        return m


def _cfg(**over) -> SttConfig:
    base = dict(
        model=PARAKEET_ID, device="cpu", device_index=0, compute_type="int8"
    )
    base.update(over)
    return SttConfig(**base)


def _collect(bus: EventBus) -> list[EngineStateChanged]:
    events: list[EngineStateChanged] = []
    bus.subscribe(EngineStateChanged, events.append)
    return events


@pytest.fixture()
def model_dir(tmp_path: Path) -> Path:
    d = tmp_path / "models" / "whisper" / PARAKEET_ID
    d.mkdir(parents=True)
    return d


# --------------------------------------------------------------------------
# load(): factory args + event sequence
# --------------------------------------------------------------------------

def test_load_passes_type_dir_quantization_and_cpu_providers(model_dir):
    bus = EventBus()
    events = _collect(bus)
    factory = _RecordingFactory()
    eng = ParakeetEngine(_cfg(), SPEC, model_dir, bus, model_factory=factory)

    eng.load()

    assert len(factory.calls) == 1
    call = factory.calls[0]
    assert call.model == "nemo-conformer-tdt"
    assert Path(call.path) == model_dir
    assert call.quantization == "int8"
    assert call.providers == ["CPUExecutionProvider"]
    assert [(e.engine, e.state) for e in events] == [("stt", "loading"), ("stt", "ready")]
    assert events[-1].detail == "cpu:int8"


def test_load_missing_model_dir_publishes_failed_and_raises(tmp_path):
    bus = EventBus()
    events = _collect(bus)
    factory = _RecordingFactory()
    eng = ParakeetEngine(
        _cfg(), SPEC, tmp_path / "nope", bus, model_factory=factory
    )

    with pytest.raises(RuntimeError, match="Models window"):
        eng.load()

    assert factory.calls == []
    assert [(e.engine, e.state) for e in events] == [("stt", "loading"), ("stt", "failed")]


def test_load_cuda_without_provider_runs_on_cpu(model_dir, monkeypatch):
    # Config says cuda but this onnxruntime build has no CUDAExecutionProvider:
    # the engine silently builds a CPU session and reports cpu in the detail.
    import onnxruntime

    monkeypatch.setattr(
        onnxruntime, "get_available_providers", lambda: ["CPUExecutionProvider"]
    )
    bus = EventBus()
    events = _collect(bus)
    factory = _RecordingFactory()
    eng = ParakeetEngine(_cfg(device="cuda"), SPEC, model_dir, bus, model_factory=factory)

    eng.load()

    assert factory.calls[0].providers == ["CPUExecutionProvider"]
    assert events[-1].state == "ready"
    assert events[-1].detail == "cpu:int8"


def test_load_cuda_session_failure_falls_back_to_cpu(model_dir, monkeypatch):
    import onnxruntime

    monkeypatch.setattr(
        onnxruntime,
        "get_available_providers",
        lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    bus = EventBus()
    events = _collect(bus)
    factory = _RecordingFactory(fail_at=(0,))
    eng = ParakeetEngine(
        _cfg(device="cuda", device_index=1), SPEC, model_dir, bus, model_factory=factory
    )

    eng.load()

    assert len(factory.calls) == 2
    assert factory.calls[0].providers == [
        ("CUDAExecutionProvider", {"device_id": 1}),
        "CPUExecutionProvider",
    ]
    assert factory.calls[1].providers == ["CPUExecutionProvider"]
    assert [(e.engine, e.state) for e in events] == [
        ("stt", "loading"), ("stt", "fallback_cpu"), ("stt", "ready")
    ]
    assert events[-1].detail == "cpu:int8"


def test_load_cpu_build_failure_publishes_failed_and_raises(model_dir):
    bus = EventBus()
    events = _collect(bus)
    factory = _RecordingFactory(fail_at=(0,))
    eng = ParakeetEngine(_cfg(), SPEC, model_dir, bus, model_factory=factory)

    with pytest.raises(RuntimeError):
        eng.load()

    assert [(e.engine, e.state) for e in events] == [("stt", "loading"), ("stt", "failed")]


# --------------------------------------------------------------------------
# transcribe()
# --------------------------------------------------------------------------

def _loaded_engine(model_dir, text="hello there", **cfg_over):
    bus = EventBus()
    factory = _RecordingFactory(text)
    eng = ParakeetEngine(_cfg(**cfg_over), SPEC, model_dir, bus, model_factory=factory)
    eng.load()
    return eng, factory


def test_transcribe_before_load_raises(model_dir):
    eng = ParakeetEngine(
        _cfg(), SPEC, model_dir, EventBus(), model_factory=_RecordingFactory()
    )
    with pytest.raises(RuntimeError, match="load"):
        eng.transcribe(np.zeros(160, dtype=np.float32))


def test_transcribe_returns_result_with_neutral_gates(model_dir):
    eng, factory = _loaded_engine(model_dir, text="  Bonjour tout le monde  ")
    result = eng.transcribe(np.zeros(1600, dtype=np.float32))

    assert isinstance(result, SttResult)
    assert result.text == "Bonjour tout le monde"
    assert result.avg_logprob == 0.0
    assert result.no_speech_prob == 0.0
    # Neutral values always pass the default SttConfig gates.
    cfg = SttConfig()
    assert result.avg_logprob >= cfg.avg_logprob_gate
    assert result.no_speech_prob <= cfg.no_speech_gate
    call = factory.built[0].calls[0]
    assert call.sample_rate == 16000
    assert call.samples.dtype == np.float32


def test_transcribe_empty_text_returns_none(model_dir):
    eng, _ = _loaded_engine(model_dir, text="   ")
    assert eng.transcribe(np.zeros(1600, dtype=np.float32)) is None


def test_transcribe_language_echoes_configured_source(model_dir):
    eng, _ = _loaded_engine(model_dir, source_language="French")
    assert eng.transcribe(np.zeros(160, dtype=np.float32)).language == "fr"


def test_transcribe_language_auto_falls_back_to_english(model_dir):
    eng, _ = _loaded_engine(model_dir, source_language="auto")
    assert eng.transcribe(np.zeros(160, dtype=np.float32)).language == "en"


def test_warm_up_transcribes_half_second_of_silence(model_dir):
    eng, factory = _loaded_engine(model_dir)
    eng.warm_up()
    call = factory.built[0].calls[0]
    assert len(call.samples) == 8000


def test_unload_drops_model_and_transcribe_raises(model_dir):
    eng, _ = _loaded_engine(model_dir)
    eng.unload()
    with pytest.raises(RuntimeError):
        eng.transcribe(np.zeros(160, dtype=np.float32))


# --------------------------------------------------------------------------
# create_stt_engine dispatch
# --------------------------------------------------------------------------

def test_factory_builds_parakeet_engine_for_parakeet_id(tmp_path):
    eng = create_stt_engine(_cfg(), tmp_path, EventBus())
    assert isinstance(eng, ParakeetEngine)


def test_factory_builds_whisper_engine_for_whisper_id(tmp_path):
    eng = create_stt_engine(_cfg(model="small"), tmp_path, EventBus())
    assert isinstance(eng, SttEngine)


def test_factory_builds_whisper_engine_for_unknown_id(tmp_path):
    # Free-form ids keep the historical faster-whisper behavior.
    eng = create_stt_engine(_cfg(model="my/custom-model"), tmp_path, EventBus())
    assert isinstance(eng, SttEngine)


def test_factory_model_id_override_wins_over_config(tmp_path):
    # Hot-swaps pass the swap target while keeping the live config object.
    cfg = _cfg(model="small")
    eng = create_stt_engine(cfg, tmp_path, EventBus(), model_id=PARAKEET_ID)
    assert isinstance(eng, ParakeetEngine)
    assert eng._cfg is cfg
