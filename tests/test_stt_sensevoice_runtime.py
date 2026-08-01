"""Tests for :meth:`SenseVoiceEngine._recheck_device_after_run`: onnxruntime's
CUDA->CPU fallback is a run-time event a build-time provider check cannot see,
so warm_up() has to re-inspect the session after its first run.

Shares the fake onnxruntime session/factory with test_stt_sensevoice.py rather
than duplicating them.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from vrcc.core.bus import EventBus
from vrcc.core.events import EngineStateChanged
from vrcc.stt.sensevoice import SenseVoiceEngine
from tests.test_stt_sensevoice import (
    SENSEVOICE,
    SENSEVOICE_ID,
    _RecordingFactory,
    _VOCAB,
    _cfg,
)


@pytest.fixture()
def model_dir(tmp_path: Path) -> Path:
    d = tmp_path / "models" / "whisper" / SENSEVOICE_ID
    d.mkdir(parents=True)
    (d / "model.int8.onnx").write_bytes(b"")
    (d / "tokens.txt").write_text(
        "\n".join(f"{piece} {i}" for i, piece in enumerate(_VOCAB)) + "\n",
        encoding="utf-8",
    )
    return d


def _collect(bus: EventBus) -> list[EngineStateChanged]:
    events: list[EngineStateChanged] = []
    bus.subscribe(EngineStateChanged, events.append)
    return events


def test_runtime_fallback_after_warm_up_is_reported(model_dir, monkeypatch):
    monkeypatch.setattr(
        "vrcc.stt.sensevoice.resolve", lambda *a, **k: ("cuda", 0, "int8")
    )
    monkeypatch.setattr(
        "onnxruntime.get_available_providers",
        lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    bus = EventBus()
    events = _collect(bus)
    factory = _RecordingFactory(
        session_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        providers_after_run=["CPUExecutionProvider"],
    )
    eng = SenseVoiceEngine(_cfg(device="cuda"), SENSEVOICE, model_dir, bus, session_factory=factory)

    eng.load()

    assert eng._device == "cuda"
    assert [e.state for e in events] == ["loading", "ready"]
    assert events[-1].detail == "cuda:int8"

    eng.warm_up()

    assert eng._device == "cpu"
    assert [e.state for e in events] == ["loading", "ready", "fallback_cpu", "ready"]
    assert events[-1].detail == "cpu:int8"


def test_healthy_gpu_run_keeps_cuda_and_fires_no_extra_events(model_dir, monkeypatch):
    monkeypatch.setattr(
        "vrcc.stt.sensevoice.resolve", lambda *a, **k: ("cuda", 0, "int8")
    )
    monkeypatch.setattr(
        "onnxruntime.get_available_providers",
        lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    bus = EventBus()
    events = _collect(bus)
    factory = _RecordingFactory(
        session_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    eng = SenseVoiceEngine(_cfg(device="cuda"), SENSEVOICE, model_dir, bus, session_factory=factory)

    eng.load()
    eng.warm_up()

    assert eng._device == "cuda"
    assert [e.state for e in events] == ["loading", "ready"]
