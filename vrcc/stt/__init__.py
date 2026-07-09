"""Speech-to-text engines (faster-whisper and onnx-asr)."""

from __future__ import annotations

from pathlib import Path

from vrcc.core.bus import EventBus
from vrcc.core.config import SttConfig
from vrcc.stt.registry import WHISPER_MODELS


def create_stt_engine(
    cfg: SttConfig, model_dir: Path, bus: EventBus, model_id: str | None = None
):
    """Build the (unloaded) STT engine for ``model_id`` (default ``cfg.model``;
    hot-swaps pass the swap's target id while keeping the live ``cfg``): an
    :class:`~vrcc.stt.onnx_asr.OnnxAsrEngine` when the registry marks the id
    as onnx-asr-backed (Parakeet), else a
    :class:`~vrcc.stt.engine.SttEngine` (faster-whisper -- also the fallback
    for ids the registry doesn't know, preserving the old free-form-model-id
    behavior). Both share the same duck-typed load/warm_up/unload/transcribe
    contract. Engine modules import lazily so building the stack never pulls
    native runtimes early.
    """
    spec = WHISPER_MODELS.get(model_id if model_id is not None else cfg.model)
    if spec is not None and spec.backend == "onnx_asr":
        from vrcc.stt.onnx_asr import OnnxAsrEngine

        return OnnxAsrEngine(cfg, spec, model_dir, bus)

    from vrcc.stt.engine import SttEngine

    return SttEngine(cfg, model_dir, bus)
