"""faster-whisper STT engine: model load, quality-gated transcription, VRAM fallback.

Wraps one ``WhisperModel``, turning mono float32 audio into an :class:`SttResult`
or ``None`` when it fails the quality gates. ``faster_whisper`` is imported lazily
in :meth:`load`; a CUDA OOM falls back to CPU int8 once. Zero Qt.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from vrcc.core.bus import EventBus
from vrcc.core.config import SttConfig
from vrcc.core.events import EngineStateChanged
from vrcc.core.hardware import resolve
from vrcc.core.languages import get

logger = logging.getLogger("vrcc.stt.engine")

# Case-insensitive substrings marking a CUDA out-of-VRAM error (one-shot CPU fallback).
_OOM_MARKERS = ("out of memory", "cublas_status_alloc_failed")

# The device/compute the engine falls back to when the GPU is out of VRAM.
_CPU_FALLBACK = ("cpu", 0, "int8")

# warm_up() transcribes this much silence (0.5s at faster-whisper's 16kHz).
_WARM_UP_SAMPLES = 8000


def _is_oom(exc: Exception) -> bool:
    """True if ``exc`` reads like a CUDA out-of-VRAM error."""
    text = str(exc).lower()
    return any(marker in text for marker in _OOM_MARKERS)


@dataclass(frozen=True)
class SttResult:
    text: str
    language: str
    avg_logprob: float
    no_speech_prob: float


class SttEngine:
    """One loaded faster-whisper model plus its quality gates.

    Single-caller contract: driven by exactly one worker thread. ``transcribe()``
    and ``unload()`` aren't thread-safe against each other (unload drops the
    model mid-flight); serialize all calls on one thread.
    """

    def __init__(
        self,
        cfg: SttConfig,
        model_dir: Path,
        bus: EventBus,
        model_factory=None,
    ) -> None:
        self._cfg = cfg
        self._model_dir = Path(model_dir)
        self._bus = bus
        # Defaults to faster_whisper.WhisperModel, resolved lazily in load() so
        # the native import stays out of module import time. Tests inject fakes.
        self._model_factory = model_factory

        self._model = None
        self._device: str | None = None
        self._compute_type: str | None = None

    # -- lifecycle -----------------------------------------------------------

    def load(self) -> None:
        """Build the ``WhisperModel`` and announce readiness.

        Publishes ``loading`` then ``ready`` (``detail="<device>:<compute>"``).
        A VRAM-OOM building the GPU model triggers one ``fallback_cpu`` + rebuild
        on ``("cpu", 0, "int8")``; any other error publishes ``failed`` and
        re-raises.
        """
        self._bus.publish(EngineStateChanged("stt", "loading"))
        try:
            device, index, compute = resolve(
                self._cfg.device, self._cfg.device_index, self._cfg.compute_type
            )
            try:
                self._model = self._build_model(device, index, compute)
            except RuntimeError as exc:
                if not _is_oom(exc):
                    raise
                self._fallback_to_cpu(str(exc))

            self._bus.publish(
                EngineStateChanged(
                    "stt", "ready", f"{self._device}:{self._compute_type}"
                )
            )
        except Exception as exc:
            self._model = None
            self._bus.publish(EngineStateChanged("stt", "failed", str(exc)))
            raise

    def warm_up(self) -> None:
        """Transcribe 0.5s of silence to prime kernels/allocations (result
        discarded). Errors are not swallowed -- a failed warm-up means the
        engine is unhealthy.
        """
        self.transcribe(np.zeros(_WARM_UP_SAMPLES, dtype=np.float32))

    def unload(self) -> None:
        """Drop the model reference. Safe to call when not loaded."""
        self._model = None

    # -- transcription ---------------------------------------------------------

    def transcribe(self, samples: np.ndarray) -> SttResult | None:
        """Transcribe ``samples`` (mono float32) into an :class:`SttResult`.

        Returns ``None`` when it fails the quality gates (no text, length-weighted
        ``avg_logprob`` below ``cfg.avg_logprob_gate``, or ``no_speech_prob`` above
        ``cfg.no_speech_gate``). Raises ``RuntimeError`` if called before :meth:`load`.
        """
        if self._model is None:
            raise RuntimeError(
                "SttEngine.transcribe() called before a successful load(); "
                "call load() first."
            )

        kwargs = self._build_kwargs()
        segments, info = self._run_transcribe(samples, kwargs)
        return self._build_result(segments, info)

    # -- internals -------------------------------------------------------------

    def _model_ctor(self):
        """The model factory, importing ``faster_whisper`` lazily if needed."""
        factory = self._model_factory
        if factory is None:
            import faster_whisper

            factory = faster_whisper.WhisperModel
        return factory

    def _build_model(self, device: str, index: int, compute: str):
        """Construct a model and record the device/compute it runs on."""
        model = self._model_ctor()(
            str(self._model_dir),
            device=device,
            device_index=index,
            compute_type=compute,
            cpu_threads=self._cfg.cpu_threads,
            num_workers=self._cfg.num_workers,
            local_files_only=True,
        )
        self._device = device
        self._compute_type = compute
        return model

    def _fallback_to_cpu(self, detail: str) -> None:
        """Announce the VRAM fallback and rebuild the model on CPU int8."""
        logger.warning("STT ran out of VRAM; falling back to CPU int8: %s", detail)
        self._bus.publish(EngineStateChanged("stt", "fallback_cpu", detail))
        self._model = self._build_model(*_CPU_FALLBACK)

    def _build_kwargs(self) -> dict:
        """Assemble the ``transcribe()`` kwargs per the task's exact contract."""
        language = (
            None
            if self._cfg.source_language == "auto"
            else get(self._cfg.source_language).whisper
        )
        kwargs = {
            "language": language,
            "beam_size": self._cfg.beam_size,
            "temperature": self._cfg.temperature,
            "condition_on_previous_text": self._cfg.condition_on_previous_text,
            "without_timestamps": self._cfg.without_timestamps,
            "word_timestamps": False,
            "vad_filter": False,
            "initial_prompt": self._cfg.initial_prompt or None,
        }
        if self._cfg.no_repeat_ngram_size > 0:
            kwargs["no_repeat_ngram_size"] = self._cfg.no_repeat_ngram_size
        kwargs.update(self._cfg.extra_transcribe_kwargs)  # user wins, last word
        return kwargs

    def _run_transcribe(self, samples: np.ndarray, kwargs: dict):
        """Call ``model.transcribe``, falling back to CPU int8 once on VRAM OOM.

        ``fallback_cpu`` is transient (like the load path), so a successful retry
        re-publishes ``("stt", "ready")`` with the new device; a failed retry
        propagates without a ready event.
        """
        try:
            segments_gen, info = self._model.transcribe(samples, **kwargs)
            return list(segments_gen), info
        except RuntimeError as exc:
            if not _is_oom(exc):
                raise
            self._fallback_to_cpu(str(exc))
            segments_gen, info = self._model.transcribe(samples, **kwargs)
            segments = list(segments_gen)
            self._bus.publish(
                EngineStateChanged(
                    "stt", "ready", f"{self._device}:{self._compute_type}"
                )
            )
            return segments, info

    def _build_result(self, segments: list, info) -> SttResult | None:
        """Join segment texts and apply the length-weighted quality gates."""
        text = " ".join(seg.text for seg in segments).strip()
        if not text:
            return None

        total_weight = 0.0
        weighted_sum = 0.0
        max_no_speech_prob = 0.0
        for seg in segments:
            weight = max(seg.end - seg.start, 1e-6)
            weighted_sum += seg.avg_logprob * weight
            total_weight += weight
            max_no_speech_prob = max(max_no_speech_prob, seg.no_speech_prob)

        avg_logprob = weighted_sum / total_weight if total_weight > 0 else None
        if avg_logprob is None or avg_logprob < self._cfg.avg_logprob_gate:
            return None
        if max_no_speech_prob > self._cfg.no_speech_gate:
            return None

        return SttResult(
            text=text,
            language=info.language,
            avg_logprob=avg_logprob,
            no_speech_prob=max_no_speech_prob,
        )
