"""onnx-asr STT engine: NVIDIA NeMo ONNX exports (Parakeet TDT, Canary AED).

Same duck-typed contract as :class:`vrcc.stt.engine.SttEngine` (load /
warm_up / unload / transcribe), turning mono float32 16 kHz audio into an
:class:`SttResult`. The ``onnx_asr`` package is imported lazily in
:meth:`load`; a failed CUDA session build falls back to CPU once. These
models report no per-segment confidence or no-speech probability, so results
carry neutral gate values (0.0) and the VAD is the effective quality gate.
Zero Qt.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from vrcc.core.bus import EventBus
from vrcc.core.config import SttConfig
from vrcc.core.events import EngineStateChanged
from vrcc.core.hardware import resolve
from vrcc.core.languages import get
from vrcc.stt.engine import SttResult
from vrcc.stt.registry import WhisperSpec

logger = logging.getLogger("vrcc.stt.onnx_asr")

# The onnx-asr model type whose decoder prompt accepts a source language
# (Canary). TDT/RNNT transducers neither need nor accept one.
_AED_TYPE = "nemo-conformer-aed"

_CPU_PROVIDERS = ("CPUExecutionProvider",)

# warm_up() transcribes this much silence (0.5s at 16kHz).
_WARM_UP_SAMPLES = 8000

_SAMPLE_RATE = 16000


class OnnxAsrEngine:
    """One loaded onnx-asr model.

    Single-caller contract (same as ``SttEngine``): driven by exactly one
    worker thread; ``transcribe()`` and ``unload()`` aren't thread-safe
    against each other.
    """

    def __init__(
        self,
        cfg: SttConfig,
        spec: WhisperSpec,
        model_dir: Path,
        bus: EventBus,
        model_factory=None,
    ) -> None:
        self._cfg = cfg
        self._spec = spec
        self._model_dir = Path(model_dir)
        self._bus = bus
        # Defaults to onnx_asr.load_model, resolved lazily in load() so the
        # onnxruntime import stays out of module import time. Tests inject fakes.
        self._model_factory = model_factory

        self._model = None
        self._device: str | None = None

    # -- lifecycle -----------------------------------------------------------

    def load(self) -> None:
        """Build the onnx-asr model and announce readiness.

        Publishes ``loading`` then ``ready`` (``detail="<device>:<quant>"``).
        Any error building a CUDA session triggers one ``fallback_cpu`` +
        rebuild on CPU (the bundled onnxruntime is usually CPU-only); any
        other error publishes ``failed`` and re-raises.
        """
        self._bus.publish(EngineStateChanged("stt", "loading"))
        try:
            if not self._model_dir.is_dir():
                raise RuntimeError(
                    f"model files for {self._spec.id!r} not found in "
                    f"{self._model_dir}; download the model in the Models window"
                )
            device, index, _compute = resolve(
                self._cfg.device, self._cfg.device_index, self._cfg.compute_type
            )
            providers = self._providers(device, index)
            try:
                self._model = self._build_model(providers)
                self._device = device if providers != _CPU_PROVIDERS else "cpu"
            except Exception as exc:  # noqa: BLE001 -- CUDA-session failures vary by runtime
                if providers == _CPU_PROVIDERS:
                    raise
                logger.warning(
                    "%s CUDA session failed; falling back to CPU: %s",
                    self._spec.id, exc,
                )
                self._bus.publish(EngineStateChanged("stt", "fallback_cpu", str(exc)))
                self._model = self._build_model(_CPU_PROVIDERS)
                self._device = "cpu"

            self._bus.publish(
                EngineStateChanged(
                    "stt", "ready", f"{self._device}:{self._spec.quantization or 'fp32'}"
                )
            )
        except Exception as exc:
            self._model = None
            self._bus.publish(EngineStateChanged("stt", "failed", str(exc)))
            raise

    def warm_up(self) -> None:
        """Transcribe 0.5s of silence to prime sessions/allocations (result
        discarded). Errors are not swallowed -- a failed warm-up means the
        engine is unhealthy.
        """
        self.transcribe(np.zeros(_WARM_UP_SAMPLES, dtype=np.float32))

    def unload(self) -> None:
        """Drop the model reference. Safe to call when not loaded."""
        self._model = None

    # -- transcription ---------------------------------------------------------

    def transcribe(self, samples: np.ndarray) -> SttResult | None:
        """Transcribe ``samples`` (mono float32, 16 kHz) into an :class:`SttResult`.

        Returns ``None`` for empty text. AED models (Canary) get the
        configured source language forced into their decoder prompt; the
        transducers auto-detect within their set but don't report it -- either
        way ``language`` echoes the configured source ("en" when set to auto,
        the MT source fallback). Raises ``RuntimeError`` if called before
        :meth:`load`.
        """
        if self._model is None:
            raise RuntimeError(
                "OnnxAsrEngine.transcribe() called before a successful load(); "
                "call load() first."
            )

        text = self._model.recognize(
            np.ascontiguousarray(samples, dtype=np.float32),
            sample_rate=_SAMPLE_RATE,
            **self._recognize_kwargs(),
        )
        text = (text or "").strip()
        if not text:
            return None

        source = self._cfg.source_language
        language = "en" if source == "auto" else get(source).whisper
        # No confidence/no-speech signals from these decoders: neutral values
        # that always pass SttConfig's gates (VAD is the effective gate).
        return SttResult(
            text=text, language=language, avg_logprob=0.0, no_speech_prob=0.0
        )

    # -- internals -------------------------------------------------------------

    def _recognize_kwargs(self) -> dict:
        """Per-utterance ``recognize()`` options: AED models take the
        configured source language (their prompt defaults to English), when it
        is one the model supports. Transducers take nothing."""
        if self._spec.asr_type != _AED_TYPE:
            return {}
        source = self._cfg.source_language
        if source == "auto":
            return {}
        code = get(source).whisper
        if self._spec.languages is not None and code not in self._spec.languages:
            return {}  # unsupported pick (combo greying bypassed): don't crash
        return {"language": code}

    def _providers(self, device: str, index: int) -> tuple:
        """onnxruntime providers for the resolved device: CUDA (pinned to the
        configured card) when requested *and* available in this onnxruntime
        build, else CPU."""
        if device == "cuda":
            import onnxruntime

            if "CUDAExecutionProvider" in onnxruntime.get_available_providers():
                return (
                    ("CUDAExecutionProvider", {"device_id": index}),
                    "CPUExecutionProvider",
                )
            logger.info(
                "CUDA requested but this onnxruntime build has no "
                "CUDAExecutionProvider; %s runs on CPU", self._spec.id,
            )
        return _CPU_PROVIDERS

    def _build_model(self, providers: tuple):
        """Construct the onnx-asr model from the already-downloaded files."""
        factory = self._model_factory
        if factory is None:
            import onnx_asr

            factory = onnx_asr.load_model
        return factory(
            self._spec.asr_type,
            self._model_dir,
            quantization=self._spec.quantization,
            providers=list(providers),
        )
