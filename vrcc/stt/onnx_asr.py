"""onnx-asr STT engine: NVIDIA NeMo ONNX exports (Parakeet TDT, Canary AED).

Same duck-typed contract as :class:`vrcc.stt.engine.SttEngine` (load /
warm_up / unload / transcribe), turning mono float32 16 kHz audio into an
:class:`SttResult`. The ``onnx_asr`` package is imported lazily in
:meth:`load`; a failed CUDA session build falls back to CPU once. Device
``auto`` runs these models on CPU even when CUDA is available: the int8
exports measured no faster on CUDA than CPU (see
benchmarks/rtx-5090-ryzen-9950x3d.json) and a CUDA session takes VRAM from
VRChat; an explicit ``cuda`` config still builds CUDA sessions. These
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
        A resolved ``auto`` device builds on CPU even when it resolves to
        CUDA (deliberate, so no ``fallback_cpu`` event); only an explicit
        ``cuda`` config builds a CUDA session.
        Any error building a CUDA session triggers one ``fallback_cpu`` +
        rebuild on CPU (the bundled onnxruntime is usually CPU-only); a CUDA
        provider that fails to *initialize* (e.g. missing CUDA runtime DLLs)
        doesn't raise -- onnxruntime quietly builds CPU sessions -- so the
        sessions are inspected and the ``ready`` detail reports the device
        they actually run on. Any other error publishes ``failed`` and
        re-raises.
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
            if device == "cuda" and self._cfg.device == "auto":
                # Deliberate, not a fallback: the int8 exports measured no
                # faster on CUDA than CPU (see
                # benchmarks/rtx-5090-ryzen-9950x3d.json) and a CUDA session
                # takes VRAM from VRChat. An explicit "cuda" config still
                # gets CUDA.
                device = "cpu"
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
            else:
                if self._device == "cuda":
                    self._check_cuda_sessions()

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

    def _check_cuda_sessions(self) -> None:
        """Downgrade ``_device`` to cpu when no built session actually got the
        CUDA provider. onnxruntime doesn't raise when a requested provider
        fails to initialize (e.g. the onnxruntime-gpu build wants CUDA runtime
        DLLs the install doesn't ship) -- it logs and builds CPU sessions, so
        the requested device can't be trusted."""
        session_providers = _session_providers(self._model)
        if not session_providers or "CUDAExecutionProvider" in session_providers:
            return
        detail = "onnxruntime built CPU-only sessions (CUDA provider failed to initialize)"
        logger.warning("%s: %s", self._spec.id, detail)
        self._bus.publish(EngineStateChanged("stt", "fallback_cpu", detail))
        self._device = "cpu"

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


def _session_providers(obj, depth: int = 3) -> set[str]:
    """Union of the execution providers on every onnxruntime session reachable
    from ``obj``'s attributes. onnx-asr wraps the model in adapters and keeps
    sessions in private attrs that vary by model class, so this duck-walks
    ``vars()`` a few levels for anything with ``get_providers()``. Empty when
    nothing session-like is found (e.g. test fakes)."""
    providers: set[str] = set()
    if depth <= 0 or not hasattr(obj, "__dict__"):
        return providers
    for value in vars(obj).values():
        get_providers = getattr(value, "get_providers", None)
        if callable(get_providers):
            try:
                providers.update(get_providers())
            except Exception:  # noqa: BLE001 -- session-like fakes may misbehave
                continue
        else:
            providers.update(_session_providers(value, depth - 1))
    return providers
