"""CTranslate2 translation engine: model load, batched translate, CPU fallback.

Wraps one ``Translator`` + :class:`MtTokenizer`, translating a phrase to all N
targets in a SINGLE ``translate_batch`` (layout keyed off ``spec.prefix_side``,
noted inline). Lazy ``ctranslate2``; an unusable CUDA (out of VRAM, or missing
its runtime libraries) -> CPU int8 once. Zero Qt.
"""

from __future__ import annotations

import logging
from pathlib import Path

from vrcc.core.bus import EventBus
from vrcc.core.config import TranslateConfig
from vrcc.core.events import EngineStateChanged
from vrcc.core.hardware import resolve
from vrcc.core.languages import Language, get
from vrcc.translate.registry import MtModelSpec
from vrcc.translate.tokenizers import MtTokenizer

logger = logging.getLogger("vrcc.translate.engine")

# Case-insensitive substrings marking CUDA as unusable (one-shot CPU fallback):
# out of VRAM, or missing runtime libraries. The last entry is the ctranslate2
# dynamic-loader message ("Library cublas64_12.dll is not found or cannot be
# loaded", same phrasing on Windows and Linux), raised at model build or the
# first CUDA op when a CPU-only install drives a visible GPU.
_CUDA_FALLBACK_MARKERS = (
    "out of memory",
    "cublas_status_alloc_failed",
    "is not found or cannot be loaded",
)

# The device/compute the engine falls back to when CUDA is unusable.
_CPU_FALLBACK = ("cpu", 0, "int8")


def _is_cuda_unusable(exc: Exception) -> bool:
    """True if ``exc`` reads like CUDA being unusable: out of VRAM, or a
    runtime library CTranslate2 needs is missing."""
    text = str(exc).lower()
    return any(marker in text for marker in _CUDA_FALLBACK_MARKERS)


class TranslateEngine:
    """One loaded CTranslate2 MT model plus its tokenizer and layout rules.

    Single-caller contract: driven by exactly one worker thread. ``translate()``
    and ``unload()`` aren't thread-safe against each other (unload drops the
    translator mid-flight); serialize all calls on one thread.
    """

    def __init__(
        self,
        spec: MtModelSpec,
        model_dir: Path,
        cfg: TranslateConfig,
        bus: EventBus,
        translator_factory=None,
    ) -> None:
        self._spec = spec
        self._model_dir = Path(model_dir)
        self._cfg = cfg
        self._bus = bus
        # Defaults to ``ctranslate2.Translator``, resolved lazily in load() so
        # the native import stays out of module import time. Tests inject fakes.
        self._translator_factory = translator_factory

        self._translator = None
        self._tokenizer: MtTokenizer | None = None
        self._device: str | None = None
        self._compute_type: str | None = None

    # -- lifecycle ---------------------------------------------------------

    def load(self) -> None:
        """Build the ``Translator`` + ``MtTokenizer`` and announce readiness.

        Publishes ``loading`` then ``ready`` (``detail="<device>:<compute>"``).
        An unusable CUDA (VRAM OOM, or a missing runtime library) while
        building the GPU translator triggers one ``fallback_cpu`` + rebuild on
        ``("cpu", 0, "int8")``; any other error publishes ``failed`` and
        re-raises.
        """
        self._bus.publish(EngineStateChanged("mt", "loading"))
        try:
            device, index, compute = resolve(
                self._cfg.device, self._cfg.device_index, self._cfg.compute_type
            )
            try:
                self._translator = self._build_translator(device, index, compute)
            except RuntimeError as exc:
                if not _is_cuda_unusable(exc):
                    raise
                self._fallback_to_cpu(str(exc))

            self._tokenizer = MtTokenizer(
                self._model_dir / self._spec.spm_file, self._spec.family
            )
            self._bus.publish(
                EngineStateChanged(
                    "mt", "ready", f"{self._device}:{self._compute_type}"
                )
            )
        except Exception as exc:
            self._translator = None
            self._tokenizer = None
            self._bus.publish(EngineStateChanged("mt", "failed", str(exc)))
            raise

    def warm_up(self) -> None:
        """Run one throwaway translation to prime kernels/allocations (result
        discarded). Translates ``"hello"`` into the first target (or Japanese).
        Errors are not swallowed -- a failed warm-up means the engine is unhealthy.
        """
        target_name = self._cfg.targets[0] if self._cfg.targets else "Japanese"
        self.translate("hello", get("English"), [get(target_name)])

    def unload(self) -> None:
        """Drop the translator and tokenizer. Safe to call when not loaded."""
        self._translator = None
        self._tokenizer = None

    # -- translation -------------------------------------------------------

    def translate(
        self, text: str, src: Language, targets: list[Language]
    ) -> list[tuple[str, str]]:
        """Translate ``text`` from ``src`` into every target in one batch.

        Returns ``[(target.display, translated_text), ...]`` in target order via
        a single ``translate_batch`` (layout per model family, below). Raises
        ``RuntimeError`` if called before :meth:`load`.
        """
        if self._translator is None or self._tokenizer is None:
            raise RuntimeError(
                "TranslateEngine.translate() called before a successful load(); "
                "call load() first."
            )

        tok = self._tokenizer
        if self._spec.prefix_side == "source":
            # madlad: target rides on the source side -> one encoding per target.
            source = [tok.encode_source(text, src, tgt=t) for t in targets]
            target_prefix = None
        else:
            # nllb / m2m100: identical source, per-entry decoder prefix.
            source_tokens = tok.encode_source(text, src)
            source = [source_tokens for _ in targets]
            target_prefix = [tok.target_prefix(t) for t in targets]

        kwargs = {"beam_size": self._cfg.beam_size, "return_scores": False}
        kwargs.update(self._cfg.extra_translate_kwargs)  # user wins, last word

        results = self._run_batch(source, target_prefix, kwargs)
        return [
            (t.display, tok.decode(list(r.hypotheses[0])))
            for t, r in zip(targets, results)
        ]

    # -- internals ---------------------------------------------------------

    def _translator_ctor(self):
        """The translator factory, importing ``ctranslate2`` lazily if needed."""
        factory = self._translator_factory
        if factory is None:
            import ctranslate2

            factory = ctranslate2.Translator
        return factory

    def _build_translator(self, device: str, index: int, compute: str):
        """Construct a translator and record the device/compute it runs on."""
        translator = self._translator_ctor()(
            str(self._model_dir),
            device=device,
            device_index=index,
            compute_type=compute,
            inter_threads=self._cfg.inter_threads,
            intra_threads=self._cfg.intra_threads or 0,
            max_queued_batches=self._cfg.max_queued_batches,
        )
        self._device = device
        self._compute_type = compute
        return translator

    def _fallback_to_cpu(self, detail: str) -> None:
        """Announce the CUDA fallback and rebuild the translator on CPU int8."""
        logger.warning("MT cannot use CUDA; falling back to CPU int8: %s", detail)
        self._bus.publish(EngineStateChanged("mt", "fallback_cpu", detail))
        self._translator = self._build_translator(*_CPU_FALLBACK)

    def _run_batch(self, source, target_prefix, kwargs):
        """Call ``translate_batch``, falling back to CPU int8 once when CUDA
        is unusable.

        ``fallback_cpu`` is transient (like the load path), so a successful retry
        re-publishes ``("mt", "ready")`` with the new device; a failed retry
        propagates without a ready event.
        """
        try:
            return self._translator.translate_batch(
                source, target_prefix=target_prefix, **kwargs
            )
        except RuntimeError as exc:
            if not _is_cuda_unusable(exc):
                raise
            self._fallback_to_cpu(str(exc))
            results = self._translator.translate_batch(
                source, target_prefix=target_prefix, **kwargs
            )
            self._bus.publish(
                EngineStateChanged(
                    "mt", "ready", f"{self._device}:{self._compute_type}"
                )
            )
            return results
