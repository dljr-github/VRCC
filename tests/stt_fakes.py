"""Shared fakes for the faster-whisper STT engine tests: a recording model
factory, a canned model, segment/config builders, and the CUDA-unusable error
texts. Imported by the STT engine test modules so no faster-whisper is needed.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from vrcc.core.bus import EventBus
from vrcc.core.config import SttConfig
from vrcc.core.events import EngineStateChanged

_OOM_TEXT = "CUDA failed with error out of memory (CUBLAS_STATUS_ALLOC_FAILED)"
# Verbatim CTranslate2 dynamic-loader error from a CPU-only install driving a
# visible GPU: cudart is statically linked, so the device enumerates, and the
# load dies at model build or the first CUDA op instead.
_MISSING_LIBRARY_TEXT = "Library cublas64_12.dll is not found or cannot be loaded"

# Both shapes of CUDA-unusable RuntimeError must take the one-shot CPU fallback.
_CUDA_UNUSABLE_TEXTS = [_OOM_TEXT, _MISSING_LIBRARY_TEXT]


def _seg(text, start, end, avg_logprob, no_speech_prob, compression_ratio=1.0):
    return SimpleNamespace(
        text=text, start=start, end=end, avg_logprob=avg_logprob,
        no_speech_prob=no_speech_prob, compression_ratio=compression_ratio,
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
