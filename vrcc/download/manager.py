"""Download, presence-check and delete STT/MT model files.

Lays models out as ``<models_dir>/{mt,whisper}/<id>`` (``whisper/`` holds
every voice model -- the directory name is historical, the onnx-asr backed
NeMo exports land there too), delegating to ``snapshot_download`` (MT +
onnx-asr) / ``download_model`` (faster-whisper) and reporting
:class:`DownloadProgress` on the bus. Downloader errors propagate.
"""

from __future__ import annotations

import contextlib
import logging
import os
import shutil
from pathlib import Path
from typing import Callable, TypeVar

import huggingface_hub
from faster_whisper import download_model
from huggingface_hub import snapshot_download
from huggingface_hub.utils import tqdm as _HfTqdm

from vrcc.core.bus import EventBus
from vrcc.core.events import DownloadProgress
from vrcc.stt.registry import WHISPER_MODELS, WhisperSpec
from vrcc.translate.registry import MtModelSpec

_MODEL_BIN = "model.bin"
_KINDS = ("mt", "whisper")
_HF_MIRROR = "https://hf-mirror.com"

_log = logging.getLogger(__name__)
_T = TypeVar("_T")


@contextlib.contextmanager
def _hf_endpoint(url: str):
    """Point every huggingface_hub download route at ``url`` for the
    duration of the block, then restore the prior state exactly.

    ``download_model`` calls ``snapshot_download`` internally with no
    ``endpoint`` argument, so it reads ``huggingface_hub.constants.ENDPOINT``
    at call time; overriding that constant (plus the env var some code paths
    re-read) redirects both callers with one mechanism.
    """
    prior_env = os.environ.get("HF_ENDPOINT")
    prior_const = huggingface_hub.constants.ENDPOINT
    os.environ["HF_ENDPOINT"] = url
    huggingface_hub.constants.ENDPOINT = url
    try:
        yield
    finally:
        if prior_env is None:
            os.environ.pop("HF_ENDPOINT", None)
        else:
            os.environ["HF_ENDPOINT"] = prior_env
        huggingface_hub.constants.ENDPOINT = prior_const


def _with_mirror_fallback(primary: Callable[[], _T]) -> _T:
    """Run ``primary``; on failure retry once against the hf-mirror.com
    endpoint (skipped if that mirror is already active, so a mirror failure
    doesn't retry itself). A retry failure chains from the original error so
    the caller's dialog still shows a real cause."""
    try:
        return primary()
    except Exception as primary_err:
        if huggingface_hub.constants.ENDPOINT == _HF_MIRROR:
            raise
        _log.info("Primary download failed, retrying via hf-mirror.com: %s", primary_err)
        try:
            with _hf_endpoint(_HF_MIRROR):
                return primary()
        except Exception as mirror_err:
            raise mirror_err from primary_err


def _onnx_asr_files(spec: WhisperSpec) -> list[str]:
    """The exact files onnx-asr needs to run ``spec`` offline (doubles as the
    snapshot allow_patterns, so nothing else is fetched). The transducer
    exports split into encoder + decoder_joint."""
    suffix = f".{spec.quantization}" if spec.quantization else ""
    return [
        "config.json",
        "vocab.txt",
        f"encoder-model{suffix}.onnx",
        f"decoder_joint-model{suffix}.onnx",
    ]


class DownloadManager:
    def __init__(self, models_dir: Path, bus: EventBus) -> None:
        self._models_dir = Path(models_dir)
        self._bus = bus

    # -- paths -------------------------------------------------------------

    @property
    def models_dir(self) -> Path:
        return self._models_dir

    def mt_model_dir(self, spec: MtModelSpec) -> Path:
        return self._models_dir / "mt" / spec.id

    def whisper_model_dir(self, model_id: str) -> Path:
        return self._models_dir / "whisper" / model_id

    # -- presence checks ---------------------------------------------------

    def is_mt_downloaded(self, spec: MtModelSpec) -> bool:
        d = self.mt_model_dir(spec)
        return (d / _MODEL_BIN).is_file() and (d / spec.spm_file).is_file()

    def is_whisper_downloaded(self, model_id: str) -> bool:
        d = self.whisper_model_dir(model_id)
        spec = WHISPER_MODELS.get(model_id)
        if spec is not None and spec.backend == "onnx_asr":
            return all((d / name).is_file() for name in _onnx_asr_files(spec))
        return (d / _MODEL_BIN).is_file()

    # -- downloads ---------------------------------------------------------

    def ensure_mt(self, spec: MtModelSpec) -> Path:
        """Download the MT model unless already present.

        Publishes a stream of :class:`DownloadProgress` and one terminal
        ``done=True``. Returns the model dir; ``snapshot_download`` exceptions
        propagate.
        """
        target = self.mt_model_dir(spec)
        if self.is_mt_downloaded(spec):
            self._publish_done(spec.id)
            return target

        # Full-repo download (no allow_patterns): CT2 repos vary in filenames,
        # so an over-narrow pattern is riskier than a few extra kB.
        _with_mirror_fallback(
            lambda: snapshot_download(
                repo_id=spec.repo,
                local_dir=str(target),
                tqdm_class=self._progress_tqdm(spec.id),
            )
        )
        self._publish_done(spec.id)
        return target

    def ensure_whisper(self, model_id: str) -> Path:
        """Download the voice model unless already present.

        onnx-asr models (Parakeet) come from their HF repo via
        ``snapshot_download`` (with byte progress, restricted to the files
        onnx-asr needs); faster-whisper models via ``download_model`` (no
        byte-level progress hook, so only the terminal ``done=True`` is
        published). Returns the model dir; downloader exceptions propagate.
        """
        target = self.whisper_model_dir(model_id)
        if self.is_whisper_downloaded(model_id):
            self._publish_done(model_id)
            return target

        spec = WHISPER_MODELS.get(model_id)
        if spec is not None and spec.backend == "onnx_asr":
            _with_mirror_fallback(
                lambda: snapshot_download(
                    repo_id=spec.repo,
                    local_dir=str(target),
                    allow_patterns=_onnx_asr_files(spec),
                    tqdm_class=self._progress_tqdm(model_id),
                )
            )
        else:
            _with_mirror_fallback(lambda: download_model(model_id, output_dir=str(target)))
        self._publish_done(model_id)
        return target

    # -- delete ------------------------------------------------------------

    def delete(self, kind: str, model_id: str) -> None:
        """Remove a downloaded model's directory tree. ``kind`` is ``"mt"`` or
        ``"whisper"`` (else ``ValueError``); a missing dir is a no-op.
        """
        if kind == "mt":
            target = self._models_dir / "mt" / model_id
        elif kind == "whisper":
            target = self._models_dir / "whisper" / model_id
        else:
            raise ValueError(
                f"Unknown model kind: {kind!r}. Expected one of {list(_KINDS)}."
            )
        if target.exists():
            shutil.rmtree(target)

    # -- internals ---------------------------------------------------------

    def _progress_tqdm(self, model_id: str) -> type:
        """A tqdm class for ``snapshot_download`` that publishes
        :class:`DownloadProgress` for ``model_id`` from the aggregated BYTES
        bar only.

        snapshot_download builds its tqdm_class twice (bytes bar + file-count
        bar); only the bytes bar publishes (no mixed units). Bytes tracked on
        our own counter (a disabled bar doesn't advance ``self.n``); skipped
        until ``total`` is known.
        """
        bus = self._bus

        class _ProgressTqdm(_HfTqdm):  # type: ignore[misc, valid-type]
            def __init__(self, *args, **kwargs) -> None:
                self._publishes = kwargs.get("unit") == "B"
                self._bytes = int(kwargs.get("initial") or 0)
                # Suppress tqdm's own console bar: we render via the bus, and
                # writing to stderr (None in the windowed .exe) raised
                # 'NoneType has no write'. total/n are still grown when disabled,
                # so byte accounting is unaffected.
                kwargs["disable"] = True
                super().__init__(*args, **kwargs)

            def update(self, n: float | None = 1) -> bool | None:
                displayed = super().update(n)
                if self._publishes:
                    if n:
                        self._bytes += int(n)
                    total = int(self.total or 0)
                    if total:
                        bus.publish(DownloadProgress(model_id, self._bytes, total))
                return displayed

        return _ProgressTqdm

    def _publish_done(self, model_id: str) -> None:
        self._bus.publish(DownloadProgress(model_id, 1, 1, done=True))
