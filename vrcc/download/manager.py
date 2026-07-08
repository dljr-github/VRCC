"""Download, presence-check and delete STT/MT model files.

Lays models out as ``<models_dir>/{mt,whisper}/<id>``, delegating to
``snapshot_download`` (MT) / ``download_model`` (STT) and reporting
:class:`DownloadProgress` on the bus. Downloader errors propagate.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from faster_whisper import download_model
from huggingface_hub import snapshot_download
from huggingface_hub.utils import tqdm as _HfTqdm

from vrcc.core.bus import EventBus
from vrcc.core.events import DownloadProgress
from vrcc.translate.registry import MtModelSpec

_MODEL_BIN = "model.bin"
_KINDS = ("mt", "whisper")


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
        return (self.whisper_model_dir(model_id) / _MODEL_BIN).is_file()

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

        bus = self._bus
        model_id = spec.id

        class _ProgressTqdm(_HfTqdm):  # type: ignore[misc, valid-type]
            """Publish DownloadProgress from the aggregated BYTES bar only.

            snapshot_download builds its tqdm_class twice (bytes bar + file-count
            bar); only the bytes bar publishes (no mixed units). Bytes tracked on
            our own counter (a disabled bar doesn't advance ``self.n``); skipped
            until ``total`` is known.
            """

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

        # Full-repo download (no allow_patterns): CT2 repos vary in filenames,
        # so an over-narrow pattern is riskier than a few extra kB.
        snapshot_download(
            repo_id=spec.repo,
            local_dir=str(target),
            tqdm_class=_ProgressTqdm,
        )
        self._publish_done(spec.id)
        return target

    def ensure_whisper(self, model_id: str) -> Path:
        """Download the faster-whisper model unless already present.

        No byte-level progress hook here, so only the terminal ``done=True`` is
        published. Returns the model dir; ``download_model`` exceptions propagate.
        """
        target = self.whisper_model_dir(model_id)
        if self.is_whisper_downloaded(model_id):
            self._publish_done(model_id)
            return target

        download_model(model_id, output_dir=str(target))
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

    def _publish_done(self, model_id: str) -> None:
        self._bus.publish(DownloadProgress(model_id, 1, 1, done=True))
