"""Qt-free startup helpers: resolve the configured audio device, default the
caption language on first launch, and check whether the configured models are
already downloaded.

Extracted from :mod:`vrcc.app` so the composition root stays under the source
cap. :func:`vrcc.app.run` re-imports these under their private names, so the
monkeypatch targets (``vrcc.app._models_ready``, ``vrcc.app._resolve_audio_device``,
``vrcc.app._default_source_language``) keep resolving.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from vrcc.audio.devices import list_input_devices
from vrcc.core.languages import match_caption_language
from vrcc.translate.registry import MT_MODELS

if TYPE_CHECKING:
    from vrcc.download.manager import DownloadManager

logger = logging.getLogger("vrcc.core.startup")


def resolve_audio_device(device_cfg: str) -> int | None:
    """Turn ``audio.device`` config into a PortAudio device index (or ``None``).

    ``"auto"`` -> ``None`` (system default); otherwise look the configured
    device *name* up via :func:`list_input_devices`, falling back to ``None``
    if absent.
    """
    if device_cfg == "auto":
        return None
    for index, name in list_input_devices():
        if name == device_cfg:
            return index
    logger.warning(
        "configured audio device %r not found; using the system default",
        device_cfg,
    )
    return None


def default_source_language(cfg, locales: list[str]) -> None:
    """Default ``stt.source_language`` to the first OS display language the
    caption registry covers; an unmatched preference list keeps the "English"
    default. Callers gate on ``ConfigStore.missing_on_load`` so an existing
    config is never rewritten."""
    for name in locales:
        matched = match_caption_language(name)
        if matched is not None:
            cfg.stt.source_language = matched
            return


def models_ready(cfg, dm: "DownloadManager") -> bool:
    """True if the configured STT model (and MT model, when translation is on)
    are already downloaded -- i.e. the app can start without the wizard."""
    if not dm.is_whisper_downloaded(cfg.stt.model):
        return False
    if cfg.translate.enabled:
        spec = MT_MODELS.get(cfg.translate.model)
        if spec is None or not dm.is_mt_downloaded(spec):
            return False
    return True
