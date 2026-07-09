"""Plain-language memory-fit warnings for a model the user is about to load or
download. Advisory heuristics only; the engines' VRAM-OOM-to-CPU fallback is the
real safety net. No jargon in the returned sentences ("graphics card" /
"processor", never "VRAM"/"GPU")."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from vrcc.core import hardware
from vrcc.i18n import tr

logger = logging.getLogger("vrcc.gui.model_fit")

# Leave room for VRChat + the OS alongside the model.
_VRAM_HEADROOM_BYTES = 2 * 1024**3
# A model in int8 needs roughly its on-disk size in VRAM, plus working overhead.
_VRAM_OVERHEAD = 1.2
_DISK_OVERHEAD = 1.1


def _human(size_mb: float) -> str:
    if size_mb >= 1000:
        return tr("about {gb:.1f} GB", gb=size_mb / 1000)
    return tr("about {mb} MB", mb=int(size_mb))


def vram_warning(size_mb: int, device: str = "auto") -> str | None:
    """Warn when ``size_mb`` likely won't fit on the graphics card. ``None`` if
    it fits, if there's no graphics card / unknown VRAM, or if the model is set
    to run on the processor (``device == "cpu"``)."""
    if device == "cpu":
        return None
    total = hardware.total_vram_bytes()
    if total is None:
        return None
    need = int(size_mb * 1024**2 * _VRAM_OVERHEAD)
    if need <= total - _VRAM_HEADROOM_BYTES:
        return None
    return tr(
        "This model may be too large for your graphics card (~{gb:.0f} GB). "
        "It could run on your processor instead (slower) or fail to load.",
        gb=total / 1024**3,
    )


def disk_warning(models_dir, size_mb: int) -> str | None:
    """Warn when there isn't enough free disk space to download ``size_mb``.
    ``None`` when there's room or the free space can't be determined."""
    if models_dir is None:
        return None
    path = Path(models_dir)
    while not path.exists() and path != path.parent:
        path = path.parent
    try:
        free = shutil.disk_usage(path).free
    except OSError:
        logger.debug("disk_usage(%s) failed", path, exc_info=True)
        return None
    if free >= int(size_mb * 1024**2 * _DISK_OVERHEAD):
        return None
    return tr(
        "Not enough free disk space to download this (needs {size}, "
        "you have about {gb_free:.1f} GB free).",
        size=_human(size_mb),
        gb_free=free / 1024**3,
    )
