"""Plain-language warnings for a model the user is about to load or download:
does it fit the card, is there room on disk, and can it write the languages the
captions ask for. Advisory heuristics only; the engines' VRAM-OOM-to-CPU
fallback is the real safety net. No jargon in the returned sentences ("graphics
card" / "processor", never "VRAM"/"GPU").

The row-warning half at the bottom is here rather than in
:mod:`vrcc.gui.models_dialog` for the 500-line cap, and because it is Qt-free:
what a model warns about is a property of the config and the machine, not of
the widget showing it."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from vrcc.core import hardware, recommend
from vrcc.core.bench_tables import mt_vram_table, stt_vram_table
from vrcc.gui import mt_prompts
from vrcc.i18n import tr
from vrcc.stt.registry import WHISPER_MODELS
from vrcc.translate.registry import MT_MODELS

logger = logging.getLogger("vrcc.gui.model_fit")

# Fallback only, for a model with no measured row. On-disk size is a poor proxy
# (STT_VRAM_MB's note records peaks running 0.89x to 6.76x the file, and not
# even in the same order), so a measured peak is preferred wherever one exists. Kept
# deliberately low rather than raised to the worst observed ratio: over-warning
# on every small model would train the user to dismiss the prompt.
_VRAM_OVERHEAD = 1.2
_DISK_OVERHEAD = 1.1


def _human(size_mb: float) -> str:
    if size_mb >= 1000:
        return tr("about {gb:.1f} GB", gb=size_mb / 1000)
    return tr("about {mb} MB", mb=int(size_mb))


def _resolved_compute(compute_type: str, device_index: int) -> str:
    """What the engines will actually run at. One implementation with the
    ranking's, in ``recommend``, because the two size against the same table
    and disagreeing puts a fit warning on a row the recommender just picked."""
    return recommend.resolved_compute_type(compute_type, device_index)


def vram_warning(
    size_mb: int, device: str = "auto", model_id: str | None = None,
    device_index: int = 0, compute_type: str = "auto",
) -> str | None:
    """Warn when a model likely won't fit on the graphics card. ``None`` if it
    fits, if there's no graphics card / unknown VRAM, or if the model is set to
    run on the processor (``device == "cpu"``).

    ``model_id`` selects the measured peak in preference to scaling
    ``size_mb``, and ``device_index`` names the card the engines are pinned to,
    so this agrees with the budget the recommender applies to the same card
    rather than sizing against card 0 on a mixed multi-GPU box. Without it (or
    for an id with no row) the size heuristic stands in, which is the only
    reason ``size_mb`` is still taken.

    ``compute_type`` picks WHICH measured peak: the same model costs 1.13x to
    1.67x more at float16 than at int8_float16, and a card on compute
    capability 12 or above has no int8 kernels, so it always pays the higher
    one. Sizing a Blackwell card off the int8 table left large-v3 silent on a
    12 GB card at a real 4379 MB against a 4093 MB budget.
    """
    if device == "cpu":
        return None
    total = hardware.total_vram_bytes(device_index)
    if total is None:
        return None
    # Both tables: the ids are disjoint, and this is called for translation
    # models too. Against the voice table alone every MT id missed and fell
    # back to the size heuristic, which reads nllb-1.3B at 1680 MB against a
    # measured 4273 at float16, so a 12 GB card was told it fits when it does
    # not.
    compute = _resolved_compute(compute_type, device_index)
    peak_mb = None
    if model_id:
        peak_mb = stt_vram_table(compute).get(model_id)
        if peak_mb is None:
            peak_mb = mt_vram_table(compute).get(model_id)
    need_mb = peak_mb if peak_mb is not None else size_mb * _VRAM_OVERHEAD
    if need_mb <= recommend.vram_budget_mb(total // 1024**2):
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


# -- Models-window row warnings ---------------------------------------------


def fit_notes(cfg) -> dict[str, str]:
    """Model id -> the graphics-card warning Settings would give that model.

    Built in one pass so a window listing every model probes the card once per
    model at construction instead of once per row on every re-render. Each
    section is sized against the device and card THAT section runs on, the way
    Settings does, so the two surfaces cannot disagree about one model.
    """
    notes: dict[str, str] = {}
    for specs, section in (
        (WHISPER_MODELS.values(), cfg.stt), (MT_MODELS.values(), cfg.translate)
    ):
        for spec in specs:
            device = hardware.resolved_device(
                section.device, section.device_index, spec.id
            )
            msg = vram_warning(
                spec.size_mb, device, spec.id, section.device_index,
                section.compute_type,
            )
            if msg:
                notes[spec.id] = msg
    return notes


def collapsed_target(cfg, model_id: str) -> tuple[str, str] | None:
    """A language ``cfg.translate.targets`` asks for that ``model_id`` cannot
    write distinctly, paired with the language it writes instead. ``None`` when
    the model keeps every configured target apart.
    """
    for target in cfg.translate.targets:
        other = mt_prompts.collapsed_target(model_id, target)
        if other is not None:
            return target, other
    return None


def row_note(cfg, kind: str, model_id: str, display_name: str, fits: dict) -> str:
    """The warnings one Models-window row carries, or ``""``.

    Both of them existed elsewhere and neither reached that window: the wizard
    greys a target the MT family collapses, Settings warns when a model will
    not fit the card. Downloading is where the cost is paid, so they belong
    there first. ``fits`` is a :func:`fit_notes` result, passed in rather than
    recomputed per row.
    """
    notes = []
    collapsed = collapsed_target(cfg, model_id) if kind == "mt" else None
    if collapsed is not None:
        notes.append(
            tr(
                "Your captions are translated into {language}. "
                "{name} writes that as {other}.",
                language=collapsed[0], name=display_name, other=collapsed[1],
            )
        )
    fit = fits.get(model_id)
    if fit:
        notes.append(fit)
    return " ".join(notes)
