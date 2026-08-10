"""What the first-run wizard is about to download, in words.

Split out of :mod:`vrcc.gui.firstrun` for the 500-line cap. Functions take the
wizard, the way :mod:`vrcc.gui.firstrun_languages` does.

Everything here is a pure read of the wizard's current state: the plan is
recomputed on every device change and every language tick, so nothing in this
module creates a widget or writes to config.
"""

from __future__ import annotations

from vrcc.core import hardware
from vrcc.gui.model_labels import fmt_size, mt_display_name, whisper_display_name
from vrcc.i18n import tr
from vrcc.stt.registry import WHISPER_MODELS
from vrcc.translate.registry import MT_MODELS

__all__ = ["pending_mb", "summary_lines", "voice_device_note"]

def pending_mb(wizard) -> int:
    """Megabytes the plan still has to fetch.

    Models the download manager already reports present are left out. "Choose
    existing models" can fetch either half of the plan and come back with the
    wizard still open, and quoting the whole plan there names a download that is
    mostly done.
    """
    total = 0
    if not wizard._dm.is_whisper_downloaded(wizard.recommended_whisper):
        total += WHISPER_MODELS[wizard.recommended_whisper].size_mb
    if wizard._translation_enabled():
        spec = MT_MODELS[wizard.recommended_mt]
        if not wizard._dm.is_mt_downloaded(spec):
            total += spec.size_mb
    return total


def voice_device_note(wizard) -> str | None:
    """Why the GPU choice leaves the voice model on the processor, when it does.

    GPU writes ``stt.device = "auto"``, and under "auto" the onnxruntime voice
    models resolve to the processor even on a usable card
    (:func:`vrcc.core.hardware.resolved_device`, which is asked here rather than
    guessed from the backend). Without this line Settings reads "Auto: using
    your processor" straight after the user picked GPU, and neither screen says
    why. The translation model still runs on the card, so the GPU segment is not
    greyed.
    """
    if wizard._cpu_chosen():
        return None
    index = wizard._store.config.stt.device_index
    if hardware.resolved_device("auto", index, wizard.recommended_whisper) != "cpu":
        return None
    return tr(
        "{label} runs on your processor either way. It is about as fast there "
        "and leaves your graphics card free for VRChat.",
        label=whisper_display_name(wizard.recommended_whisper),
    )


def _cpu_tier_label() -> str:
    """Detected-line label for the "cpu" tier. That tier also covers a visible
    CUDA device this install cannot drive (no loadable cuBLAS), where "no
    graphics card" would be plainly false."""
    if hardware.cuda_device_count() > 0:
        return tr("graphics card that this version cannot use, using your processor")
    return tr("no graphics card, using your processor")


def _tier_label(tier: str) -> str:
    return {
        "gpu_high": tr("fast graphics card"),
        "gpu_low": tr("graphics card"),
        "cpu": _cpu_tier_label(),
    }[tier]


def summary_lines(wizard) -> list[str]:
    """The Detected / Speech / Translation / Total block, ready to join."""
    whisper = WHISPER_MODELS[wizard.recommended_whisper]
    lines = [
        tr("Detected: {tier}", tier=_tier_label(wizard.tier)), "",
        tr("Speech: {label} ({size})",
           label=whisper_display_name(wizard.recommended_whisper),
           size=fmt_size(whisper.size_mb)),
    ]
    note = voice_device_note(wizard)
    if note is not None:
        lines.append(note)
    if wizard._translation_enabled():
        mt = MT_MODELS[wizard.recommended_mt]
        lines.append(tr("Translation: {label} ({size})",
                        label=mt_display_name(mt.id), size=fmt_size(mt.size_mb)))
    lines.append("")
    lines.append(tr("Total download: {size}", size=fmt_size(pending_mb(wizard))))
    return lines
