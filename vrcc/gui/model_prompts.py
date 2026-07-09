"""Question prompts about the active voice model, shared by Settings and the
main window: offer the CPU when an onnx-asr model meets an explicit CUDA
device, and offer a better downloaded model when the spoken language changes
to one the active model cannot transcribe. The decision helpers are Qt-free;
only the offer functions build a dialog.
"""

from __future__ import annotations

from vrcc.core import recommend
from vrcc.core.languages import LANGUAGES
from vrcc.gui.model_labels import whisper_display_name
from vrcc.i18n import tr, tr_noop
from vrcc.stt.registry import WHISPER_MODELS

# Hedged on purpose ("usually"): relative CPU/GPU speed varies by machine; on
# the reference box the int8 exports measured no faster on CUDA than CPU.
_CPU_OFFER = tr_noop(
    "Parakeet and Canary usually run about as fast on the CPU as on the GPU, "
    "and GPU mode takes VRAM away from VRChat. Use the CPU for this model?"
)

_SWITCH_OFFER = tr_noop("{name} cannot transcribe {language}. Switch to {other}?")


def cpu_offer_needed(cfg, model_id: str) -> bool:
    """Only an EXPLICIT ``cuda`` device warrants the CPU offer: "auto" already
    runs onnx-asr models on the CPU (OnnxAsrEngine's deliberate resolution)."""
    spec = WHISPER_MODELS.get(model_id)
    return (
        spec is not None and spec.backend == "onnx_asr" and cfg.stt.device == "cuda"
    )


def maybe_prefer_cpu(dlg, model_id: str) -> None:
    """Settings hook for both explicit-CUDA triggers (an onnx-asr model picked
    while on cuda; the device switched to cuda under an onnx-asr model): ask,
    and on Yes flip ``stt.device`` to "cpu" and re-point the Advanced device
    combo without re-firing its handler."""
    if not cpu_offer_needed(dlg._cfg, model_id):
        return
    from PySide6.QtWidgets import QMessageBox

    answer = QMessageBox.question(
        dlg,
        tr("Use the CPU?"),
        tr(_CPU_OFFER),
        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        QMessageBox.StandardButton.Yes,
    )
    if answer != QMessageBox.StandardButton.Yes:
        return
    dlg._cfg.stt.device = "cpu"
    dlg._cfg.stt.device_index = 0
    combo = dlg._stt_device_combo
    # itemData comparison by loop: findData is unreliable for tuple data
    # (same reason _make_device_combo selects its initial entry by loop).
    for i in range(combo.count()):
        if combo.itemData(i) == ("cpu", 0):
            dlg._loading = True
            try:
                combo.setCurrentIndex(i)
            finally:
                dlg._loading = False
            break
    dlg._changed()


def _covers(spec, code: str) -> bool:
    """Whether ``spec`` can transcribe the Whisper language ``code`` (the same
    english_only-on-top-of-languages reading recommend._rank_whisper uses)."""
    if spec.english_only and code != "en":
        return False
    return spec.languages is None or code in spec.languages


def propose_language_switch(cfg, dm, source_display: str) -> str | None:
    """A downloaded voice model suited to ``source_display`` when the active
    one cannot transcribe it, else ``None`` (also for "auto" / an unknown
    name / no download manager / nothing compatible downloaded). Qt-free."""
    if dm is None:
        return None
    spec = WHISPER_MODELS.get(cfg.stt.model)
    lang = LANGUAGES.get(source_display)  # "auto" has no registry entry
    if spec is None or lang is None or _covers(spec, lang.whisper):
        return None
    candidate, _ = recommend.best_downloaded(
        dm,
        translate=False,
        tier=recommend.tier_for_config(cfg),
        language=lang.whisper,
    )
    if candidate is None or candidate == cfg.stt.model:
        return None
    if not _covers(WHISPER_MODELS[candidate], lang.whisper):
        # Every downloaded model is equally unable; a swap would not help.
        return None
    return candidate


def offer_language_switch(parent, cfg, dm, source_display: str) -> str | None:
    """Ask about a better model for the newly chosen spoken language. Returns
    the accepted model id, or ``None`` (no candidate, or the user declined).
    The caller applies the switch through its own model-change path."""
    candidate = propose_language_switch(cfg, dm, source_display)
    if candidate is None:
        return None
    from PySide6.QtWidgets import QMessageBox

    answer = QMessageBox.question(
        parent,
        tr("Switch voice model?"),
        tr(
            _SWITCH_OFFER,
            name=whisper_display_name(cfg.stt.model),
            language=source_display,
            other=whisper_display_name(candidate),
        ),
        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        QMessageBox.StandardButton.Yes,
    )
    return candidate if answer == QMessageBox.StandardButton.Yes else None


def maybe_switch_model_for_language(dlg) -> None:
    """Settings hook: apply an accepted switch through the voice-model combo,
    so the normal change path runs (fit prompt, Mode lock, hot-swap)."""
    candidate = offer_language_switch(
        dlg, dlg._cfg, dlg._download_manager, dlg._cfg.stt.source_language
    )
    if candidate is None or dlg._model_combo is None:
        return
    idx = dlg._model_combo.findData(candidate)
    if idx >= 0:
        dlg._model_combo.setCurrentIndex(idx)
