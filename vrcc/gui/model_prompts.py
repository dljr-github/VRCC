"""Question prompts and combo greying for the active voice model, shared by
Settings and the main window: offer the CPU when an onnx-asr model meets an
explicit CUDA device, grey the spoken languages the active model cannot
transcribe, and (main window) offer a better downloaded model when the stored
spoken language sits outside the active model's set. Greying makes that state
unreachable interactively, so the nudge fires from config loads. The decision
helpers are Qt-free; the offer functions build a dialog and the greying edits
a combo.
"""

from __future__ import annotations

from vrcc.core import recommend
from vrcc.core.languages import LANGUAGES
from vrcc.gui.model_labels import whisper_display_name
from vrcc.i18n import tr, tr_noop
from vrcc.stt.registry import WHISPER_MODELS

_AUTO = "auto"

# Hedged on purpose ("usually"): relative CPU/GPU speed varies by machine; on
# the reference box the int8 exports measured no faster on CUDA than CPU.
_CPU_OFFER = tr_noop(
    "Parakeet usually runs about as fast on the CPU as on the GPU, "
    "and GPU mode takes VRAM away from VRChat. Use the CPU for this model?"
)

_SWITCH_OFFER = tr_noop("{name} cannot transcribe {language}. Switch to {other}?")

# Per-item tooltip on a spoken-language entry the active voice model can't do.
_LANGUAGE_LOCKED_TIP = tr_noop(
    "{name} cannot transcribe this language. Choose another voice model first."
)


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


def unsupported_stored_language(cfg) -> bool:
    """Whether the stored spoken language is outside the active voice model's
    set (the greying predicate): "auto" is unsupported when the model cannot
    detect the language itself; an unknown model or language (a hand-edited
    config) restricts nothing. Qt-free."""
    spec = WHISPER_MODELS.get(cfg.stt.model)
    if spec is None:
        return False
    if cfg.stt.source_language == _AUTO:
        return not spec.auto_language
    lang = LANGUAGES.get(cfg.stt.source_language)
    return lang is not None and not _covers(spec, lang.whisper)


def run_language_nudge(window) -> None:
    """Offer the best downloaded voice model for the main window's stored
    spoken language when the active model cannot transcribe it. Yes applies
    the switch through the window's save/on_model_change pair and re-greys
    the language combo; a decline is remembered as the (model, language)
    pair so config reloads do not re-ask until the mismatch changes. No-op
    without a candidate (including "auto", which names no target language)."""
    cfg = window._store.config
    source = cfg.stt.source_language
    candidate = propose_language_switch(cfg, window._download_manager, source)
    if candidate is None or window._nudge_declined == (cfg.stt.model, source):
        return
    from PySide6.QtWidgets import QMessageBox

    answer = QMessageBox.question(
        window,
        tr("Switch voice model?"),
        tr(
            _SWITCH_OFFER,
            name=whisper_display_name(cfg.stt.model),
            language=source,
            other=whisper_display_name(candidate),
        ),
        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        QMessageBox.StandardButton.Yes,
    )
    if answer != QMessageBox.StandardButton.Yes:
        window._nudge_declined = (cfg.stt.model, source)
        return
    cfg.stt.model = candidate
    window._store.save_soon()
    if window._on_model_change is not None:
        window._on_model_change("stt")
    grey_unsupported_languages(window._source_combo, cfg.stt.model)


def schedule_language_nudge(window) -> None:
    """Queue :func:`run_language_nudge` once for a stored language the active
    model cannot transcribe: greying makes that state unreachable from the
    combo popup, so without this the user gets a disabled-but-selected entry
    and silently wrong captions. Zero-delay so it runs after construction or
    reload settles; the pending flag keeps repeated reloads from stacking
    prompts. Headless windows (no download manager / model-change hook)
    never schedule."""
    if window._download_manager is None or window._on_model_change is None:
        return
    if window._nudge_pending or not unsupported_stored_language(window._store.config):
        return
    window._nudge_pending = True
    from PySide6.QtCore import QTimer

    QTimer.singleShot(0, lambda: _run_scheduled_nudge(window))


def _run_scheduled_nudge(window) -> None:
    import shiboken6

    window._nudge_pending = False
    # The queued shot can outlive the window (the UI-language rebuild closes
    # it): a dead wrapper must not parent a dialog. Re-check the mismatch too;
    # a reload may have resolved it while the shot was queued.
    if shiboken6.isValid(window) and unsupported_stored_language(window._store.config):
        run_language_nudge(window)


def grey_unsupported_languages(combo, model_id: str) -> None:
    """Disable the spoken-language entries the active voice model can't
    transcribe, each with a tooltip naming the model. "auto" is enabled only
    when the model detects the spoken language itself; an unknown model id
    (hand-edited config) restricts nothing. Symmetric to the model-combo
    greying: the user switches the voice model first, which re-enables the
    languages, so the two directions never deadlock."""
    spec = WHISPER_MODELS.get(model_id)
    tip = tr(_LANGUAGE_LOCKED_TIP, name=whisper_display_name(model_id))
    item_model = combo.model()
    for i in range(combo.count()):
        item = item_model.item(i)
        if item is None:
            continue
        text = combo.itemText(i)
        if text == _AUTO:
            enabled = spec is None or spec.auto_language
        else:
            lang = LANGUAGES.get(text)
            enabled = spec is None or lang is None or _covers(spec, lang.whisper)
        item.setEnabled(enabled)
        item.setToolTip("" if enabled else tip)
