"""Settings helpers that need config/hardware reasoning: the muted "Auto is
using your GPU/CPU" labels under the Advanced device combos, and the
"Reset to recommended settings" flow. Each function takes the ``SettingsDialog``
and reaches into its widgets (the same pattern as ``settings_advanced``); this
module imports no page builder, so it never cycles with them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from vrcc.core import recommend
from vrcc.core.hardware import resolved_device
from vrcc.gui import model_prompts, settings_live
from vrcc.gui.model_labels import mt_display_name, whisper_display_name
from vrcc.i18n import tr, tr_noop

if TYPE_CHECKING:
    from vrcc.gui.settings import SettingsDialog

_AUTO = "auto"

_AUTO_GPU_TEXT = tr_noop("Auto: using your graphics card")
_AUTO_CPU_TEXT = tr_noop("Auto: using your processor")

_RESET_BTN = tr_noop("Reset to recommended settings")
_RESET_TIP = tr_noop("Pick the models and settings that suit this machine.")
_RESET_TITLE = tr_noop("Reset to recommended settings?")
_RESET_BODY = tr_noop(
    "This picks the voice and translation models for your hardware and spoken "
    "language, returns the device and thread settings to automatic, and sets "
    "the performance mode. Your languages, microphone, OSC address and "
    "appearance stay as they are."
)
_RESET_MODELS = tr_noop("Voice model: {voice}. Translation model: {translate}.")

_RESET_DEFAULTS_BTN = tr_noop("Reset to default settings")
_RESET_DEFAULTS_TIP = tr_noop(
    "Put the tuning back to normal. Keeps your microphone, languages, "
    "models, chatbox address and appearance."
)
_RESET_DEFAULTS_TITLE = tr_noop("Reset to default settings?")
_RESET_DEFAULTS_BODY = tr_noop(
    "This puts sensitivity, microphone boost, timing and other tuning back "
    "to their normal values. Your microphone, languages, models, chatbox "
    "address and appearance stay as they are."
)

# Tuning fields reset to AppConfig() defaults; everything else is preserved.
_RESET_FIELDS = {
    "vad": (
        "threshold", "silence_threshold", "speculative_silence_ms",
        "finalize_silence_ms", "min_utterance_ms", "pre_roll_ms",
        "max_utterance_s", "sentence_inject", "sentence_min_words",
    ),
    "audio": ("gain_db", "auto_gain", "energy_gate_enabled", "energy_threshold"),
    "stt": (
        "beam_size", "temperature", "avg_logprob_gate", "no_speech_gate",
        "no_repeat_ngram_size", "compression_ratio_gate",
        "condition_on_previous_text",
    ),
    "translate": ("beam_size", "repetition_penalty", "no_repeat_ngram_size"),
    "gui": ("profile", "update_check_enabled"),
}

# (dialog attr, config section, field) for the automatic-reset thread spins,
# re-synced from config after a recommended reset.
_THREAD_SPINS = (
    ("_stt_cpu_threads_spin", "stt", "cpu_threads"),
    ("_stt_workers_spin", "stt", "num_workers"),
    ("_mt_inter_spin", "translate", "inter_threads"),
    ("_mt_intra_spin", "translate", "intra_threads"),
    ("_mt_queued_spin", "translate", "max_queued_batches"),
)


# -- Auto device labels ------------------------------------------------------


def reset_button_text() -> str:
    return tr(_RESET_BTN)


def reset_button_tooltip() -> str:
    return tr(_RESET_TIP)


def reset_defaults_button_text() -> str:
    return tr(_RESET_DEFAULTS_BTN)


def reset_defaults_button_tooltip() -> str:
    return tr(_RESET_DEFAULTS_TIP)


def _auto_device_text(resolved: str) -> str:
    return tr(_AUTO_GPU_TEXT if resolved == "cuda" else _AUTO_CPU_TEXT)


def _update_one(label, section, model_id) -> None:
    if label is None:
        return
    if section.device != _AUTO:
        label.setVisible(False)
        return
    resolved = resolved_device(section.device, section.device_index, model_id)
    label.setText(_auto_device_text(resolved))
    label.setVisible(True)


def update_device_auto_labels(dlg: "SettingsDialog") -> None:
    """Show each device combo's Auto label (which concrete device Auto picks)
    only while that combo is on Auto. The STT label depends on the voice model
    (the onnx-asr auto->cpu override), the translate label does not."""
    _update_one(getattr(dlg, "_stt_device_auto_label", None), dlg._cfg.stt, dlg._cfg.stt.model)
    _update_one(getattr(dlg, "_mt_device_auto_label", None), dlg._cfg.translate, None)


def refresh_after_stt_device(dlg: "SettingsDialog") -> None:
    """STT device changed: the Auto label and the Mode recommendation (which
    reads the resolved device) both depend on it."""
    update_device_auto_labels(dlg)
    dlg._update_mode_for_model()


# -- Reset to recommended settings -------------------------------------------


def confirm_and_reset(dlg: "SettingsDialog") -> None:
    """Preview the recommended outcome on a config copy, confirm while naming
    the concrete models, and on Yes apply for real. No changes on No. Works
    headless (``apply`` / ``download_manager`` None)."""
    from PySide6.QtWidgets import QMessageBox

    preview = dlg._cfg.model_copy(deep=True)
    outcome = recommend.reset_to_recommended(preview, dlg._download_manager)
    body = tr(_RESET_BODY)
    voice, translate = outcome["stt_model"], outcome["mt_model"]
    if voice and translate:
        body = body + "\n\n" + tr(
            _RESET_MODELS,
            voice=whisper_display_name(voice),
            translate=mt_display_name(translate),
        )
    answer = QMessageBox.question(
        dlg,
        tr(_RESET_TITLE),
        body,
        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        QMessageBox.StandardButton.No,
    )
    if answer == QMessageBox.StandardButton.Yes:
        _apply_reset(dlg)


def _engine_snapshot(cfg, kind: str) -> tuple:
    if kind == "stt":
        return tuple(getattr(cfg.stt, f) for f in settings_live._STT_ENGINE_FIELDS)
    return tuple(getattr(cfg.translate, f) for f in settings_live._MT_ENGINE_FIELDS)


def _apply_reset(dlg: "SettingsDialog") -> None:
    before_stt_model = dlg._cfg.stt.model
    before_mt_model = dlg._cfg.translate.model
    stt_engine = _engine_snapshot(dlg._cfg, "stt")
    mt_engine = _engine_snapshot(dlg._cfg, "mt")

    recommend.reset_to_recommended(dlg._cfg, dlg._download_manager)

    dlg._loading = True
    try:
        _resync_all_widgets(dlg)
    finally:
        dlg._loading = False

    dlg._store.save_soon()
    # The reset's apply_profile changed the VAD timings; push them to the
    # running Segmenter before the re-baseline below swallows the diff. VAD is
    # the only non-engine live group the reset can move: audio/osc/mute are
    # personal fields it never writes, and the gui group's live value is
    # font_scale, not gui.profile.
    if dlg._apply is not None:
        dlg._apply.apply_vad(dlg._cfg.vad)
    # Re-baseline every live-apply group so the debounced/close flush cannot
    # rebuild an engine this reset already pushed by hand.
    dlg._applied = settings_live.snapshot(dlg._specs())

    _push_engine_reset(dlg, "stt", before_stt_model, stt_engine)
    _push_engine_reset(dlg, "mt", before_mt_model, mt_engine)


def _push_engine_reset(dlg, kind: str, before_model: str, before_engine: tuple) -> None:
    """Rebuild ``kind`` exactly once. A model change takes the model-swap path
    (which rebuilds with the reset device/compute), so it must NOT also
    reload_engine; a same-model engine-field change forces one reload_engine."""
    section = dlg._cfg.stt if kind == "stt" else dlg._cfg.translate
    if section.model != before_model:
        if dlg._on_model_change is not None:
            dlg._on_model_change(kind)
        return
    if _engine_snapshot(dlg._cfg, kind) != before_engine and dlg._apply is not None:
        dlg._apply.reload_engine(kind)


def _select_data(combo, data) -> None:
    idx = combo.findData(data)
    if idx >= 0:
        combo.setCurrentIndex(idx)


def _select_device(combo, device: str, index: int) -> None:
    for i in range(combo.count()):
        if combo.itemData(i) == (device, index):
            combo.setCurrentIndex(i)
            return


def _resync_all_widgets(dlg: "SettingsDialog") -> None:
    """Re-read config into every widget the reset touched. Call under the
    ``_loading`` guard so no bound handler re-fires or re-saves."""
    cfg = dlg._cfg
    if dlg._model_combo is not None:
        _select_data(dlg._model_combo, cfg.stt.model)
        dlg._voice_selected_id = dlg._model_combo.currentData()
    tmc = getattr(dlg, "_translate_model_combo", None)
    if tmc is not None:
        _select_data(tmc, cfg.translate.model)
        dlg._mt_selected_id = tmc.currentData()
    _select_device(dlg._stt_device_combo, cfg.stt.device, cfg.stt.device_index)
    dlg._stt_compute_combo.setCurrentText(cfg.stt.compute_type)
    _select_device(dlg._mt_device_combo, cfg.translate.device, cfg.translate.device_index)
    dlg._mt_compute_combo.setCurrentText(cfg.translate.compute_type)
    for attr, section, field in _THREAD_SPINS:
        spin = getattr(dlg, attr, None)
        if spin is not None:
            spin.setValue(getattr(getattr(cfg, section), field))
    for field, spin in dlg._vad_spins.items():
        spin.setValue(getattr(cfg.vad, field))
    if dlg._stt_beam_spin is not None:
        dlg._stt_beam_spin.setValue(cfg.stt.beam_size)
    if dlg._stt_temp_spin is not None:
        dlg._stt_temp_spin.setValue(cfg.stt.temperature)
    if dlg._stt_norepeat_spin is not None:
        dlg._stt_norepeat_spin.setValue(cfg.stt.no_repeat_ngram_size)
    if dlg._stt_compression_spin is not None:
        dlg._stt_compression_spin.setValue(cfg.stt.compression_ratio_gate)
    if dlg._mt_beam_spin is not None:
        dlg._mt_beam_spin.setValue(cfg.translate.beam_size)
    if dlg._mt_rep_spin is not None:
        dlg._mt_rep_spin.setValue(cfg.translate.repetition_penalty)
    if dlg._mt_norepeat_spin is not None:
        dlg._mt_norepeat_spin.setValue(cfg.translate.no_repeat_ngram_size)
    if dlg._mode is not None:
        dlg._mode.set_value("Quality" if cfg.gui.profile == "quality" else "Speed")
    dlg._update_mode_for_model()
    update_device_auto_labels(dlg)
    source = getattr(dlg, "_source_combo", None)
    if source is not None:
        dlg._update_language_limited_items()
        model_prompts.grey_unsupported_languages(
            source, cfg.stt.model, translating=cfg.translate.enabled
        )


# -- Reset to default settings -----------------------------------------------


def confirm_and_reset_defaults(dlg: "SettingsDialog") -> None:
    """Confirm, then put the tuning knobs back to ``AppConfig()`` defaults.
    Personal choices (mic, languages, models, OSC, appearance) are untouched.
    No changes on No; works headless (``apply`` None)."""
    from PySide6.QtWidgets import QMessageBox

    answer = QMessageBox.question(
        dlg,
        tr(_RESET_DEFAULTS_TITLE),
        tr(_RESET_DEFAULTS_BODY),
        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        QMessageBox.StandardButton.No,
    )
    if answer == QMessageBox.StandardButton.Yes:
        _apply_reset_defaults(dlg)


def _apply_reset_defaults(dlg: "SettingsDialog") -> None:
    from vrcc.core.config import AppConfig

    defaults = AppConfig()
    for section, fields in _RESET_FIELDS.items():
        target = getattr(dlg._cfg, section)
        source = getattr(defaults, section)
        for field in fields:
            setattr(target, field, getattr(source, field))

    dlg._loading = True
    try:
        _resync_reset_widgets(dlg)
    finally:
        dlg._loading = False

    dlg._store.save_soon()
    if dlg._apply is not None:
        dlg._apply.apply_vad(dlg._cfg.vad)
        dlg._apply.apply_audio_gain(dlg._cfg.audio)
    dlg._applied = settings_live.snapshot(dlg._specs())


def _resync_reset_widgets(dlg: "SettingsDialog") -> None:
    """Re-read every reset field into its widget (under the ``_loading``
    guard), so an open dialog reflects the reset immediately."""
    cfg = dlg._cfg
    sens = getattr(dlg, "_sensitivity", None)
    if sens is not None:
        sens.setValue(90 - int(round(cfg.vad.threshold * 100)))
    for field, spin in dlg._vad_spins.items():
        spin.setValue(getattr(cfg.vad, field))
    inject = getattr(dlg, "_sentence_inject_check", None)
    if inject is not None:
        inject.setChecked(cfg.vad.sentence_inject)

    gain = getattr(dlg, "_gain_slider", None)
    if gain is not None:
        gain.setValue(int(round(cfg.audio.gain_db)))
    auto = getattr(dlg, "_auto_gain_check", None)
    if auto is not None:
        auto.setChecked(cfg.audio.auto_gain)
    if gain is not None:
        # The resync runs under _loading, so auto-gain's toggled handler
        # (which normally flips this) never fires: set it by hand.
        gain.setEnabled(not cfg.audio.auto_gain)
    gate = getattr(dlg, "_gate_check", None)
    if gate is not None:
        gate.setChecked(cfg.audio.energy_gate_enabled)
    noise = getattr(dlg, "_noise_slider", None)
    if noise is not None:
        noise.setValue(cfg.audio.energy_threshold)
    noise_label = getattr(dlg, "_noise_value_label", None)
    if noise_label is not None:
        noise_label.setText(str(cfg.audio.energy_threshold))

    stt_beam = getattr(dlg, "_stt_beam_spin", None)
    if stt_beam is not None:
        stt_beam.setValue(cfg.stt.beam_size)
    stt_temp = getattr(dlg, "_stt_temp_spin", None)
    if stt_temp is not None:
        stt_temp.setValue(cfg.stt.temperature)
    stt_norepeat = getattr(dlg, "_stt_norepeat_spin", None)
    if stt_norepeat is not None:
        stt_norepeat.setValue(cfg.stt.no_repeat_ngram_size)
    stt_compression = getattr(dlg, "_stt_compression_spin", None)
    if stt_compression is not None:
        stt_compression.setValue(cfg.stt.compression_ratio_gate)

    mt_beam = getattr(dlg, "_mt_beam_spin", None)
    if mt_beam is not None:
        mt_beam.setValue(cfg.translate.beam_size)
    mt_rep = getattr(dlg, "_mt_rep_spin", None)
    if mt_rep is not None:
        mt_rep.setValue(cfg.translate.repetition_penalty)
    mt_norepeat = getattr(dlg, "_mt_norepeat_spin", None)
    if mt_norepeat is not None:
        mt_norepeat.setValue(cfg.translate.no_repeat_ngram_size)

    update_check = getattr(dlg, "_update_check", None)
    if update_check is not None:
        update_check.setChecked(cfg.gui.update_check_enabled)

    mode = getattr(dlg, "_mode", None)
    if mode is not None:
        mode.set_value("Quality" if cfg.gui.profile == "quality" else "Speed")
