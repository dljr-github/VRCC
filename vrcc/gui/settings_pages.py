"""Page builders for the friendly Settings tabs (Simple / Voice / Translation).

Each ``build_*_page(dlg)`` returns the tab widget and writes live control refs
back onto ``dlg`` (the :class:`SettingsDialog`), reusing its bind/spin helpers.
Imports from ``settings`` are type-only (settings imports this, never reverse).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QLineEdit,
    QPushButton,
    QSlider,
    QWidget,
)
from PySide6.QtCore import Qt

from vrcc.core.languages import LANGUAGES
from vrcc.gui import model_prompts, settings_audio, settings_mode, settings_reset
from vrcc.gui.model_labels import mt_display_name, whisper_display_name
from vrcc.gui.widgets import combo_value, fill_spoken_languages, set_combo_value, SegmentedControl, no_wheel
from vrcc.i18n import UI_LANGUAGES, tr, tr_noop

if TYPE_CHECKING:
    from vrcc.gui.settings import SettingsDialog

_AUTO = "auto"

# Shown under the interface-language picker: tr() runs at construction, so the
# open dialog keeps the old language and only the main-window rebuild on close
# shows the new one. Without this the picker looks broken.
_UI_LANGUAGE_HINT = tr_noop("The new language appears when you close Settings.")

# Labels double as SegmentedControl values (compared/persisted via scale_map);
# tr_noop keeps them stable values while making them catalog-extractable for
# the dynamic tr(label) at the control build site.
_FONT_SCALE_PRESETS = [
    (tr_noop("Small"), 0.9),
    (tr_noop("Normal"), 1.0),
    (tr_noop("Large"), 1.2),
]
_DELETED_MODEL_TEXT = tr_noop("Current model (deleted) - choose another")


def _add_deleted_placeholder_if_needed(combo: QComboBox, specs, configured_id) -> None:
    if specs and not any(s.id == configured_id for s in specs):
        combo.addItem(tr(_DELETED_MODEL_TEXT), None)
        combo.model().item(0).setEnabled(False)


def build_simple_page(dlg: "SettingsDialog") -> QWidget:
    page = QWidget()
    form = QFormLayout(page)
    form.setContentsMargins(24, 16, 24, 16)

    form.addRow(tr("Microphone"), settings_audio.make_input_device_row(dlg))

    dlg._sensitivity = no_wheel(QSlider(Qt.Orientation.Horizontal))
    dlg._sensitivity.setRange(30, 60)
    # Higher = more sensitive = lower VAD speech threshold, so the slider reads
    # the way its label promises. threshold 0.60..0.30 maps to slider 30..60.
    dlg._sensitivity.setValue(90 - int(round(dlg._cfg.vad.threshold * 100)))
    dlg._sensitivity.setToolTip(
        tr(
            "How easily VRCC picks up your speech. Higher catches quieter or "
            "softer talking; lower ignores more."
        )
    )

    def on_sensitivity(v):
        if dlg._loading:
            return
        dlg._cfg.vad.threshold = (90 - v) / 100.0
        dlg._changed()
    dlg._sensitivity.valueChanged.connect(on_sensitivity)
    sens_row, dlg._sensitivity_low, dlg._sensitivity_high = dlg._anchored_slider(dlg._sensitivity)
    form.addRow(tr("Microphone sensitivity"), sens_row)

    settings_mode.build_mode_control(dlg, form)

    dlg._send_check = QCheckBox(tr("Send my captions to VRChat"))
    dlg._send_check.setChecked(dlg._cfg.osc.send_to_vrchat)
    dlg._send_check.setToolTip(tr("Show your captions in the VRChat chatbox."))
    dlg._bind_checkbox(dlg._send_check, dlg._cfg.osc, "send_to_vrchat")
    form.addRow(dlg._send_check)

    dlg._translate_check = QCheckBox(tr("Translate my speech"))
    dlg._translate_check.setChecked(dlg._cfg.translate.enabled)
    dlg._translate_check.setToolTip(tr("Also show a translation of what you say."))
    # Translate on/off applies live via a dedicated handler that pokes
    # on_model_change("mt"), not the restart-gated generic binding.
    dlg._translate_check.toggled.connect(dlg._on_translate_toggled)
    form.addRow(dlg._translate_check)

    dlg._include_original_check = QCheckBox(tr("Show my original words in the chatbox"))
    dlg._include_original_check.setChecked(dlg._cfg.osc.include_original)
    dlg._include_original_check.setToolTip(
        tr("Turn off to send only the translations. If translation is off, "
           "your words are always sent.")
    )
    dlg._bind_checkbox(dlg._include_original_check, dlg._cfg.osc, "include_original")
    form.addRow(dlg._include_original_check)

    # Interface language. Unlike the other fields it can't retint live widgets
    # (tr() runs at construction), so SettingsDialog rebuilds the main window on
    # close when it changed. Data is the code; labels are each language's own
    # name, so a user stuck in the wrong language can still find theirs.
    ui_lang = no_wheel(QComboBox())
    ui_lang.addItem(tr("Auto (match my system)"), "auto")
    for code, native_name in UI_LANGUAGES.items():
        ui_lang.addItem(native_name, code)
    li = ui_lang.findData(dlg._cfg.gui.ui_language)
    if li >= 0:
        ui_lang.setCurrentIndex(li)
    ui_lang.setToolTip(tr("The language of VRCC's interface."))
    dlg._bind_data_combo(ui_lang, dlg._cfg.gui, "ui_language")
    dlg._ui_language_combo = ui_lang
    form.addRow(tr("Language"), ui_lang)

    dlg._ui_language_hint = QLabel(tr(_UI_LANGUAGE_HINT))
    dlg._ui_language_hint.setWordWrap(True)
    dlg._ui_language_hint.setStyleSheet(dlg._muted_style)
    form.addRow("", dlg._ui_language_hint)

    scale_map = dict(_FONT_SCALE_PRESETS)
    cur = min(scale_map, key=lambda k: abs(scale_map[k] - dlg._cfg.gui.font_scale))
    dlg._text_size = SegmentedControl([(label, tr(label)) for label in scale_map], cur)
    dlg._text_size.setToolTip(tr("Make all text larger or smaller."))

    def on_text_size(label):
        if not dlg._loading:
            dlg._cfg.gui.font_scale = scale_map[label]
            dlg._changed()
    dlg._text_size.changed.connect(on_text_size)
    form.addRow(tr("Text size"), dlg._text_size)

    # The recommended reset lives with the everyday controls. Switching Mode
    # above applies the same presets, so the per-profile reset buttons are
    # gone; re-applying the CURRENT profile over hand-tuned Advanced knobs
    # now goes through this reset (a deliberate trade for a simpler page).
    reset = QPushButton(settings_reset.reset_button_text())
    reset.setToolTip(settings_reset.reset_button_tooltip())
    reset.clicked.connect(lambda: settings_reset.confirm_and_reset(dlg))
    form.addRow("", reset)

    # A separate reset for the tuning knobs (VAD/denoise/STT/MT quality gates):
    # it never touches the personal choices the recommended reset also spares.
    reset_defaults = QPushButton(settings_reset.reset_defaults_button_text())
    reset_defaults.setToolTip(settings_reset.reset_defaults_button_tooltip())
    reset_defaults.clicked.connect(lambda: settings_reset.confirm_and_reset_defaults(dlg))
    form.addRow("", reset_defaults)

    return page


def build_voice_page(dlg: "SettingsDialog") -> QWidget:
    page = QWidget()
    form = QFormLayout(page)
    form.setContentsMargins(24, 16, 24, 16)

    dlg._model_combo = no_wheel(QComboBox())
    # Downloaded voice models only (or all, headless). Rebuild the language-
    # limited index list against this FILTERED order so greying lines up with
    # combo rows.
    dlg._limited_model_indices = []
    voice_specs = dlg._downloaded_whisper_specs()
    _add_deleted_placeholder_if_needed(dlg._model_combo, voice_specs, dlg._cfg.stt.model)
    for spec in voice_specs:
        i = dlg._model_combo.count()
        dlg._model_combo.addItem(whisper_display_name(spec.id), spec.id)
        # Models that cannot report the language they heard join even without a
        # language restriction: the auto-plus-translation greying keys on that,
        # not on the set.
        if spec.languages is not None or not spec.reports_language:
            dlg._limited_model_indices.append((i, spec))
    mi = dlg._model_combo.findData(dlg._cfg.stt.model)
    if mi >= 0:
        dlg._model_combo.setCurrentIndex(mi)  # else: index 0 is already the placeholder
    dlg._voice_selected_id = dlg._model_combo.currentData()
    dlg._model_combo.setToolTip(
        tr("Bigger models are more accurate but slower and larger.")
    )
    dlg._model_combo.currentIndexChanged.connect(dlg._on_voice_model_changed)
    # The STT Auto device label depends on the model (onnx-asr auto -> cpu).
    dlg._model_combo.currentIndexChanged.connect(
        lambda _i: settings_reset.update_device_auto_labels(dlg)
    )

    form.addRow(tr("Voice model"), dlg._model_combo)
    if not voice_specs:
        dlg._model_combo.setEnabled(False)
        hint = QLabel(
            tr("No voice models downloaded yet. Get one in the Models window.")
        )
        hint.setStyleSheet(dlg._muted_style)
        hint.setWordWrap(True)
        form.addRow("", hint)

    dlg._source_combo = no_wheel(QComboBox())
    fill_spoken_languages(
        dlg._source_combo, tr("Auto (detect)"), _AUTO, LANGUAGES.keys()
    )
    set_combo_value(dlg._source_combo, dlg._cfg.stt.source_language)
    dlg._source_combo.setToolTip(
        tr("The language you speak. Auto tries to detect it.")
    )

    def on_source(_i):
        dlg._update_language_limited_items()
        if dlg._loading:
            return
        dlg._cfg.stt.source_language = combo_value(dlg._source_combo)
        dlg._changed()
    dlg._source_combo.currentIndexChanged.connect(on_source)

    form.addRow(tr("Spoken language"), dlg._source_combo)

    # Energy gate.
    gate = QCheckBox(tr("Ignore quiet background noise"))
    gate.setChecked(dlg._cfg.audio.energy_gate_enabled)
    gate.setToolTip(
        tr("Skip very quiet sounds so background noise doesn't trigger captions.")
    )
    dlg._bind_checkbox(gate, dlg._cfg.audio, "energy_gate_enabled")
    dlg._gate_check = gate
    form.addRow(gate)

    def on_gate_toggled(checked: bool) -> None:
        # The gate is off by default and energy_gate.gated_level() returns
        # before reading the threshold, so a live slider here gives full
        # feedback for a control that does nothing. The denoise pair two rows
        # below already greys together; this matches it.
        for w in (slider, dlg._noise_value_label, dlg._noise_low, dlg._noise_high):
            w.setEnabled(bool(checked))

    gate.toggled.connect(on_gate_toggled)

    slider = no_wheel(QSlider(Qt.Orientation.Horizontal))
    slider.setRange(0, 2000)
    slider.setValue(dlg._cfg.audio.energy_threshold)
    slider.setToolTip(
        tr("How loud a sound must be to count. Higher ignores more background noise.")
    )
    dlg._noise_value_label = QLabel(str(dlg._cfg.audio.energy_threshold))
    dlg._noise_value_label.setStyleSheet(dlg._muted_style)

    def on_gate(v):
        dlg._noise_value_label.setText(str(v))
        if dlg._loading:
            return
        dlg._cfg.audio.energy_threshold = int(v)
        dlg._changed()
    slider.valueChanged.connect(on_gate)
    gate_row, dlg._noise_low, dlg._noise_high = dlg._anchored_slider(slider, dlg._noise_value_label)
    dlg._noise_slider = slider
    form.addRow(tr("Background noise level"), gate_row)
    on_gate_toggled(gate.isChecked())

    settings_audio.build_denoise_controls(dlg, form)

    # Advanced fine-tuning (quality gates + prompt).
    adv = QGroupBox(tr("Advanced (fine-tuning)"))
    adv_form = QFormLayout(adv)

    beam = dlg._spin(1, 10, dlg._cfg.stt.beam_size)
    beam.setToolTip(
        tr("Higher considers more options: a little more accurate, a little slower.")
    )
    dlg._bind_int(beam, dlg._cfg.stt, "beam_size")
    dlg._stt_beam_spin = beam
    adv_form.addRow(tr("Search width"), beam)

    temp = dlg._dspin(0.0, 1.0, dlg._cfg.stt.temperature, 2, 0.1)
    temp.setToolTip(tr("Higher lets the model guess more freely when it's unsure."))
    dlg._bind_float(temp, dlg._cfg.stt, "temperature")
    dlg._stt_temp_spin = temp
    adv_form.addRow(tr("Guessing"), temp)

    avg_gate = dlg._dspin(-5.0, 0.0, dlg._cfg.stt.avg_logprob_gate, 2, 0.1)
    avg_gate.setToolTip(tr("Drop captions the model isn't confident about."))
    dlg._bind_float(avg_gate, dlg._cfg.stt, "avg_logprob_gate")
    dlg._stt_avg_gate_spin = avg_gate
    adv_form.addRow(tr("Confidence cutoff"), avg_gate)

    ns_gate = dlg._dspin(0.0, 1.0, dlg._cfg.stt.no_speech_gate, 2, 0.05)
    ns_gate.setToolTip(
        tr("How sure the model must be that you actually spoke before captioning.")
    )
    dlg._bind_float(ns_gate, dlg._cfg.stt, "no_speech_gate")
    dlg._stt_ns_gate_spin = ns_gate
    adv_form.addRow(tr("Silence sensitivity"), ns_gate)

    norepeat = dlg._spin(0, 6, dlg._cfg.stt.no_repeat_ngram_size)
    norepeat.setToolTip(
        tr("Stops the model looping on the same short phrase. 0 turns it off.")
    )
    dlg._bind_int(norepeat, dlg._cfg.stt, "no_repeat_ngram_size")
    dlg._stt_norepeat_spin = norepeat
    adv_form.addRow(tr("Block repeats"), norepeat)

    comp = dlg._dspin(1.5, 10.0, dlg._cfg.stt.compression_ratio_gate, 1, 0.5)
    comp.setToolTip(
        tr(
            "Drops a caption that is just the same short phrase repeated, "
            "which can happen when the audio is misheard. Higher allows more "
            "repetition before dropping."
        )
    )
    dlg._bind_float(comp, dlg._cfg.stt, "compression_ratio_gate")
    dlg._stt_compression_spin = comp
    adv_form.addRow(tr("Drop repeats"), comp)

    prompt = QLineEdit(dlg._cfg.stt.initial_prompt)
    prompt.setToolTip(
        tr("Optional words to help the model spell names or jargon correctly.")
    )
    dlg._bind_line(prompt, dlg._cfg.stt, "initial_prompt")
    adv_form.addRow(tr("Context hint"), prompt)

    cond = QCheckBox(tr("Use earlier speech as context"))
    cond.setChecked(dlg._cfg.stt.condition_on_previous_text)
    cond.setToolTip(
        tr("Feed previous captions back in for consistency (can drift after a mistake).")
    )
    dlg._bind_checkbox(cond, dlg._cfg.stt, "condition_on_previous_text")
    dlg._stt_cond_check = cond
    adv_form.addRow(cond)

    form.addRow(adv)

    dlg._update_language_limited_items()
    model_prompts.grey_unsupported_languages(
        dlg._source_combo, dlg._cfg.stt.model,
        translating=dlg._cfg.translate.enabled,
    )
    return page


def build_translation_page(dlg: "SettingsDialog") -> QWidget:
    page = QWidget()
    form = QFormLayout(page)
    form.setContentsMargins(24, 16, 24, 16)

    note = QLabel(tr("Turn translation on or off on the Simple tab."))
    note.setStyleSheet(dlg._muted_style)
    note.setWordWrap(True)
    form.addRow(note)

    model = no_wheel(QComboBox())
    # List only downloaded translation models (or all, headless).
    mt_specs = dlg._downloaded_mt_specs()
    _add_deleted_placeholder_if_needed(model, mt_specs, dlg._cfg.translate.model)
    for spec in mt_specs:
        model.addItem(mt_display_name(spec.id), spec.id)
    mi = model.findData(dlg._cfg.translate.model)
    if mi >= 0:
        model.setCurrentIndex(mi)  # else: index 0 is already the placeholder
    model.setToolTip(tr("The model that translates your speech."))
    dlg._translate_model_combo = model
    dlg._mt_selected_id = model.currentData()
    model.currentIndexChanged.connect(dlg._on_mt_model_changed)
    form.addRow(tr("Translation model"), model)
    if not mt_specs:
        model.setEnabled(False)
        hint = QLabel(
            tr("No translation models downloaded yet. Get one in the Models window.")
        )
        hint.setStyleSheet(dlg._muted_style)
        hint.setWordWrap(True)
        form.addRow("", hint)

    adv = QGroupBox(tr("Advanced (fine-tuning)"))
    adv_form = QFormLayout(adv)
    beam = dlg._spin(1, 10, dlg._cfg.translate.beam_size)
    beam.setToolTip(
        tr("Higher considers more options: a little more accurate, a little slower.")
    )
    dlg._bind_int(beam, dlg._cfg.translate, "beam_size")
    dlg._mt_beam_spin = beam
    adv_form.addRow(tr("Search width"), beam)

    rep = dlg._dspin(1.0, 2.0, dlg._cfg.translate.repetition_penalty, 2, 0.05)
    rep.setToolTip(
        tr("Discourages the model from repeating itself when it gets stuck.")
    )
    dlg._bind_float(rep, dlg._cfg.translate, "repetition_penalty")
    dlg._mt_rep_spin = rep
    adv_form.addRow(tr("Repetition guard"), rep)

    norepeat = dlg._spin(0, 6, dlg._cfg.translate.no_repeat_ngram_size)
    norepeat.setToolTip(
        tr("Stops the model looping on the same short phrase. 0 turns it off.")
    )
    dlg._bind_int(norepeat, dlg._cfg.translate, "no_repeat_ngram_size")
    dlg._mt_norepeat_spin = norepeat
    adv_form.addRow(tr("Block repeats"), norepeat)
    form.addRow(adv)

    dlg._translate_page_widgets = [adv]
    # A combo left disabled because nothing is downloaded must stay disabled,
    # so it only joins the follow-the-toggle set when it has something to offer.
    if mt_specs:
        dlg._translate_page_widgets.append(model)
        model_label = form.labelForField(model)
        if model_label is not None:
            dlg._translate_page_widgets.append(model_label)
    set_translation_page_enabled(dlg, dlg._cfg.translate.enabled)
    return page


def set_translation_page_enabled(dlg: "SettingsDialog", enabled: bool) -> None:
    """Grey the Translation page with the feature, the way the denoise slider
    follows its checkbox: every control here writes a field nothing reads while
    ``translate.enabled`` is off. The note at the top stays live, since it says
    where to turn translation back on."""
    for widget in getattr(dlg, "_translate_page_widgets", ()):
        widget.setEnabled(bool(enabled))
