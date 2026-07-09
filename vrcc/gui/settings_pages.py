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
    QSlider,
    QWidget,
)
from PySide6.QtCore import Qt

from vrcc.core.languages import LANGUAGES
from vrcc.gui import model_prompts
from vrcc.gui.model_labels import mt_display_name, whisper_display_name
from vrcc.gui.widgets import SegmentedControl
from vrcc.i18n import UI_LANGUAGES, tr, tr_noop
from vrcc.stt.registry import WHISPER_MODELS

if TYPE_CHECKING:
    from vrcc.gui.settings import SettingsDialog

_AUTO = "auto"

# Plain-language Speed/Quality explanation (Mode tooltip + visible description).
_MODE_TOOLTIP = tr_noop(
    "Speed shows captions almost instantly. Quality is more accurate and "
    "clips fewer words off the ends of sentences, but each caption takes a "
    "little longer."
)
_MODE_DESC = tr_noop(
    "Speed shows captions almost instantly; Quality is more accurate and "
    "clips fewer words, but each caption takes a little longer."
)
# Replaces _MODE_TOOLTIP while the active voice model decodes greedily (the
# onnx_asr backend ignores beam size and temperature, the profile's headline
# caption-quality effect).
_MODE_LOCKED_TOOLTIP = tr_noop(
    "Parakeet and Canary always decode at full accuracy, so Speed / Quality "
    "does not change their captions."
)

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

    form.addRow(tr("Microphone"), dlg._make_input_device_combo())

    dlg._sensitivity = QSlider(Qt.Orientation.Horizontal)
    dlg._sensitivity.setRange(30, 60)
    dlg._sensitivity.setValue(int(round(dlg._cfg.vad.threshold * 100)))
    dlg._sensitivity.setToolTip(
        tr("How loud you need to speak before captioning starts.")
    )

    def on_sensitivity(v):
        if dlg._loading:
            return
        dlg._cfg.vad.threshold = v / 100.0
        dlg._changed()
    dlg._sensitivity.valueChanged.connect(on_sensitivity)
    sens_row, dlg._sensitivity_low, dlg._sensitivity_high = dlg._anchored_slider(dlg._sensitivity)
    form.addRow(tr("Microphone sensitivity"), sens_row)

    # Mode: Speed <-> Quality (maps to apply_profile), with tooltip + description.
    dlg._mode = SegmentedControl(
        [("Speed", tr("Speed")), ("Quality", tr("Quality"))],
        "Quality" if dlg._cfg.gui.profile == "quality" else "Speed",
    )
    dlg._mode.setToolTip(tr(_MODE_TOOLTIP))
    dlg._mode.changed.connect(dlg._on_mode_changed)
    form.addRow(tr("Mode"), dlg._mode)

    dlg._mode_desc = QLabel(tr(_MODE_DESC))
    dlg._mode_desc.setWordWrap(True)
    dlg._mode_desc.setStyleSheet(dlg._muted_style)
    form.addRow("", dlg._mode_desc)

    def update_mode_for_model():
        # onnx_asr models decode greedily, so the profile's beam/temperature
        # presets can't tune their captions: grey the control in place. The
        # stored profile is untouched (its VAD/translation parts still apply,
        # and the Advanced knobs stay usable). The visible description must
        # not advertise a trade-off the locked control can't deliver, so it
        # swaps to the locked explanation and back.
        spec = WHISPER_MODELS.get(dlg._cfg.stt.model)
        locked = spec is not None and spec.backend == "onnx_asr"
        dlg._mode.setEnabled(not locked)
        dlg._mode.setToolTip(tr(_MODE_LOCKED_TOOLTIP if locked else _MODE_TOOLTIP))
        dlg._mode_desc.setText(tr(_MODE_LOCKED_TOOLTIP if locked else _MODE_DESC))
    dlg._update_mode_for_model = update_mode_for_model
    update_mode_for_model()

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

    # Appearance.
    theme = QComboBox()
    for label, value in (
        (tr("System"), "system"), (tr("Dark"), "dark"), (tr("Light"), "light")
    ):
        theme.addItem(label, value)
    ti = theme.findData(dlg._cfg.gui.theme)
    if ti >= 0:
        theme.setCurrentIndex(ti)
    theme.setToolTip(tr("Dark, light, or match your system."))
    dlg._bind_data_combo(theme, dlg._cfg.gui, "theme")
    form.addRow(tr("Theme"), theme)

    # Interface language (restart-applied, like the theme). Data is the code;
    # labels are each language's own name, so a user stuck in the wrong
    # language can still find theirs.
    ui_lang = QComboBox()
    ui_lang.addItem(tr("Auto (match my system)"), "auto")
    for code, native_name in UI_LANGUAGES.items():
        ui_lang.addItem(native_name, code)
    li = ui_lang.findData(dlg._cfg.gui.ui_language)
    if li >= 0:
        ui_lang.setCurrentIndex(li)
    ui_lang.setToolTip(
        tr("The language of VRCC's interface. Applies after restarting VRCC.")
    )
    dlg._bind_data_combo(ui_lang, dlg._cfg.gui, "ui_language")
    dlg._ui_language_combo = ui_lang
    form.addRow(tr("Language"), ui_lang)

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

    return page


def build_voice_page(dlg: "SettingsDialog") -> QWidget:
    page = QWidget()
    form = QFormLayout(page)
    form.setContentsMargins(24, 16, 24, 16)

    dlg._model_combo = QComboBox()
    # Downloaded voice models only (or all, headless). Rebuild the language-
    # limited index list against this FILTERED order so greying lines up with
    # combo rows.
    dlg._limited_model_indices = []
    voice_specs = dlg._downloaded_whisper_specs()
    _add_deleted_placeholder_if_needed(dlg._model_combo, voice_specs, dlg._cfg.stt.model)
    for spec in voice_specs:
        i = dlg._model_combo.count()
        dlg._model_combo.addItem(whisper_display_name(spec.id), spec.id)
        if spec.languages is not None:
            dlg._limited_model_indices.append((i, spec))
    mi = dlg._model_combo.findData(dlg._cfg.stt.model)
    if mi >= 0:
        dlg._model_combo.setCurrentIndex(mi)  # else: index 0 is already the placeholder
    dlg._voice_selected_id = dlg._model_combo.currentData()
    dlg._model_combo.setToolTip(
        tr("Bigger models are more accurate but slower and larger.")
    )
    dlg._model_combo.currentIndexChanged.connect(dlg._on_voice_model_changed)

    form.addRow(tr("Voice model"), dlg._model_combo)
    if not voice_specs:
        dlg._model_combo.setEnabled(False)
        hint = QLabel(
            tr("No voice models downloaded yet. Get one in the Models window.")
        )
        hint.setStyleSheet(dlg._muted_style)
        hint.setWordWrap(True)
        form.addRow("", hint)

    dlg._source_combo = QComboBox()
    dlg._source_combo.addItem(_AUTO)
    dlg._source_combo.addItems(list(LANGUAGES.keys()))
    dlg._set_combo_text(dlg._source_combo, dlg._cfg.stt.source_language)
    dlg._source_combo.setToolTip(
        tr("The language you speak. Auto tries to detect it.")
    )

    def on_source(_i):
        dlg._update_language_limited_items()
        if dlg._loading:
            return
        dlg._cfg.stt.source_language = dlg._source_combo.currentText()
        dlg._changed()
        model_prompts.maybe_switch_model_for_language(dlg)
    dlg._source_combo.currentIndexChanged.connect(on_source)

    form.addRow(tr("Spoken language"), dlg._source_combo)

    # Energy gate.
    gate = QCheckBox(tr("Ignore quiet background noise"))
    gate.setChecked(dlg._cfg.audio.energy_gate_enabled)
    gate.setToolTip(
        tr("Skip very quiet sounds so background noise doesn't trigger captions.")
    )
    dlg._bind_checkbox(gate, dlg._cfg.audio, "energy_gate_enabled")
    form.addRow(gate)

    slider = QSlider(Qt.Orientation.Horizontal)
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
    form.addRow(tr("Background noise level"), gate_row)

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
    adv_form.addRow(tr("Confidence cutoff"), avg_gate)

    ns_gate = dlg._dspin(0.0, 1.0, dlg._cfg.stt.no_speech_gate, 2, 0.05)
    ns_gate.setToolTip(
        tr("How sure the model must be that you actually spoke before captioning.")
    )
    dlg._bind_float(ns_gate, dlg._cfg.stt, "no_speech_gate")
    adv_form.addRow(tr("Silence sensitivity"), ns_gate)

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
    adv_form.addRow(cond)

    form.addRow(adv)

    dlg._update_language_limited_items()
    return page


def build_translation_page(dlg: "SettingsDialog") -> QWidget:
    page = QWidget()
    form = QFormLayout(page)
    form.setContentsMargins(24, 16, 24, 16)

    note = QLabel(tr("Turn translation on or off on the Simple tab."))
    note.setStyleSheet(dlg._muted_style)
    note.setWordWrap(True)
    form.addRow(note)

    model = QComboBox()
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
    form.addRow(adv)

    return page
