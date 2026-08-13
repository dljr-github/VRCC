"""The Simple settings page.

Split from ``settings_pages`` for the file-length cap. Rows are grouped under
headings, and a control that depends on a tick sits inside that tick's section.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QLabel,
    QPushButton,
    QSlider,
    QWidget,
)

from vrcc.gui import settings_audio, settings_heard, settings_mode, settings_reset
from vrcc.gui.style import PALETTE, resolve_theme
from vrcc.gui.widgets import SegmentedControl, no_wheel
from vrcc.i18n import UI_LANGUAGES, tr, tr_noop

if TYPE_CHECKING:
    from vrcc.gui.settings import SettingsDialog

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

_MICROPHONE = tr_noop("Microphone")
_RECOGNITION = tr_noop("Speed and accuracy")
_MY_CAPTIONS = tr_noop("What I say")
_OTHERS = tr_noop("What other people say")
_APP = tr_noop("VRCC itself")


def _section_style(dlg: "SettingsDialog") -> str:
    """Section titles: muted, tracked, and smaller than the control labels.

    Deliberately quieter than the labels under them. A heading that competes
    with its rows adds a second thing to read per row instead of letting the
    eye skip to the group it wants. Theme-resolved and scaled the same way
    SettingsDialog builds its own hint styles, so headings follow the text-size
    preset like everything else on the page.
    """
    p = PALETTE[resolve_theme(dlg._cfg.gui.theme)]
    scale = max(0.5, min(2.0, dlg._cfg.gui.font_scale))
    return (
        f"color: {p['muted']}; font-size: {round(10 * scale)}px; "
        "letter-spacing: 1px; background: transparent;"
    )


def _section(dlg: "SettingsDialog", form: QFormLayout, title: str) -> QFormLayout:
    """Start a titled group in the page's one form, and hand that form back.

    One form for the whole page, not one per section: QFormLayout sizes its
    label column to its own widest label, so a form per group left every
    group's controls starting at a different x. A single column is what makes
    the page scan as one list of settings rather than five stacked tables.

    The heading spans both columns, which is what addRow with a single widget
    does, so it sits flush left of the labels rather than inside the control
    column.
    """
    style = _section_style(dlg)
    if form.rowCount():
        # Space above a heading, not below it, so a title reads as attached to
        # the rows it introduces.
        style += " margin-top: 16px;"
    heading = QLabel(tr(title))
    heading.setStyleSheet(style)
    form.addRow(heading)
    return form


def build_simple_page(dlg: "SettingsDialog") -> QWidget:
    page = QWidget()
    form = QFormLayout(page)
    form.setContentsMargins(24, 16, 24, 16)

    _build_microphone(dlg, _section(dlg, form, _MICROPHONE))
    settings_mode.build_mode_control(dlg, _section(dlg, form, _RECOGNITION))
    _build_my_captions(dlg, _section(dlg, form, _MY_CAPTIONS))
    settings_heard.build_heard_controls(dlg, _section(dlg, form, _OTHERS))
    _build_app(dlg, _section(dlg, form, _APP))

    return page


def _build_microphone(dlg: "SettingsDialog", form: QFormLayout) -> None:
    form.addRow(tr("Device"), settings_audio.make_input_device_row(dlg))

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
    row, dlg._sensitivity_low, dlg._sensitivity_high = dlg._anchored_slider(
        dlg._sensitivity
    )
    form.addRow(tr("Sensitivity"), row)


def _build_my_captions(dlg: "SettingsDialog", form: QFormLayout) -> None:
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


def _build_app(dlg: "SettingsDialog", form: QFormLayout) -> None:
    # Data is the language code; labels are each language's own name, so a user
    # stuck in the wrong language can still find theirs.
    ui_lang = no_wheel(QComboBox())
    ui_lang.addItem(tr("Auto (match my system)"), "auto")
    for code, native_name in UI_LANGUAGES.items():
        ui_lang.addItem(native_name, code)
    index = ui_lang.findData(dlg._cfg.gui.ui_language)
    if index >= 0:
        ui_lang.setCurrentIndex(index)
    ui_lang.setToolTip(tr("The language of VRCC's interface."))
    dlg._bind_data_combo(ui_lang, dlg._cfg.gui, "ui_language")
    dlg._ui_language_combo = ui_lang
    # Not "Language". This page has a second language picker three rows up that
    # chooses what other people's speech is shown in, and the bare word does
    # not distinguish them.
    form.addRow(tr("VRCC's language"), ui_lang)

    dlg._ui_language_hint = QLabel(tr(_UI_LANGUAGE_HINT))
    dlg._ui_language_hint.setWordWrap(True)
    dlg._ui_language_hint.setStyleSheet(dlg._muted_style)
    form.addRow("", dlg._ui_language_hint)

    scale_map = dict(_FONT_SCALE_PRESETS)
    current = min(scale_map, key=lambda k: abs(scale_map[k] - dlg._cfg.gui.font_scale))
    dlg._text_size = SegmentedControl(
        [(label, tr(label)) for label in scale_map], current
    )
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
