"""Settings controls for the microphone's automatic language preference."""

from PySide6.QtWidgets import QComboBox, QLabel

from vrcc.gui.firstrun_languages import build_picker, checked_in
from vrcc.gui.widgets import no_wheel, set_combo_value
from vrcc.i18n import tr
from vrcc.stt.registry import WHISPER_MODELS


def build_controls(dlg, form) -> None:
    cfg = dlg._cfg.stt
    combo = dlg._language_bias_combo = no_wheel(QComboBox())
    combo.addItem(tr("Wizard languages"), "wizard")
    combo.addItem(tr("Custom languages"), "custom")
    combo.addItem(tr("Off"), "off")
    set_combo_value(combo, cfg.language_bias_mode)
    form.addRow(tr("Auto-detection bias"), combo)

    def on_languages():
        if not dlg._loading:
            cfg.language_bias_languages = checked_in(picker)
            dlg._changed()

    picker = dlg._language_bias_picker = build_picker(
        dlg._cfg.gui.font_scale, dlg._cfg, on_languages,
        selected_languages=cfg.language_bias_languages,
    )
    form.addRow("", picker)
    hint = QLabel(tr(
        "Used only with Auto (detect) on Whisper and SenseVoice. "
        "An empty custom list applies no bias."
    ))
    hint.setWordWrap(True)
    hint.setStyleSheet(dlg._muted_style)
    form.addRow("", hint)

    def refresh(*_args):
        spec = WHISPER_MODELS.get(cfg.model)
        supported = spec is None or (spec.backend != "onnx_asr" and spec.auto_language)
        enabled = cfg.source_language == "auto" and supported
        combo.setEnabled(enabled)
        picker.setEnabled(enabled)
        picker.setVisible(cfg.language_bias_mode == "custom")

    def on_mode(_index):
        if not dlg._loading:
            cfg.language_bias_mode = combo.currentData()
            dlg._changed()
        refresh()

    combo.currentIndexChanged.connect(on_mode)
    dlg._source_combo.currentIndexChanged.connect(refresh)
    dlg._model_combo.currentIndexChanged.connect(refresh)
    refresh()
