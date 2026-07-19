"""Microphone gain + device-refresh controls for the Settings Voice page.

Kept out of settings_pages.py to hold that file under the source cap, mirroring
its build_*_page(dlg) pattern (imports from settings are type-only).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QCheckBox, QLabel, QSlider

from vrcc.i18n import tr

if TYPE_CHECKING:
    from PySide6.QtWidgets import QFormLayout

    from vrcc.gui.settings import SettingsDialog


def build_gain_controls(dlg: "SettingsDialog", form: "QFormLayout") -> None:
    """Add a mic-boost slider (dB) and an automatic-level checkbox."""
    dlg._auto_gain_check = QCheckBox(tr("Set my microphone level automatically"))
    dlg._auto_gain_check.setChecked(dlg._cfg.audio.auto_gain)
    dlg._auto_gain_check.setToolTip(
        tr("Keep your voice at a steady loudness without setting a level by hand.")
    )

    dlg._gain_slider = QSlider(Qt.Orientation.Horizontal)
    dlg._gain_slider.setRange(0, 30)  # 0..30 dB boost
    dlg._gain_slider.setValue(int(round(dlg._cfg.audio.gain_db)))
    dlg._gain_slider.setToolTip(
        tr("Boost a quiet microphone. Raise this if your captions miss soft speech.")
    )
    dlg._gain_value_label = QLabel(f"{int(round(dlg._cfg.audio.gain_db))} dB")
    dlg._gain_value_label.setStyleSheet(dlg._muted_style)

    def on_gain(v):
        dlg._gain_value_label.setText(f"{int(v)} dB")
        if dlg._loading:
            return
        dlg._cfg.audio.gain_db = float(v)
        dlg._changed()

    def on_auto(checked):
        dlg._gain_slider.setEnabled(not checked)
        if dlg._loading:
            return
        dlg._cfg.audio.auto_gain = bool(checked)
        dlg._changed()

    dlg._gain_slider.valueChanged.connect(on_gain)
    dlg._auto_gain_check.toggled.connect(on_auto)
    dlg._gain_slider.setEnabled(not dlg._cfg.audio.auto_gain)

    gain_row, dlg._gain_low, dlg._gain_high = dlg._anchored_slider(
        dlg._gain_slider, dlg._gain_value_label
    )
    form.addRow(dlg._auto_gain_check)
    form.addRow(tr("Microphone boost"), gain_row)
