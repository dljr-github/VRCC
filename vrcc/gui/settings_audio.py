"""Noise-reduction + device-picker/refresh controls for the Settings Voice
and Simple pages.

Kept out of settings_pages.py to hold that file under the source cap, mirroring
its build_*_page(dlg) pattern (imports from settings are type-only).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QWidget,
)

from vrcc.audio.devices import list_input_devices
from vrcc.gui.widgets import no_wheel
from vrcc.i18n import tr

if TYPE_CHECKING:
    from PySide6.QtWidgets import QFormLayout

    from vrcc.gui.settings import SettingsDialog

logger = logging.getLogger("vrcc.gui.settings_audio")

_AUTO = "auto"


def denoise_strength_enabled(dlg: "SettingsDialog", enabled: bool) -> None:
    """Grey the strength row with its checkbox, anchors and readout included:
    a live slider beside a feature that is off gives feedback for a control
    that does nothing."""
    for widget in (
        dlg._denoise_strength,
        dlg._denoise_value_label,
        dlg._denoise_low,
        dlg._denoise_high,
    ):
        widget.setEnabled(bool(enabled))


def build_denoise_controls(dlg: "SettingsDialog", form: "QFormLayout") -> None:
    """Add a "Reduce background noise" checkbox and its strength slider."""
    dlg._denoise_check = QCheckBox(tr("Reduce background noise"))
    dlg._denoise_check.setChecked(dlg._cfg.audio.denoise_enabled)
    dlg._denoise_check.setToolTip(
        tr(
            "Clean up steady background noise before transcription. Gentle "
            "by design. Uses a little more of your graphics card."
        )
    )

    dlg._denoise_strength = no_wheel(QSlider(Qt.Orientation.Horizontal))
    dlg._denoise_strength.setRange(0, 100)
    dlg._denoise_strength.setValue(int(round(dlg._cfg.audio.denoise_strength * 100)))
    # Low is not off: the model still runs and still shapes the audio. The
    # checkbox above is the only way to leave the microphone untouched.
    dlg._denoise_strength.setToolTip(
        tr(
            "How much of the cleaned-up audio is used. Low still cleans a "
            "little; untick the box above to leave your microphone alone."
        )
    )
    dlg._denoise_value_label = QLabel(_percent(dlg._denoise_strength.value()))
    dlg._denoise_value_label.setStyleSheet(dlg._muted_style)

    def on_strength(v):
        dlg._denoise_value_label.setText(_percent(v))
        if dlg._loading:
            return
        dlg._cfg.audio.denoise_strength = v / 100.0
        dlg._changed()

    def on_toggle(checked):
        denoise_strength_enabled(dlg, checked)
        if dlg._loading:
            return
        dlg._cfg.audio.denoise_enabled = bool(checked)
        dlg._changed()

    dlg._denoise_strength.valueChanged.connect(on_strength)
    dlg._denoise_check.toggled.connect(on_toggle)

    row, dlg._denoise_low, dlg._denoise_high = dlg._anchored_slider(
        dlg._denoise_strength, dlg._denoise_value_label
    )
    denoise_strength_enabled(dlg, dlg._cfg.audio.denoise_enabled)

    form.addRow(dlg._denoise_check)
    form.addRow(tr("Noise reduction strength"), row)


def _percent(value: int) -> str:
    return f"{value}%"


def _fill_input_devices(dlg: "SettingsDialog", combo: QComboBox) -> None:
    combo.clear()
    combo.addItem(tr("Auto (system default)"), _AUTO)
    try:
        for _index, name in list_input_devices():
            combo.addItem(name, name)
    except Exception:  # noqa: BLE001
        logger.debug("could not list input devices", exc_info=True)
    cur = dlg._cfg.audio.device
    idx = combo.findData(cur)
    if idx < 0:
        combo.addItem(cur, cur)
        idx = combo.findData(cur)
    combo.setCurrentIndex(idx)


def _repopulate_input_devices(dlg: "SettingsDialog") -> None:
    """Refresh the device list in place under the loading guard (no swap)."""
    combo = getattr(dlg, "_input_device_combo", None)
    if combo is None:
        return
    was_loading = dlg._loading
    dlg._loading = True
    try:
        _fill_input_devices(dlg, combo)
    finally:
        dlg._loading = was_loading


def make_input_device_row(dlg: "SettingsDialog") -> QWidget:
    """The microphone picker plus a Refresh button on the Simple page."""
    combo = no_wheel(QComboBox())
    dlg._input_device_combo = combo
    _fill_input_devices(dlg, combo)
    combo.setToolTip(tr("Which microphone to listen to."))

    def on_device(_i):
        if dlg._loading:
            return
        dlg._cfg.audio.device = combo.currentData()
        dlg._changed()
    combo.currentIndexChanged.connect(on_device)

    refresh = QPushButton(tr("Refresh"))
    refresh.setToolTip(tr("Look for microphones you plugged in after opening VRCC."))

    def on_refresh():
        if dlg._apply is not None:
            dlg._apply.refresh_input_devices(dlg._cfg.audio.device)
        _repopulate_input_devices(dlg)
    refresh.clicked.connect(on_refresh)

    row = QHBoxLayout()
    row.addWidget(combo, stretch=1)
    row.addWidget(refresh)
    holder = QWidget()
    holder.setLayout(row)
    return holder
