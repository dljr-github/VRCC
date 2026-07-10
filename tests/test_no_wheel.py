"""The scroll wheel must never edit a value in passing. Every spin box and
combo box ignores wheel events, so scrolling a settings page cannot change a
field the pointer happens to cross.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtGui import QWheelEvent
from PySide6.QtWidgets import QAbstractSpinBox, QApplication, QComboBox, QSpinBox

from vrcc.core.config import ConfigStore, default_paths
from vrcc.gui.settings import SettingsDialog
from vrcc.gui.widgets import no_wheel


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _scroll(widget, steps: int) -> None:
    ev = QWheelEvent(
        QPointF(5, 5),
        QPointF(5, 5),
        QPoint(0, 0),
        QPoint(0, 120 * steps),
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
        Qt.ScrollPhase.NoScrollPhase,
        False,
    )
    QApplication.sendEvent(widget, ev)


def test_wheel_probe_steps_an_unguarded_spin(qapp):
    # Sanity check on the probe itself: a plain QSpinBox must step on wheel,
    # otherwise the assertions below pass without testing anything.
    spin = QSpinBox()
    spin.setRange(0, 10)
    spin.setValue(5)
    _scroll(spin, 1)
    assert spin.value() == 6


def test_no_wheel_leaves_spin_and_combo_untouched(qapp):
    spin = no_wheel(QSpinBox())
    spin.setRange(0, 10)
    spin.setValue(5)
    _scroll(spin, 1)
    assert spin.value() == 5
    _scroll(spin, -1)
    assert spin.value() == 5

    combo = no_wheel(QComboBox())
    combo.addItems(["a", "b", "c"])
    combo.setCurrentIndex(1)
    _scroll(combo, 1)
    assert combo.currentIndex() == 1
    _scroll(combo, -1)
    assert combo.currentIndex() == 1


def test_every_settings_input_ignores_wheel(qapp, tmp_path):
    store = ConfigStore(default_paths(portable=True, app_dir=tmp_path).config_file)
    dlg = SettingsDialog(store)
    try:
        spins = dlg.findChildren(QAbstractSpinBox)
        combos = dlg.findChildren(QComboBox)
        assert spins and combos
        for w in spins:
            before = w.text()
            for steps in (1, -1):
                _scroll(w, steps)
                assert w.text() == before, w.toolTip()
        for c in combos:
            before = c.currentIndex()
            for steps in (1, -1):
                _scroll(c, steps)
                assert c.currentIndex() == before, c.toolTip()
    finally:
        dlg.close()
        dlg.deleteLater()
