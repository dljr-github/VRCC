"""Offscreen GUI tests for the Settings dialog's audio/microphone controls."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication

from vrcc.core.config import ConfigStore, default_paths
from vrcc.gui.settings import SettingsDialog


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _store(tmp_path):
    return ConfigStore(default_paths(portable=True, app_dir=tmp_path).config_file)


def test_sensitivity_slider_is_inverted(qapp, tmp_path):
    # Default 0.50 -> slider 40; moving the slider UP lowers the threshold.
    store = _store(tmp_path)
    store.config.vad.threshold = 0.50
    dlg = SettingsDialog(store)
    try:
        assert dlg._sensitivity.value() == 40
        dlg._sensitivity.setValue(60)
        assert store.config.vad.threshold == 0.30
        dlg._sensitivity.setValue(30)
        assert store.config.vad.threshold == 0.60
    finally:
        dlg.close()
        dlg.deleteLater()
