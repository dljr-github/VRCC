"""The pinyin checkbox is offered only with a Chinese target, and greys with
translation itself."""

from __future__ import annotations

import pytest
from PySide6.QtWidgets import QApplication

from vrcc.core.config import ConfigStore
from vrcc.gui.settings import SettingsDialog


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _dialog(tmp_path, targets, enabled=True):
    store = ConfigStore(tmp_path / "config.json")
    store.config.stt.device = "cpu"
    store.config.translate.targets = targets
    store.config.translate.enabled = enabled
    return SettingsDialog(store)


def _row_visible(dlg) -> bool:
    form = dlg._pinyin_check.parentWidget().layout()
    return form.isRowVisible(dlg._pinyin_check)


def test_hidden_without_a_chinese_target(qapp, tmp_path):
    dlg = _dialog(tmp_path, ["Japanese", "English"])
    assert not _row_visible(dlg)
    dlg.deleteLater()


@pytest.mark.parametrize("target", ["Chinese Simplified", "Chinese Traditional"])
def test_shown_with_a_chinese_target(qapp, tmp_path, target):
    dlg = _dialog(tmp_path, ["Japanese", target])
    assert _row_visible(dlg)
    dlg.deleteLater()


def test_greys_with_the_translate_toggle(qapp, tmp_path):
    dlg = _dialog(tmp_path, ["Chinese Simplified"], enabled=False)
    assert not dlg._pinyin_check.isEnabled()
    dlg._translate_check.setChecked(True)
    assert dlg._pinyin_check.isEnabled()
    dlg._translate_check.setChecked(False)
    assert not dlg._pinyin_check.isEnabled()
    dlg.deleteLater()
