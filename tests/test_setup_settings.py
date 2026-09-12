"""The Simple page's "Bring back the setup steps" button: it must clear and
persist ``gui.setup_check_done`` so the running setup check controller (which
this dialog never touches directly) notices the change on its own next poll.

Split from tests/test_settings_ui.py, which is at the repo's 500-line cap.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication

from vrcc.core.config import ConfigStore, default_paths
from vrcc.gui.settings import SettingsDialog


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _dlg(tmp_path):
    store = ConfigStore(default_paths(portable=True, app_dir=tmp_path).config_file)
    return SettingsDialog(store), store


def test_button_clears_and_persists_the_setup_check_flag(qapp, tmp_path, monkeypatch):
    dlg, store = _dlg(tmp_path)
    saves = []
    monkeypatch.setattr(store, "save_soon", lambda: saves.append(1))
    try:
        # The common case for pressing this: the check already finished once,
        # and the flag reflects that.
        store.config.gui.setup_check_done = True

        dlg._show_setup_btn.click()

        assert store.config.gui.setup_check_done is False
        assert saves == [1]
    finally:
        dlg.close()
        dlg.deleteLater()


def test_button_does_not_import_or_touch_the_panel_controller():
    # The design is deliberately a flag flip, not a wired-through reopen call:
    # settings_simple must not know the setup check module exists.
    import vrcc.gui.settings_simple as mod

    assert "setup_check" not in vars(mod)
    assert "setup_panel" not in vars(mod)


def test_button_label_and_tooltip_are_set(qapp, tmp_path):
    dlg, _ = _dlg(tmp_path)
    try:
        assert dlg._show_setup_btn.text() == "Bring back the setup steps"
        assert dlg._show_setup_btn.toolTip() == (
            "The setup steps come back beside the main window once you close "
            "Settings."
        )
    finally:
        dlg.close()
        dlg.deleteLater()
