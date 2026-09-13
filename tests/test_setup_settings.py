"""The Simple page's "Bring back the setup steps" button: it must bump
``gui.setup_check_requests`` and clear ``gui.setup_check_done``, then persist
both, so the running setup check controller (which this dialog never touches
directly) notices on its own next poll.

Split from tests/test_settings_ui.py, which is at the repo's 500-line cap.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path

import pytest
from PySide6.QtWidgets import QApplication

import vrcc.gui.settings_simple
from vrcc.core.config import ConfigStore, default_paths
from vrcc.gui.settings import SettingsDialog


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _dlg(tmp_path):
    store = ConfigStore(default_paths(portable=True, app_dir=tmp_path).config_file)
    return SettingsDialog(store), store


def test_button_records_a_request_and_clears_the_flag(qapp, tmp_path, monkeypatch):
    dlg, store = _dlg(tmp_path)
    saves = []
    monkeypatch.setattr(store, "save_soon", lambda: saves.append(1))
    try:
        # The check already finished once, and the flag reflects that.
        store.config.gui.setup_check_done = True
        before = store.config.gui.setup_check_requests

        dlg._show_setup_btn.click()

        assert store.config.gui.setup_check_requests == before + 1
        assert store.config.gui.setup_check_done is False
        assert saves == [1]
    finally:
        dlg.close()
        dlg.deleteLater()


def test_every_press_records_a_fresh_request(qapp, tmp_path, monkeypatch):
    """The state the flag alone cannot express. Someone who dismissed the
    panel before finishing setup already has setup_check_done False, so
    clearing it writes nothing new; and a press deliberately leaves it False,
    so a second press in the same session is the same problem again. The
    counter has to move both times or the button is dead."""
    dlg, store = _dlg(tmp_path)
    monkeypatch.setattr(store, "save_soon", lambda: None)
    try:
        assert store.config.gui.setup_check_done is False
        assert store.config.gui.setup_check_requests == 0

        dlg._show_setup_btn.click()
        assert store.config.gui.setup_check_requests == 1

        dlg._show_setup_btn.click()
        assert store.config.gui.setup_check_requests == 2
        assert store.config.gui.setup_check_done is False
    finally:
        dlg.close()
        dlg.deleteLater()


def test_the_button_is_a_config_write_not_a_controller_call(qapp, tmp_path, monkeypatch):
    """The design is deliberately a config write, not a wired-through reopen
    call: settings_simple must not know the setup check modules exist. The
    press still has to land with nothing whatsoever listening, which is the
    half an "these two names are absent from the module" assertion could not
    see (an empty module passes that)."""
    import ast

    dlg, store = _dlg(tmp_path)
    monkeypatch.setattr(store, "save_soon", lambda: None)
    try:
        dlg._show_setup_btn.click()
        assert store.config.gui.setup_check_requests == 1
    finally:
        dlg.close()
        dlg.deleteLater()

    source = Path(vrcc.gui.settings_simple.__file__).read_text(encoding="utf-8")
    imported = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            imported += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            imported += [(node.module or "")] + [alias.name for alias in node.names]
    assert not [name for name in imported if "setup_check" in name or "setup_panel" in name]


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
