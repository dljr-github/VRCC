"""Offscreen GUI tests for the language-change model nudge: switching the
spoken language to one the active voice model cannot transcribe offers the
best downloaded compatible model. Settings applies an accepted switch through
the model combo (the normal change path: fit prompt, Mode lock, hot-swap
callback); the main window writes config directly and pokes the callback.
No compatible download, "auto", and a covered language must all stay silent.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication, QMessageBox

from vrcc.core.bus import EventBus
from vrcc.core.config import ConfigStore, default_paths
from vrcc.gui import settings as settings_mod
from vrcc.gui.bridge import BusBridge
from vrcc.gui.settings import SettingsDialog


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


class _FakeDM:
    """Presence checks only, like the real DownloadManager's is_*_downloaded."""

    def __init__(self, whisper=(), mt=()):
        self._w, self._m = set(whisper), set(mt)

    def is_whisper_downloaded(self, mid):
        return mid in self._w

    def is_mt_downloaded(self, spec):
        return spec.id in self._m


def _capture_question(monkeypatch, answer):
    asked: list[str] = []

    def question(parent, title, text, *args, **kwargs):
        asked.append(text)
        return answer

    monkeypatch.setattr(QMessageBox, "question", question)
    return asked


# -- Settings ----------------------------------------------------------------


def _dialog(tmp_path, monkeypatch, downloaded, model_id):
    # The fit prompt is not under test and would block offscreen; skip it.
    monkeypatch.setattr(settings_mod.model_fit, "vram_warning", lambda *a, **k: None)
    store = ConfigStore(default_paths(portable=True, app_dir=tmp_path).config_file)
    store.config.stt.model = model_id
    store.config.stt.device = "cpu"  # pins tier_for_config, machine-independent
    swaps: list[str] = []
    dlg = SettingsDialog(
        store,
        download_manager=_FakeDM(whisper=downloaded),
        on_model_change=swaps.append,
    )
    return dlg, store, swaps


def test_settings_nudge_yes_switches_through_model_combo(qapp, tmp_path, monkeypatch):
    dlg, store, swaps = _dialog(
        tmp_path, monkeypatch,
        downloaded={"parakeet-tdt-0.6b-v3", "small"},
        model_id="parakeet-tdt-0.6b-v3",
    )
    try:
        asked = _capture_question(monkeypatch, QMessageBox.StandardButton.Yes)
        dlg._source_combo.setCurrentText("Japanese")  # outside Parakeet's set
        assert len(asked) == 1
        assert "Japanese" in asked[0]
        assert store.config.stt.model == "small"
        assert dlg._model_combo.currentData() == "small"
        assert swaps == ["stt"]  # hot-swap ran via the normal change path
    finally:
        dlg.close()
        dlg.deleteLater()


def test_settings_nudge_no_keeps_model_and_language(qapp, tmp_path, monkeypatch):
    dlg, store, swaps = _dialog(
        tmp_path, monkeypatch,
        downloaded={"parakeet-tdt-0.6b-v3", "small"},
        model_id="parakeet-tdt-0.6b-v3",
    )
    try:
        asked = _capture_question(monkeypatch, QMessageBox.StandardButton.No)
        dlg._source_combo.setCurrentText("Japanese")
        assert len(asked) == 1
        assert store.config.stt.model == "parakeet-tdt-0.6b-v3"
        assert store.config.stt.source_language == "Japanese"  # change kept
        assert swaps == []
    finally:
        dlg.close()
        dlg.deleteLater()


def test_settings_nudge_silent_without_compatible_download(qapp, tmp_path, monkeypatch):
    dlg, store, swaps = _dialog(
        tmp_path, monkeypatch,
        downloaded={"parakeet-tdt-0.6b-v3"},  # nothing else on disk
        model_id="parakeet-tdt-0.6b-v3",
    )
    try:
        asked = _capture_question(monkeypatch, QMessageBox.StandardButton.Yes)
        dlg._source_combo.setCurrentText("Japanese")
        assert asked == []
        assert store.config.stt.model == "parakeet-tdt-0.6b-v3"
        assert swaps == []
    finally:
        dlg.close()
        dlg.deleteLater()


def test_settings_nudge_silent_for_auto_and_covered_language(qapp, tmp_path, monkeypatch):
    dlg, store, swaps = _dialog(
        tmp_path, monkeypatch,
        downloaded={"parakeet-tdt-0.6b-v3", "small"},
        model_id="parakeet-tdt-0.6b-v3",
    )
    try:
        asked = _capture_question(monkeypatch, QMessageBox.StandardButton.Yes)
        dlg._source_combo.setCurrentText("French")  # inside Parakeet's set
        dlg._source_combo.setCurrentText("auto")
        assert asked == []
        assert store.config.stt.model == "parakeet-tdt-0.6b-v3"
        assert swaps == []
    finally:
        dlg.close()
        dlg.deleteLater()


# -- main window ---------------------------------------------------------------


class _Pipeline:
    captioning_enabled = False

    def set_captioning(self, value):
        pass


def _window(tmp_path, downloaded, model_id):
    from vrcc.gui.main_window import MainWindow

    store = ConfigStore(tmp_path / "config.json")
    store.config.stt.model = model_id
    store.config.stt.device = "cpu"  # pins tier_for_config, machine-independent
    bridge = BusBridge(EventBus())
    swaps: list[str] = []
    window = MainWindow(
        bridge,
        store,
        _Pipeline(),
        lambda: None,
        lambda: None,
        download_manager=_FakeDM(whisper=downloaded),
        on_model_change=swaps.append,
    )
    return window, store, bridge, swaps


def test_main_window_nudge_yes_switches_and_hotswaps(qapp, tmp_path, monkeypatch):
    window, store, bridge, swaps = _window(
        tmp_path, {"parakeet-tdt-0.6b-v3", "small"}, "parakeet-tdt-0.6b-v3"
    )
    try:
        asked = _capture_question(monkeypatch, QMessageBox.StandardButton.Yes)
        window._source_combo.setCurrentText("Japanese")
        assert len(asked) == 1
        assert store.config.stt.model == "small"
        assert swaps == ["stt"]
    finally:
        window.close()
        window.deleteLater()
        bridge.detach()


def test_main_window_nudge_no_leaves_everything(qapp, tmp_path, monkeypatch):
    window, store, bridge, swaps = _window(
        tmp_path, {"parakeet-tdt-0.6b-v3", "small"}, "parakeet-tdt-0.6b-v3"
    )
    try:
        asked = _capture_question(monkeypatch, QMessageBox.StandardButton.No)
        window._source_combo.setCurrentText("Japanese")
        assert len(asked) == 1
        assert store.config.stt.model == "parakeet-tdt-0.6b-v3"
        assert store.config.stt.source_language == "Japanese"
        assert swaps == []
    finally:
        window.close()
        window.deleteLater()
        bridge.detach()
