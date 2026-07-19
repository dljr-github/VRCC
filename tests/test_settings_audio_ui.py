"""Offscreen GUI tests for the Settings dialog's audio/microphone controls."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication, QPushButton

from vrcc.core.config import ConfigStore, default_paths
from vrcc.gui import settings_audio
from vrcc.gui.settings import SettingsDialog

_SIMPLE_TAB_INDEX = 0


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _store(tmp_path):
    return ConfigStore(default_paths(portable=True, app_dir=tmp_path).config_file)


class _RecordingApply:
    """Records refresh_input_devices calls; nothing else touches this fake."""

    def __init__(self) -> None:
        self.refresh_calls: list = []

    def refresh_input_devices(self, device_cfg):
        self.refresh_calls.append(device_cfg)
        return [(1, "Mic A")]


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


def test_gain_controls_bind(qapp, tmp_path):
    store = _store(tmp_path)
    dlg = SettingsDialog(store)
    try:
        # Start from a known state (auto off), since auto is on by default.
        dlg._auto_gain_check.setChecked(False)
        assert store.config.audio.auto_gain is False
        assert dlg._gain_slider.isEnabled()
        dlg._gain_slider.setValue(12)
        assert store.config.audio.gain_db == 12.0
        dlg._auto_gain_check.setChecked(True)
        assert store.config.audio.auto_gain is True
        # Auto on greys the manual slider.
        assert not dlg._gain_slider.isEnabled()
    finally:
        dlg.close()
        dlg.deleteLater()


def test_device_refresh_repopulates_without_changing_selection(qapp, tmp_path, monkeypatch):
    store = _store(tmp_path)
    store.load()

    monkeypatch.setattr(
        settings_audio, "list_input_devices",
        lambda: [(1, "Mic A"), (2, "Mic B")], raising=False,
    )

    class FakeApply:
        def refresh_input_devices(self, device_cfg):
            return [(1, "Mic A"), (2, "Mic B")]

    dlg = SettingsDialog(store, apply=FakeApply())
    try:
        before = store.config.audio.device
        settings_audio._repopulate_input_devices(dlg)
        assert dlg._input_device_combo.count() >= 3  # Auto + two mics
        assert store.config.audio.device == before  # no spurious swap
    finally:
        dlg.close()
        dlg.deleteLater()


def test_opening_settings_does_not_refresh_input_devices(qapp, tmp_path):
    # Bug: __init__ used to call refresh_input_devices on every open. That is
    # a full PortAudio _terminate()/_initialize() plus pipeline stop()/
    # start(), which can freeze the GUI for seconds and start() calls
    # segmenter.reset(), silently discarding a mid-sentence utterance. The
    # combo is already filled cheaply at build time (list_input_devices(),
    # no reinit), so opening the dialog must not touch the apply handle.
    store = _store(tmp_path)
    apply = _RecordingApply()
    dlg = SettingsDialog(store, apply=apply)
    try:
        assert apply.refresh_calls == []
        assert dlg._input_device_combo.count() >= 1  # still filled at build time
    finally:
        dlg.close()
        dlg.deleteLater()


def test_refresh_button_still_calls_refresh_input_devices(qapp, tmp_path):
    # The explicit Refresh button is the only remaining path to the heavy
    # PortAudio reinit; it must be unchanged by removing it from __init__.
    store = _store(tmp_path)
    apply = _RecordingApply()
    dlg = SettingsDialog(store, apply=apply)
    try:
        page = dlg._tabs.widget(_SIMPLE_TAB_INDEX).widget()
        refresh_btn = next(
            b for b in page.findChildren(QPushButton) if b.text() == "Refresh"
        )
        refresh_btn.click()
        assert apply.refresh_calls == [store.config.audio.device]
    finally:
        dlg.close()
        dlg.deleteLater()
