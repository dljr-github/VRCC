"""Offscreen GUI tests for the Settings dialog's live apply: each widget edit
must push into the running stack through the injected apply handle (engine
device/threads -> reload_engine, VAD/OSC/mute -> their apply_* methods, theme/
text size -> a QApplication retint), coalesced onto a debounce timer and flushed
on close. A bare dialog (apply=None) stays inert, and the restart banner is gone.
Also guards MainWindow.disconnect_bridge (the UI-language rebuild's teardown).
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication, QCheckBox, QSpinBox

from vrcc.core.config import ConfigStore, default_paths
from vrcc.gui import settings as settings_mod
from vrcc.gui.settings import SettingsDialog

_VRCHAT_TAB_INDEX = 3


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


class _RecordingApply:
    def __init__(self):
        self.calls = []

    def apply_audio_device(self, device):
        self.calls.append(("audio", device))
        return True

    def reload_engine(self, kind):
        self.calls.append(("reload", kind))

    def apply_osc(self, cfg):
        self.calls.append(("osc", cfg.port, cfg.ip))

    def apply_mute_sync(self, enabled):
        self.calls.append(("mute", enabled))

    def apply_vad(self, cfg):
        self.calls.append(("vad", round(cfg.threshold, 4), cfg.finalize_silence_ms))


def _store(tmp_path):
    return ConfigStore(default_paths(portable=True, app_dir=tmp_path).config_file)


def _select_data(combo, data):
    """Select the combo row whose itemData equals ``data`` (findData is
    unreliable for tuple user-data)."""
    for i in range(combo.count()):
        if combo.itemData(i) == data:
            combo.setCurrentIndex(i)
            return
    raise AssertionError(f"no combo item with data {data!r}")


# -- engine fields rebuild through reload_engine (one path) ------------------


def test_stt_device_change_reloads_stt_and_not_the_model_path(qapp, tmp_path):
    store = _store(tmp_path)
    apply = _RecordingApply()
    model_calls = []
    dlg = SettingsDialog(store, apply=apply, on_model_change=model_calls.append)
    try:
        _select_data(dlg._stt_device_combo, ("cpu", 0))
        dlg._apply_live_changes()
        assert apply.calls == [("reload", "stt")]
        assert model_calls == []  # device changes never take the model-swap path
        # Idempotent: flushing again without a change re-fires nothing.
        dlg._apply_live_changes()
        assert apply.calls == [("reload", "stt")]
    finally:
        dlg.deleteLater()


def test_mt_device_change_reloads_mt(qapp, tmp_path):
    store = _store(tmp_path)
    apply = _RecordingApply()
    dlg = SettingsDialog(store, apply=apply)
    try:
        _select_data(dlg._mt_device_combo, ("cpu", 0))
        dlg._apply_live_changes()
        assert apply.calls == [("reload", "mt")]
    finally:
        dlg.deleteLater()


def test_stt_compute_change_reloads_stt(qapp, tmp_path):
    store = _store(tmp_path)
    apply = _RecordingApply()
    dlg = SettingsDialog(store, apply=apply)
    try:
        dlg._stt_compute_combo.setCurrentText("int8")
        dlg._apply_live_changes()
        assert apply.calls == [("reload", "stt")]
    finally:
        dlg.deleteLater()


# -- non-engine live fields --------------------------------------------------


def test_vad_timing_change_applies_vad(qapp, tmp_path):
    store = _store(tmp_path)
    apply = _RecordingApply()
    dlg = SettingsDialog(store, apply=apply)
    try:
        dlg._vad_spins["finalize_silence_ms"].setValue(750)
        dlg._apply_live_changes()
        assert apply.calls == [
            ("vad", round(store.config.vad.threshold, 4), 750)
        ]
    finally:
        dlg.deleteLater()


def test_sensitivity_slider_applies_vad(qapp, tmp_path):
    store = _store(tmp_path)
    apply = _RecordingApply()
    dlg = SettingsDialog(store, apply=apply)
    try:
        dlg._sensitivity.setValue(45)  # -> vad.threshold 0.45
        dlg._apply_live_changes()
        assert len(apply.calls) == 1
        tag, thr, _fin = apply.calls[0]
        assert tag == "vad" and abs(thr - 0.45) < 1e-6
    finally:
        dlg.deleteLater()


def test_osc_port_change_applies_osc(qapp, tmp_path):
    store = _store(tmp_path)
    apply = _RecordingApply()
    dlg = SettingsDialog(store, apply=apply)
    try:
        page = dlg._tabs.widget(_VRCHAT_TAB_INDEX).widget()
        port = next(s for s in page.findChildren(QSpinBox) if s.maximum() == 65535)
        port.setValue(9001)
        dlg._apply_live_changes()
        assert apply.calls == [("osc", 9001, store.config.osc.ip)]
    finally:
        dlg.deleteLater()


def test_mute_toggle_applies_mute_sync(qapp, tmp_path):
    store = _store(tmp_path)
    apply = _RecordingApply()
    dlg = SettingsDialog(store, apply=apply)
    try:
        page = dlg._tabs.widget(_VRCHAT_TAB_INDEX).widget()
        check = next(
            c for c in page.findChildren(QCheckBox)
            if c.text() == "React when I mute myself in VRChat"
        )
        check.setChecked(not check.isChecked())
        dlg._apply_live_changes()
        assert apply.calls == [("mute", store.config.mute_sync.enabled)]
    finally:
        dlg.deleteLater()


def test_font_scale_change_retints_app_not_via_apply(qapp, tmp_path, monkeypatch):
    store = _store(tmp_path)
    apply = _RecordingApply()
    retint = []
    monkeypatch.setattr(
        settings_mod, "apply_theme_guarded", lambda *a, **k: retint.append("theme")
    )
    monkeypatch.setattr(
        settings_mod, "apply_font_scale", lambda *a, **k: retint.append("font")
    )
    dlg = SettingsDialog(store, apply=apply)
    try:
        dlg._text_size.set_value("Large")
        dlg._apply_live_changes()
        assert retint == ["theme", "font"]
        assert store.config.gui.font_scale == 1.2
        assert apply.calls == []  # appearance never routes through the apply handle
    finally:
        dlg.deleteLater()


# -- coalescing, close-flush, headless ---------------------------------------


def test_widget_edit_arms_the_debounce_timer(qapp, tmp_path):
    store = _store(tmp_path)
    apply = _RecordingApply()
    dlg = SettingsDialog(store, apply=apply)
    try:
        assert not dlg._apply_timer.isActive()
        _select_data(dlg._stt_device_combo, ("cpu", 0))
        assert dlg._apply_timer.isActive()  # coalesced, not applied per keystroke
    finally:
        dlg.deleteLater()


def test_close_flushes_pending_live_changes(qapp, tmp_path):
    store = _store(tmp_path)
    apply = _RecordingApply()
    dlg = SettingsDialog(store, apply=apply)
    _select_data(dlg._stt_device_combo, ("cpu", 0))
    dlg.done(0)  # accept/reject/close all funnel through done() and must flush
    assert apply.calls == [("reload", "stt")]
    dlg.deleteLater()


def test_apply_none_is_inert_but_still_writes_config(qapp, tmp_path):
    store = _store(tmp_path)
    dlg = SettingsDialog(store)  # headless: no apply handle
    try:
        _select_data(dlg._stt_device_combo, ("cpu", 0))
        assert not dlg._apply_timer.isActive()  # nothing to flush into
        dlg._apply_live_changes()  # must not raise
        dlg.done(0)  # close path must not raise either
        assert store.config.stt.device == "cpu"  # config still saved as before
    finally:
        dlg.deleteLater()


# -- UI-language rebuild teardown: the old window must detach from the bridge -


def test_disconnect_bridge_stops_events_reaching_the_window(qapp, tmp_path):
    from vrcc.core.bus import EventBus
    from vrcc.core.events import MuteChanged
    from vrcc.gui.bridge import BusBridge
    from vrcc.gui.main_window import MainWindow

    class _Pipe:
        captioning_enabled = False

        def set_captioning(self, value):
            pass

    bus = EventBus()
    bridge = BusBridge(bus)
    store = _store(tmp_path)
    win = MainWindow(bridge, store, _Pipe(), lambda: None, lambda: None)
    try:
        bus.publish(MuteChanged(muted=True))
        QApplication.instance().processEvents()
        assert win._mute_chip.text() == "MUTED"  # connected: the slot ran

        win.disconnect_bridge()
        bus.publish(MuteChanged(muted=False))  # would flip to LIVE if connected
        QApplication.instance().processEvents()
        assert win._mute_chip.text() == "MUTED"  # detached: no delivery
    finally:
        win.close()
        win.deleteLater()
        bridge.detach()
