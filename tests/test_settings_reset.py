"""Offscreen GUI tests for the "Reset to recommended settings" button: on Yes
it resets the models + device/thread fields, re-syncs every dialog widget from
config, persists, and rebuilds each engine kind exactly once; on No it changes
nothing; personal choices always survive; and a bare (headless) dialog works.
Monkeypatch QMessageBox.question both ways, as test_settings_onnx_cpu_prompt.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication, QMessageBox, QPushButton

from vrcc.core.config import ConfigStore, default_paths
from vrcc.gui import settings as settings_mod
from vrcc.gui import settings_reset
from vrcc.gui.settings import SettingsDialog


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


class _RecordingApply:
    def __init__(self):
        self.calls = []

    def reload_engine(self, kind):
        self.calls.append(("reload", kind))

    # Present so the live-apply façade contract is satisfied if ever reached.
    def apply_audio_device(self, device):
        return True

    def apply_osc(self, cfg):
        pass

    def apply_mute_sync(self, enabled):
        pass

    def apply_vad(self, cfg):
        pass


def _answer(monkeypatch, button):
    monkeypatch.setattr(
        QMessageBox, "question", staticmethod(lambda *a, **k: button)
    )


# Non-default engine values + personal sentinels, so a reset must visibly move
# every engine field while leaving the personal choices untouched.
def _dialog(tmp_path, monkeypatch, *, apply=None, on_model_change=None):
    monkeypatch.setattr(settings_mod, "device_names", lambda: ["Fake GPU"])
    monkeypatch.setattr(settings_mod.model_fit, "vram_warning", lambda *a, **k: None)
    store = ConfigStore(default_paths(portable=True, app_dir=tmp_path).config_file)
    cfg = store.config
    cfg.stt.model = "tiny"
    cfg.stt.device = "cuda"
    cfg.stt.device_index = 0
    cfg.stt.compute_type = "float32"
    cfg.stt.cpu_threads = 8
    cfg.stt.num_workers = 3
    cfg.translate.enabled = True
    cfg.translate.model = "m2m100-418M-int8"
    cfg.translate.device = "cuda"
    cfg.translate.compute_type = "float32"
    cfg.translate.inter_threads = 4
    cfg.translate.intra_threads = 8
    cfg.translate.max_queued_batches = 5
    # Personal choices the recommender must not touch.
    cfg.stt.source_language = "French"
    cfg.audio.device = "My Mic"
    cfg.osc.ip = "10.0.0.5"
    cfg.osc.port = 9100
    cfg.gui.font_scale = 1.2
    cfg.gui.ui_language = "ja"
    cfg.translate.targets = ["German"]
    dlg = SettingsDialog(
        store, download_manager=None, on_model_change=on_model_change, apply=apply
    )
    return dlg, store


def _rebuild_count(kind, model_calls, apply):
    return model_calls.count(kind) + apply.calls.count(("reload", kind))


def test_button_present_with_expected_text(qapp, tmp_path, monkeypatch):
    dlg, _ = _dialog(tmp_path, monkeypatch)
    try:
        buttons = [b.text() for b in dlg.findChildren(QPushButton)]
        assert "Reset to recommended settings" in buttons
    finally:
        dlg.close()
        dlg.deleteLater()


def test_no_changes_nothing(qapp, tmp_path, monkeypatch):
    model_calls = []
    apply = _RecordingApply()
    dlg, store = _dialog(
        tmp_path, monkeypatch, apply=apply, on_model_change=model_calls.append
    )
    try:
        before = store.config.model_dump()
        _answer(monkeypatch, QMessageBox.StandardButton.No)
        settings_reset.confirm_and_reset(dlg)
        assert store.config.model_dump() == before
        assert model_calls == []
        assert apply.calls == []
        assert dlg._stt_device_combo.currentData() == ("cuda", 0)
    finally:
        dlg.close()
        dlg.deleteLater()


def test_yes_resets_config_and_resyncs_widgets(qapp, tmp_path, monkeypatch):
    dlg, store = _dialog(tmp_path, monkeypatch, apply=_RecordingApply())
    try:
        _answer(monkeypatch, QMessageBox.StandardButton.Yes)
        settings_reset.confirm_and_reset(dlg)
        cfg = store.config
        # Engine fields returned to automatic in config...
        assert (cfg.stt.device, cfg.stt.device_index, cfg.stt.compute_type) == (
            "auto", 0, "auto",
        )
        assert (cfg.stt.cpu_threads, cfg.stt.num_workers) == (0, 1)
        assert (cfg.translate.device, cfg.translate.compute_type) == ("auto", "auto")
        assert (cfg.translate.inter_threads, cfg.translate.intra_threads) == (1, 0)
        assert cfg.translate.max_queued_batches == 0
        # ...and mirrored back onto every widget.
        assert dlg._stt_device_combo.currentData() == ("auto", 0)
        assert dlg._stt_compute_combo.currentText() == "auto"
        assert dlg._stt_cpu_threads_spin.value() == 0
        assert dlg._stt_workers_spin.value() == 1
        assert dlg._mt_device_combo.currentData() == ("auto", 0)
        assert dlg._mt_compute_combo.currentText() == "auto"
        assert dlg._mt_inter_spin.value() == 1
        assert dlg._mt_intra_spin.value() == 0
        assert dlg._mt_queued_spin.value() == 0
        assert dlg._model_combo.currentData() == cfg.stt.model
        assert dlg._translate_model_combo.currentData() == cfg.translate.model
        expected_mode = "Quality" if cfg.gui.profile == "quality" else "Speed"
        assert dlg._mode.value() == expected_mode
    finally:
        dlg.close()
        dlg.deleteLater()


def test_personal_fields_survive(qapp, tmp_path, monkeypatch):
    dlg, store = _dialog(tmp_path, monkeypatch, apply=_RecordingApply())
    try:
        _answer(monkeypatch, QMessageBox.StandardButton.Yes)
        settings_reset.confirm_and_reset(dlg)
        cfg = store.config
        assert cfg.stt.source_language == "French"
        assert cfg.audio.device == "My Mic"
        assert cfg.osc.ip == "10.0.0.5"
        assert cfg.osc.port == 9100
        assert cfg.gui.font_scale == 1.2
        assert cfg.gui.ui_language == "ja"
        assert cfg.translate.targets == ["German"]
    finally:
        dlg.close()
        dlg.deleteLater()


def test_each_engine_kind_rebuilds_exactly_once(qapp, tmp_path, monkeypatch):
    model_calls = []
    apply = _RecordingApply()
    dlg, _ = _dialog(
        tmp_path, monkeypatch, apply=apply, on_model_change=model_calls.append
    )
    try:
        _answer(monkeypatch, QMessageBox.StandardButton.Yes)
        settings_reset.confirm_and_reset(dlg)
        assert _rebuild_count("stt", model_calls, apply) == 1
        assert _rebuild_count("mt", model_calls, apply) == 1
        # A model change takes the swap path; it must NOT also reload_engine it.
        for kind in ("stt", "mt"):
            if kind in model_calls:
                assert ("reload", kind) not in apply.calls
    finally:
        dlg.close()
        dlg.deleteLater()


def test_headless_construction_and_reset_do_not_raise(qapp, tmp_path, monkeypatch):
    monkeypatch.setattr(settings_mod, "device_names", lambda: ["Fake GPU"])
    store = ConfigStore(default_paths(portable=True, app_dir=tmp_path).config_file)
    store.config.stt.device = "cuda"
    dlg = SettingsDialog(store)  # bare: no apply / download_manager / on_model_change
    try:
        _answer(monkeypatch, QMessageBox.StandardButton.Yes)
        settings_reset.confirm_and_reset(dlg)  # must not raise
        assert store.config.stt.device == "auto"
        assert dlg._stt_device_combo.currentData() == ("auto", 0)
    finally:
        dlg.close()
        dlg.deleteLater()
