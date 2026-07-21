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

from vrcc.core import recommend
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

    def apply_vad(self, cfg):
        self.calls.append(("vad", cfg))

    # Present so the live-apply façade contract is satisfied if ever reached.
    def apply_audio_device(self, device):
        return True

    def apply_osc(self, cfg):
        pass

    def apply_mute_sync(self, enabled):
        pass


def _answer(monkeypatch, button):
    monkeypatch.setattr(
        QMessageBox, "question", staticmethod(lambda *a, **k: button)
    )


# Non-default engine values + personal sentinels, so a reset must visibly move
# every engine field while leaving the personal choices untouched.
def _dialog(tmp_path, monkeypatch, *, apply=None, on_model_change=None):
    monkeypatch.setattr(settings_mod.settings_advanced, "device_names", lambda: ["Fake GPU"])
    monkeypatch.setattr(settings_mod.model_fit, "vram_warning", lambda *a, **k: None)
    # Pin the hardware verdict: a CPU verdict makes the reset bind the device
    # to "cpu" (by design), so unpinned tests would assert different devices
    # on a GPU dev box and a GPU-less CI runner.
    monkeypatch.setattr(recommend, "default_device_choice", lambda: "gpu")
    monkeypatch.setattr(recommend, "detect_tier", lambda: "gpu_high")
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
        assert "Recommended setup for this PC" in buttons
    finally:
        dlg.close()
        dlg.deleteLater()


def test_reset_button_on_simple_tab_triggers_confirm_and_reset(qapp, tmp_path, monkeypatch):
    dlg, _ = _dialog(tmp_path, monkeypatch)
    try:
        simple = dlg._tabs.widget(0).widget()
        buttons = [
            b
            for b in simple.findChildren(QPushButton)
            if b.text() == settings_reset.reset_button_text()
        ]
        assert len(buttons) == 1
        calls = []
        monkeypatch.setattr(settings_reset, "confirm_and_reset", calls.append)
        buttons[0].click()
        assert calls == [dlg]
    finally:
        dlg.close()
        dlg.deleteLater()


def test_advanced_page_has_no_reset_buttons(qapp, tmp_path, monkeypatch):
    # The Mode control on the Simple tab applies the same Speed/Quality
    # profiles, and the recommended reset lives there too, so the Advanced
    # page carries no reset buttons at all.
    dlg, _ = _dialog(tmp_path, monkeypatch)
    try:
        advanced = next(
            dlg._tabs.widget(i).widget()
            for i in range(dlg._tabs.count())
            if dlg._tabs.tabText(i) == "Advanced / Power users"
        )
        texts = [b.text() for b in advanced.findChildren(QPushButton)]
        assert "Reset to Speed preset" not in texts
        assert "Reset to Quality preset" not in texts
        assert settings_reset.reset_button_text() not in texts
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


def test_yes_pushes_vad_timings_live_exactly_once(qapp, tmp_path, monkeypatch):
    # The reset's apply_profile changes the VAD timings; the running Segmenter
    # must get them now, not on the next unrelated VAD edit.
    apply = _RecordingApply()
    dlg, store = _dialog(tmp_path, monkeypatch, apply=apply)
    try:
        _answer(monkeypatch, QMessageBox.StandardButton.Yes)
        settings_reset.confirm_and_reset(dlg)
        assert [c for c in apply.calls if c[0] == "vad"] == [("vad", store.config.vad)]
        # The re-baseline covers the hand-pushed VAD: the close flush re-fires nothing.
        dlg._apply_live_changes()
        assert [c for c in apply.calls if c[0] == "vad"] == [("vad", store.config.vad)]
    finally:
        dlg.close()
        dlg.deleteLater()


def test_headless_construction_and_reset_do_not_raise(qapp, tmp_path, monkeypatch):
    monkeypatch.setattr(settings_mod.settings_advanced, "device_names", lambda: ["Fake GPU"])
    # Bare construction skips the _dialog helper, so the hardware verdict is
    # pinned here too: a CPU verdict binds the device to "cpu" by design.
    monkeypatch.setattr(recommend, "default_device_choice", lambda: "gpu")
    monkeypatch.setattr(recommend, "detect_tier", lambda: "gpu_high")
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


def test_reset_defaults_resets_tuning_keeps_personal(tmp_path, monkeypatch):
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication, QMessageBox
    from vrcc.core.config import AppConfig, ConfigStore
    from vrcc.gui import settings_reset
    from vrcc.gui.settings import SettingsDialog

    QApplication.instance() or QApplication([])
    store = ConfigStore(tmp_path / "config.json")
    store.load()
    # Personal choices to preserve.
    store.config.audio.device = "My USB Mic"
    store.config.stt.source_language = "Japanese"
    store.config.translate.targets = ["English"]
    store.config.osc.ip = "10.0.0.5"
    # Tuning to be reset away from defaults.
    store.config.vad.threshold = 0.60
    store.config.audio.gain_db = 12.0
    store.config.audio.auto_gain = False
    store.config.vad.sentence_inject = False
    store.config.gui.update_check_enabled = False
    store.config.stt.avg_logprob_gate = -2.5
    store.config.stt.no_speech_gate = 0.9
    store.config.stt.condition_on_previous_text = True
    store.config.audio.denoise_enabled = True
    store.config.audio.denoise_strength = 0.9

    dlg = SettingsDialog(store)
    monkeypatch.setattr(QMessageBox, "question",
                        lambda *a, **k: QMessageBox.StandardButton.Yes)
    try:
        settings_reset.confirm_and_reset_defaults(dlg)
        d = AppConfig()
        # Tuning reset.
        assert store.config.vad.threshold == d.vad.threshold
        assert store.config.audio.gain_db == d.audio.gain_db
        assert store.config.audio.auto_gain == d.audio.auto_gain
        assert store.config.vad.sentence_inject == d.vad.sentence_inject
        assert store.config.gui.update_check_enabled == d.gui.update_check_enabled
        assert store.config.audio.denoise_enabled == d.audio.denoise_enabled
        assert store.config.audio.denoise_strength == d.audio.denoise_strength
        # Personal preserved.
        assert store.config.audio.device == "My USB Mic"
        assert store.config.stt.source_language == "Japanese"
        assert store.config.translate.targets == ["English"]
        assert store.config.osc.ip == "10.0.0.5"
        # Widgets themselves reflect the reset, not just the config: an open
        # Advanced group must not show stale values after a reset.
        assert dlg._stt_avg_gate_spin.value() == d.stt.avg_logprob_gate
        assert dlg._stt_ns_gate_spin.value() == d.stt.no_speech_gate
        assert dlg._stt_cond_check.isChecked() == d.stt.condition_on_previous_text
        assert dlg._sensitivity.value() == 90 - round(d.vad.threshold * 100)
        assert dlg._sentence_inject_check.isChecked() == d.vad.sentence_inject
        assert dlg._update_check.isChecked() == d.gui.update_check_enabled
        assert dlg._denoise_check.isChecked() == d.audio.denoise_enabled
        assert dlg._denoise_strength.value() == round(d.audio.denoise_strength * 100)
        assert dlg._denoise_strength.isEnabled() == d.audio.denoise_enabled
    finally:
        dlg.close()
        dlg.deleteLater()
