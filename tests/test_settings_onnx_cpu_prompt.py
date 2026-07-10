"""Offscreen GUI tests for the explicit-CUDA CPU offer: onnx-asr models
(Parakeet) run about as fast on the CPU, so picking one while stt.device ==
"cuda" (or picking cuda while one is active) asks whether to use the CPU
instead. The prompt must never fire for device "auto" -- the engine already
prefers the CPU there.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication, QMessageBox

from vrcc.core.config import ConfigStore, default_paths
from vrcc.gui import settings as settings_mod
from vrcc.gui.settings import SettingsDialog


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _dialog(tmp_path, monkeypatch, model_id="small", device="auto"):
    # The fit prompt is not under test and would block offscreen; skip it.
    monkeypatch.setattr(settings_mod.model_fit, "vram_warning", lambda *a, **k: None)
    # A cuda combo entry must exist even on GPU-less test machines.
    monkeypatch.setattr(settings_mod.settings_advanced, "device_names", lambda: ["Fake GPU"])
    store = ConfigStore(default_paths(portable=True, app_dir=tmp_path).config_file)
    store.config.stt.model = model_id
    store.config.stt.device = device
    return SettingsDialog(store), store  # headless: all models offered


def _capture_question(monkeypatch, answer):
    asked: list[str] = []

    def question(parent, title, text, *args, **kwargs):
        asked.append(text)
        return answer

    monkeypatch.setattr(QMessageBox, "question", question)
    return asked


def _select_device(dlg, data):
    combo = dlg._stt_device_combo
    for i in range(combo.count()):
        if combo.itemData(i) == data:
            combo.setCurrentIndex(i)
            return
    raise AssertionError(f"device entry {data!r} not offered")


class _RecordingApply:
    """Records reload_engine only; the other facade hooks are quiet no-ops."""

    def __init__(self):
        self.calls = []

    def reload_engine(self, kind):
        self.calls.append(("reload", kind))

    def apply_audio_device(self, device):
        return True

    def apply_osc(self, cfg):
        pass

    def apply_mute_sync(self, enabled):
        pass

    def apply_vad(self, cfg):
        pass


def test_onnx_pick_on_cuda_yes_flips_device_to_cpu(qapp, tmp_path, monkeypatch):
    dlg, store = _dialog(tmp_path, monkeypatch, model_id="small", device="cuda")
    try:
        asked = _capture_question(monkeypatch, QMessageBox.StandardButton.Yes)
        combo = dlg._model_combo
        combo.setCurrentIndex(combo.findData("parakeet-tdt-0.6b-v3"))
        assert len(asked) == 1
        assert "VRAM" in asked[0]
        assert store.config.stt.model == "parakeet-tdt-0.6b-v3"  # switch kept
        assert store.config.stt.device == "cpu"
        assert dlg._stt_device_combo.currentData() == ("cpu", 0)
    finally:
        dlg.close()
        dlg.deleteLater()


def test_onnx_pick_on_cuda_no_keeps_cuda(qapp, tmp_path, monkeypatch):
    dlg, store = _dialog(tmp_path, monkeypatch, model_id="small", device="cuda")
    try:
        asked = _capture_question(monkeypatch, QMessageBox.StandardButton.No)
        combo = dlg._model_combo
        combo.setCurrentIndex(combo.findData("parakeet-tdt-0.6b-v3"))
        assert len(asked) == 1
        assert store.config.stt.model == "parakeet-tdt-0.6b-v3"
        assert store.config.stt.device == "cuda"
    finally:
        dlg.close()
        dlg.deleteLater()


def test_onnx_pick_on_auto_never_prompts(qapp, tmp_path, monkeypatch):
    dlg, store = _dialog(tmp_path, monkeypatch, model_id="small", device="auto")
    try:
        asked = _capture_question(monkeypatch, QMessageBox.StandardButton.Yes)
        combo = dlg._model_combo
        combo.setCurrentIndex(combo.findData("parakeet-tdt-0.6b-v3"))
        assert asked == []
        assert store.config.stt.device == "auto"
    finally:
        dlg.close()
        dlg.deleteLater()


def test_accepted_offer_rebuilds_the_engine_exactly_once(qapp, tmp_path, monkeypatch):
    # The model hot-swap already rebuilds with the flipped cpu device (config
    # is re-read at build); the debounced flush must not force a second,
    # identical rebuild against its stale device baseline.
    monkeypatch.setattr(settings_mod.model_fit, "vram_warning", lambda *a, **k: None)
    monkeypatch.setattr(settings_mod.settings_advanced, "device_names", lambda: ["Fake GPU"])
    store = ConfigStore(default_paths(portable=True, app_dir=tmp_path).config_file)
    store.config.stt.model = "small"
    store.config.stt.device = "cuda"
    apply = _RecordingApply()
    swaps = []
    dlg = SettingsDialog(store, on_model_change=swaps.append, apply=apply)
    try:
        _capture_question(monkeypatch, QMessageBox.StandardButton.Yes)
        combo = dlg._model_combo
        combo.setCurrentIndex(combo.findData("parakeet-tdt-0.6b-v3"))
        assert store.config.stt.device == "cpu"
        assert swaps == ["stt"]
        # Flush the pending debounce the way dialog close does.
        dlg._apply_timer.stop()
        dlg._apply_live_changes()
        assert apply.calls == []
    finally:
        dlg.close()
        dlg.deleteLater()


def test_cuda_pick_with_onnx_active_yes_flips_back(qapp, tmp_path, monkeypatch):
    dlg, store = _dialog(
        tmp_path, monkeypatch, model_id="parakeet-tdt-0.6b-v3", device="cpu"
    )
    try:
        asked = _capture_question(monkeypatch, QMessageBox.StandardButton.Yes)
        _select_device(dlg, ("cuda", 0))
        assert len(asked) == 1
        assert store.config.stt.device == "cpu"
        assert dlg._stt_device_combo.currentData() == ("cpu", 0)
    finally:
        dlg.close()
        dlg.deleteLater()


def test_cuda_pick_with_onnx_active_no_keeps_cuda(qapp, tmp_path, monkeypatch):
    dlg, store = _dialog(
        tmp_path, monkeypatch, model_id="parakeet-tdt-0.6b-v3", device="cpu"
    )
    try:
        asked = _capture_question(monkeypatch, QMessageBox.StandardButton.No)
        _select_device(dlg, ("cuda", 0))
        assert len(asked) == 1
        assert store.config.stt.device == "cuda"
        assert dlg._stt_device_combo.currentData() == ("cuda", 0)
    finally:
        dlg.close()
        dlg.deleteLater()


def test_cuda_pick_with_whisper_model_never_prompts(qapp, tmp_path, monkeypatch):
    dlg, store = _dialog(tmp_path, monkeypatch, model_id="small", device="cpu")
    try:
        asked = _capture_question(monkeypatch, QMessageBox.StandardButton.Yes)
        _select_device(dlg, ("cuda", 0))
        assert asked == []
        assert store.config.stt.device == "cuda"
    finally:
        dlg.close()
        dlg.deleteLater()


def test_translate_device_cuda_never_prompts(qapp, tmp_path, monkeypatch):
    # The offer concerns the voice device only; the MT device combo picking
    # cuda under an onnx-asr voice model must stay silent.
    dlg, store = _dialog(
        tmp_path, monkeypatch, model_id="parakeet-tdt-0.6b-v3", device="cpu"
    )
    try:
        asked = _capture_question(monkeypatch, QMessageBox.StandardButton.Yes)
        combo = dlg._mt_device_combo
        for i in range(combo.count()):
            if combo.itemData(i) == ("cuda", 0):
                combo.setCurrentIndex(i)
                break
        assert asked == []
        assert store.config.translate.device == "cuda"
        assert store.config.stt.device == "cpu"
    finally:
        dlg.close()
        dlg.deleteLater()
