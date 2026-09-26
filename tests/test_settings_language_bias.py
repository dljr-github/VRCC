"""Settings overrides persist separately from the wizard and apply without reload."""

import json
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from vrcc.core.bus import EventBus
from vrcc.core.config import ConfigStore
from vrcc.gui.settings import SettingsDialog
from vrcc.gui.widgets import set_combo_value
from vrcc.stt.engine import SttEngine
from vrcc.stt.language_bias import preferred_codes
from .stt_fakes import MODEL_DIR
from .test_stt_language_bias import DetectionModel


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def dialog(qapp, tmp_path):
    store = ConfigStore(tmp_path / "config.json")
    store.config.stt.source_language = "auto"
    store.config.stt.spoken_languages = ["English", "Japanese"]
    store.config.stt.device = "cpu"
    store.config.translate.enabled = False
    dlg = SettingsDialog(store)
    yield dlg, store
    dlg.close()
    dlg.deleteLater()
    store.save_now()


def tick(picker, name, checked):
    for i in range(picker.count()):
        item = picker.item(i)
        if item.text() == name:
            item.setCheckState(Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked)
            return
    raise AssertionError(f"missing language {name}")


def test_settings_override_changes_next_transcript_without_reloading(dialog):
    dlg, store = dialog
    model = DetectionModel([("fr", 0.45), ("en", 0.35), ("ja", 0.20)])
    engine = SttEngine(store.config.stt, MODEL_DIR, EventBus(), model_factory=lambda *a, **k: model)
    engine.load()
    audio = np.zeros(1600, dtype=np.float32)
    assert engine.transcribe(audio).language == "en"

    set_combo_value(dlg._language_bias_combo, "off")
    assert engine.transcribe(audio).language == "fr"
    set_combo_value(dlg._language_bias_combo, "custom")
    tick(dlg._language_bias_picker, "French", True)
    assert engine.transcribe(audio).language == "fr"
    tick(dlg._language_bias_picker, "French", False)
    tick(dlg._language_bias_picker, "English", True)
    assert engine.transcribe(audio).language == "en"
    assert store.config.stt.spoken_languages == ["English", "Japanese"]
    assert store.config.stt.source_language == "auto"


def test_custom_list_survives_switching_modes_and_reopening(dialog):
    dlg, store = dialog
    set_combo_value(dlg._language_bias_combo, "custom")
    tick(dlg._language_bias_picker, "Korean", True)
    for mode in ("off", "wizard", "custom"):
        set_combo_value(dlg._language_bias_combo, mode)
    assert preferred_codes(store.config.stt, False) == {"ko"}
    store.save_now()
    reloaded = ConfigStore(store.path)
    reloaded.load()
    reopened = SettingsDialog(reloaded)
    try:
        assert reopened._language_bias_combo.currentData() == "custom"
        assert preferred_codes(reloaded.config.stt, False) == {"ko"}
        assert reloaded.config.stt.spoken_languages == ["English", "Japanese"]
        tick(reopened._language_bias_picker, "Korean", False)
        assert preferred_codes(reloaded.config.stt, False) == set()
    finally:
        reopened.close()
        reopened.deleteLater()
        reloaded.save_now()


def test_empty_custom_list_does_not_fall_back_to_wizard(dialog):
    dlg, store = dialog
    set_combo_value(dlg._language_bias_combo, "custom")
    assert preferred_codes(store.config.stt, False) == set()
    assert not dlg._language_bias_picker.isHidden()
    set_combo_value(dlg._language_bias_combo, "wizard")
    assert preferred_codes(store.config.stt, False) == {"en", "ja"}
    assert dlg._language_bias_picker.isHidden()


def test_controls_follow_auto_language_and_supported_model(dialog):
    dlg, store = dialog
    set_combo_value(dlg._language_bias_combo, "custom")
    assert dlg._language_bias_combo.isEnabled()
    set_combo_value(dlg._source_combo, "English")
    assert not dlg._language_bias_combo.isEnabled()
    assert not dlg._language_bias_picker.isEnabled()
    set_combo_value(dlg._source_combo, "auto")
    assert dlg._language_bias_combo.isEnabled()
    set_combo_value(dlg._model_combo, "parakeet-tdt-0.6b-v3")
    assert not dlg._language_bias_combo.isEnabled()
    set_combo_value(dlg._model_combo, "small")
    assert dlg._language_bias_combo.isEnabled()
    assert store.config.stt.language_bias_mode == "custom"


def test_old_config_defaults_to_wizard_and_off_survives_reload(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"stt": {"source_language": "auto", "spoken_languages": ["Japanese"]}}))
    store = ConfigStore(path)
    store.load()
    assert store.config.stt.language_bias_mode == "wizard"
    assert preferred_codes(store.config.stt, False) == {"ja"}
    store.config.stt.language_bias_mode = "off"
    store.save_now()
    reloaded = ConfigStore(path)
    reloaded.load()
    assert preferred_codes(reloaded.config.stt, False) == set()
