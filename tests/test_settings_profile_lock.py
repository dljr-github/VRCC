"""Offscreen GUI tests for the Speed/Quality Mode greying: onnx-asr voice
models (Parakeet) decode greedily, so the profile's beam/temperature presets
can't tune their captions -- the Mode control must grey out with an
explanatory tooltip (and its visible description label must swap to the same
explanation) while such a model is active, and recover on a switch.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication

from vrcc.core.config import ConfigStore, default_paths
from vrcc.gui import settings as settings_mod
from vrcc.gui.settings import SettingsDialog
from vrcc.gui.settings_pages import _MODE_DESC, _MODE_LOCKED_TOOLTIP, _MODE_TOOLTIP


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _dialog(tmp_path, model_id):
    store = ConfigStore(default_paths(portable=True, app_dir=tmp_path).config_file)
    store.config.stt.model = model_id
    return SettingsDialog(store), store  # headless: all models offered


@pytest.mark.parametrize("model_id", ["parakeet-tdt-0.6b-v3"])
def test_mode_disabled_with_tooltip_for_greedy_models(qapp, tmp_path, model_id):
    dlg, store = _dialog(tmp_path, model_id)
    try:
        assert not dlg._mode.isEnabled()
        assert dlg._mode.toolTip() == _MODE_LOCKED_TOOLTIP
        # The visible description must not advertise a Speed/Quality trade-off
        # the locked control can't deliver.
        assert dlg._mode_desc.text() == _MODE_LOCKED_TOOLTIP
        # The segments grey with the control, and the stored profile stays
        # put: its VAD/translation parts still apply at the current position.
        assert not dlg._mode._buttons["Quality"].isEnabled()
        assert store.config.gui.profile == "latency"
    finally:
        dlg.close()
        dlg.deleteLater()


@pytest.mark.parametrize("model_id", ["small", "large-v3-turbo"])
def test_mode_enabled_for_beam_search_models(qapp, tmp_path, model_id):
    dlg, _ = _dialog(tmp_path, model_id)
    try:
        assert dlg._mode.isEnabled()
        assert dlg._mode.toolTip() == _MODE_TOOLTIP
        # A recommendation line may follow the base explanation (Task D), so the
        # description leads with _MODE_DESC rather than equalling it exactly.
        assert dlg._mode_desc.text().startswith(_MODE_DESC)
    finally:
        dlg.close()
        dlg.deleteLater()


def test_mode_reacts_to_model_switch_in_dialog(qapp, tmp_path, monkeypatch):
    # The fit prompt is not under test and would block offscreen; skip it.
    monkeypatch.setattr(settings_mod.model_fit, "vram_warning", lambda *a, **k: None)
    dlg, store = _dialog(tmp_path, "small")
    try:
        combo = dlg._model_combo
        combo.setCurrentIndex(combo.findData("parakeet-tdt-0.6b-v3"))
        assert store.config.stt.model == "parakeet-tdt-0.6b-v3"
        assert not dlg._mode.isEnabled()
        assert dlg._mode.toolTip() == _MODE_LOCKED_TOOLTIP
        assert dlg._mode_desc.text() == _MODE_LOCKED_TOOLTIP
        assert store.config.gui.profile == "latency"  # position untouched

        combo.setCurrentIndex(combo.findData("small"))
        assert dlg._mode.isEnabled()
        assert dlg._mode.toolTip() == _MODE_TOOLTIP
        assert dlg._mode_desc.text().startswith(_MODE_DESC)
    finally:
        dlg.close()
        dlg.deleteLater()


# -- Task D: Speed/Quality recommendation line follows the base explanation --


def _rec_dialog(tmp_path, monkeypatch, verdict):
    from vrcc.gui import settings_pages

    monkeypatch.setattr(
        settings_pages.recommend, "recommended_profile", lambda *a, **k: verdict
    )
    return _dialog(tmp_path, "small")


def test_mode_desc_appends_quality_recommendation(qapp, tmp_path, monkeypatch):
    from vrcc.gui import settings_pages

    dlg, _ = _rec_dialog(tmp_path, monkeypatch, "quality")
    try:
        dlg._update_mode_for_model()
        text = dlg._mode_desc.text()
        assert text.startswith(_MODE_DESC)
        assert settings_pages._MODE_RECOMMEND_QUALITY in text
    finally:
        dlg.close()
        dlg.deleteLater()


def test_mode_desc_appends_speed_recommendation(qapp, tmp_path, monkeypatch):
    from vrcc.gui import settings_pages

    dlg, _ = _rec_dialog(tmp_path, monkeypatch, "latency")
    try:
        dlg._update_mode_for_model()
        assert settings_pages._MODE_RECOMMEND_SPEED in dlg._mode_desc.text()
    finally:
        dlg.close()
        dlg.deleteLater()


def test_mode_desc_has_no_recommendation_when_none(qapp, tmp_path, monkeypatch):
    dlg, _ = _rec_dialog(tmp_path, monkeypatch, None)
    try:
        dlg._update_mode_for_model()
        assert dlg._mode_desc.text() == _MODE_DESC
    finally:
        dlg.close()
        dlg.deleteLater()
