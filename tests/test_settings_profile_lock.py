"""Offscreen GUI tests for the Speed/Quality Mode greying: the
onnxruntime-backed voice models (Parakeet, SenseVoice) decode greedily, so the
profile's beam/temperature presets can't tune their captions -- the Mode
control must grey out with an explanatory tooltip naming the model (and its
visible description label must swap to the same explanation) while such a
model is active, and recover on a switch.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication

from vrcc.core.config import ConfigStore, default_paths
from vrcc.gui import settings as settings_mod
from vrcc.gui.settings import SettingsDialog
from vrcc.gui.model_labels import whisper_display_name
from vrcc.gui.settings_mode import _MODE_DESC, _MODE_LOCKED_TOOLTIP, _MODE_TOOLTIP


def _locked_text(model_id):
    return _MODE_LOCKED_TOOLTIP.format(name=whisper_display_name(model_id))


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _dialog(tmp_path, model_id):
    store = ConfigStore(default_paths(portable=True, app_dir=tmp_path).config_file)
    store.config.stt.model = model_id
    return SettingsDialog(store), store  # headless: all models offered


@pytest.mark.parametrize("model_id", ["parakeet-tdt-0.6b-v3", "sense-voice-small"])
def test_mode_disabled_with_tooltip_for_greedy_models(qapp, tmp_path, model_id):
    dlg, store = _dialog(tmp_path, model_id)
    try:
        assert not dlg._mode.isEnabled()
        assert dlg._mode.toolTip() == _locked_text(model_id)
        # The visible description must not advertise a Speed/Quality trade-off
        # the locked control can't deliver.
        assert dlg._mode_desc.text() == _locked_text(model_id)
        # The segments grey with the control, and the stored profile stays
        # put: its VAD parts still apply at the current position.
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
        assert dlg._mode.toolTip() == _locked_text("parakeet-tdt-0.6b-v3")
        assert dlg._mode_desc.text() == _locked_text("parakeet-tdt-0.6b-v3")
        assert store.config.gui.profile == "latency"  # position untouched

        combo.setCurrentIndex(combo.findData("small"))
        assert dlg._mode.isEnabled()
        assert dlg._mode.toolTip() == _MODE_TOOLTIP
        assert dlg._mode_desc.text().startswith(_MODE_DESC)
    finally:
        dlg.close()
        dlg.deleteLater()


# -- the description covers what a switch overwrites --------------------------


def _hand_tune(store):
    """Values matching neither bundle, so only a person could have set them."""
    store.config.vad.finalize_silence_ms = 1234
    store.config.vad.pre_roll_ms = 999
    store.config.stt.beam_size = 7


def test_mode_switch_asks_before_replacing_hand_tuned_fields(qapp, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QMessageBox

    dlg, store = _dialog(tmp_path, "small")
    try:
        _hand_tune(store)
        asked = []
        monkeypatch.setattr(
            QMessageBox, "question",
            lambda *a, **k: asked.append(a[2]) or QMessageBox.StandardButton.Yes,
        )

        dlg._mode.set_value("Quality")

        assert asked, "a hand-set value must not be replaced silently"
        # It names the controls the user tuned, under the labels they saw.
        assert "Wait before finishing a caption" in asked[0]
        assert "Search width" in asked[0]
        # Confirmed, so the bundle applies and the Advanced spins show it.
        assert store.config.vad.finalize_silence_ms == 800
        assert store.config.stt.beam_size == 5
        assert dlg._vad_spins["finalize_silence_ms"].value() == 800
    finally:
        dlg.close()
        dlg.deleteLater()


def test_declining_the_mode_switch_keeps_every_hand_tuned_value(qapp, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QMessageBox

    dlg, store = _dialog(tmp_path, "small")
    try:
        _hand_tune(store)
        monkeypatch.setattr(
            QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.No
        )

        dlg._mode.set_value("Quality")

        assert store.config.vad.finalize_silence_ms == 1234
        assert store.config.vad.pre_roll_ms == 999
        assert store.config.stt.beam_size == 7
        assert store.config.gui.profile == "latency"
        # The control goes back, or it would claim a mode that never applied.
        assert dlg._mode.value() == "Speed"
    finally:
        dlg.close()
        dlg.deleteLater()


def test_flipping_between_the_two_modes_never_asks(qapp, tmp_path, monkeypatch):
    """The reason the prompt can afford to be modal: values that came from a
    bundle are not hand-set, so someone comparing the modes never sees it."""
    from PySide6.QtWidgets import QMessageBox

    dlg, store = _dialog(tmp_path, "small")
    try:
        asked = []
        monkeypatch.setattr(
            QMessageBox, "question",
            lambda *a, **k: asked.append(1) or QMessageBox.StandardButton.Yes,
        )

        for value in ("Quality", "Speed", "Quality", "Speed"):
            dlg._mode.set_value(value)

        assert asked == []
        assert store.config.gui.profile == "latency"
    finally:
        dlg.close()
        dlg.deleteLater()


def test_mode_description_names_what_a_switch_rewrites(qapp, tmp_path):
    # Shown to everyone; the modal above fires only when something would
    # actually be lost.
    dlg, _ = _dialog(tmp_path, "small")
    try:
        text = dlg._mode_desc.text().lower()
        assert "advanced" in text
        assert "timing" in text
        assert "search width" in text
    finally:
        dlg.close()
        dlg.deleteLater()


# -- Task D: Speed/Quality recommendation line follows the base explanation --


def _rec_dialog(tmp_path, monkeypatch, verdict):
    from vrcc.gui import settings_mode

    monkeypatch.setattr(
        settings_mode.recommend, "recommended_profile", lambda *a, **k: verdict
    )
    return _dialog(tmp_path, "small")


def test_mode_desc_appends_quality_recommendation(qapp, tmp_path, monkeypatch):
    from vrcc.gui import settings_mode

    dlg, _ = _rec_dialog(tmp_path, monkeypatch, "quality")
    try:
        dlg._update_mode_for_model()
        text = dlg._mode_desc.text()
        assert text.startswith(_MODE_DESC)
        assert settings_mode._MODE_RECOMMEND_QUALITY in text
    finally:
        dlg.close()
        dlg.deleteLater()


def test_mode_desc_appends_speed_recommendation(qapp, tmp_path, monkeypatch):
    from vrcc.gui import settings_mode

    dlg, _ = _rec_dialog(tmp_path, monkeypatch, "latency")
    try:
        dlg._update_mode_for_model()
        assert settings_mode._MODE_RECOMMEND_SPEED in dlg._mode_desc.text()
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


def test_every_profile_field_actually_differs_between_the_modes():
    """A field the two bundles agree on is overwritten by a mode flip for no
    benefit: the user loses a hand-tuned value and the mode gains nothing.
    min_utterance_ms and max_utterance_s were both, until they were removed.
    Fields the mode has no opinion about belong to "Reset tuning to defaults".
    """
    from vrcc.core.config import PROFILES

    latency, quality = PROFILES["latency"], PROFILES["quality"]
    assert set(latency) == set(quality)
    for section in latency:
        assert set(latency[section]) == set(quality[section]), section
        for field, value in latency[section].items():
            assert value != quality[section][field], f"{section}.{field}"
