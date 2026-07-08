"""Offscreen GUI tests for the friendly Settings dialog: the Simple page,
the Speed/Quality mode profile bundle, the moved Send/Translate toggles, and
the re-homed microphone-sensitivity slider writing ``vad.threshold``.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication, QComboBox, QDoubleSpinBox, QLabel, QMessageBox

from vrcc.core.config import ConfigStore, default_paths
from vrcc.gui import settings as settings_mod
from vrcc.gui.settings import SettingsDialog, _RESTART_FIELDS
from vrcc.stt.registry import WHISPER_MODELS
from vrcc.translate.registry import MT_MODELS


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


class _FakeDM:
    """Minimal DownloadManager stand-in: only the presence checks the dialog
    calls to filter its model combos."""

    def __init__(self, whisper=(), mt=()):
        self._w, self._m = set(whisper), set(mt)

    def is_whisper_downloaded(self, mid):
        return mid in self._w

    def is_mt_downloaded(self, spec):
        return spec.id in self._m


def _store(tmp_path):
    return ConfigStore(default_paths(portable=True, app_dir=tmp_path).config_file)


def _dlg(tmp_path):
    store = ConfigStore(default_paths(portable=True, app_dir=tmp_path).config_file)
    return SettingsDialog(store), store


def test_first_tab_is_simple(qapp, tmp_path):
    dlg, _ = _dlg(tmp_path)
    try:
        assert dlg._tabs.tabText(0) == "Simple"
    finally:
        dlg.close()
        dlg.deleteLater()


def test_mode_control_applies_profile(qapp, tmp_path):
    dlg, store = _dlg(tmp_path)
    try:
        assert store.config.stt.beam_size == 1  # latency default
        dlg._mode.set_value("Quality")
        assert store.config.stt.beam_size == 5
        assert store.config.gui.profile == "quality"
    finally:
        dlg.close()
        dlg.deleteLater()


def test_send_and_translate_toggles_write_config(qapp, tmp_path):
    dlg, store = _dlg(tmp_path)
    try:
        dlg._send_check.setChecked(False)
        assert store.config.osc.send_to_vrchat is False
        dlg._translate_check.setChecked(False)
        assert store.config.translate.enabled is False
    finally:
        dlg.close()
        dlg.deleteLater()


def test_microphone_sensitivity_writes_threshold(qapp, tmp_path):
    dlg, store = _dlg(tmp_path)
    try:
        dlg._sensitivity.setValue(45)
        assert abs(store.config.vad.threshold - 0.45) < 1e-6
    finally:
        dlg.close()
        dlg.deleteLater()


def test_mode_initial_reflects_config_profile(qapp, tmp_path):
    from vrcc.gui.settings import SettingsDialog

    store = ConfigStore(default_paths(portable=True, app_dir=tmp_path).config_file)
    store.config.gui.profile = "quality"
    dlg = SettingsDialog(store)
    try:
        assert dlg._mode.value() == "Quality"
    finally:
        dlg.close()
        dlg.deleteLater()


def test_mode_control_has_explanation(qapp, tmp_path):
    """The Speed/Quality control must carry a plain-language tooltip and a
    visible one-line description (user explicitly requested this)."""
    dlg, _ = _dlg(tmp_path)
    try:
        assert dlg._mode.toolTip().strip()
        assert dlg._mode_desc.text().strip()
        # No jargon in the visible description.
        text = dlg._mode_desc.text().lower()
        for word in ("beam", "vad", "ctranslate", "compute type"):
            assert word not in text
    finally:
        dlg.close()
        dlg.deleteLater()


def test_translation_model_combo_shows_friendly_name(qapp, tmp_path):
    """The Translation model dropdown must display a friendly name (Task 7
    polish), not the raw model id, while still storing/binding the id."""
    dlg, store = _dlg(tmp_path)
    try:
        combo = dlg._translate_model_combo
        idx = combo.findData("nllb-600M-int8")
        assert idx >= 0
        assert combo.itemText(idx) != "nllb-600M-int8"
        assert "NLLB 600M" in combo.itemText(idx)

        combo.setCurrentIndex(idx)
        assert store.config.translate.model == "nllb-600M-int8"
    finally:
        dlg.close()
        dlg.deleteLater()


def test_advanced_controls_have_tooltips(qapp, tmp_path):
    """A sample of advanced controls must expose non-empty tooltips."""
    dlg, _ = _dlg(tmp_path)
    try:
        for attr in ("_stt_device_combo", "_stt_compute_combo"):
            widget = getattr(dlg, attr, None)
            assert widget is not None, attr
            assert widget.toolTip().strip(), attr
    finally:
        dlg.close()
        dlg.deleteLater()


# -- Task 3: downloaded-only pickers + live hotswap trigger ----------------


def test_voice_combo_lists_only_downloaded(qapp, tmp_path):
    ids = list(WHISPER_MODELS)
    store = _store(tmp_path)
    store.config.stt.model = ids[0]
    dm = _FakeDM(whisper={ids[0]})
    dlg = SettingsDialog(store, download_manager=dm)
    try:
        data = [dlg._model_combo.itemData(i) for i in range(dlg._model_combo.count())]
        assert data == [ids[0]]
    finally:
        dlg.deleteLater()


def test_model_change_calls_on_model_change(qapp, tmp_path):
    ids = list(WHISPER_MODELS)
    store = _store(tmp_path)
    store.config.stt.model = ids[0]
    dm = _FakeDM(whisper=set(ids[:2]))
    calls = []
    dlg = SettingsDialog(store, download_manager=dm, on_model_change=calls.append)
    try:
        dlg._model_combo.setCurrentIndex(1)  # a different downloaded model
        assert store.config.stt.model == ids[1]
        assert calls == ["stt"]
    finally:
        dlg.deleteLater()


def test_translate_toggle_calls_on_model_change(qapp, tmp_path):
    store = _store(tmp_path)
    store.config.translate.enabled = False
    calls = []
    dlg = SettingsDialog(store, on_model_change=calls.append)
    try:
        dlg._translate_check.setChecked(True)
        assert store.config.translate.enabled is True
        assert calls == ["mt"]
    finally:
        dlg.deleteLater()


def test_vram_warning_cancel_reverts_and_skips_hotswap(qapp, tmp_path, monkeypatch):
    ids = list(WHISPER_MODELS)
    store = _store(tmp_path)
    store.config.stt.model = ids[0]
    dm = _FakeDM(whisper=set(ids[:2]))
    calls = []
    dlg = SettingsDialog(store, download_manager=dm, on_model_change=calls.append)
    try:
        monkeypatch.setattr(
            settings_mod.model_fit, "vram_warning", lambda *a, **k: "too big"
        )
        monkeypatch.setattr(
            settings_mod.QMessageBox,
            "question",
            staticmethod(lambda *a, **k: QMessageBox.StandardButton.No),
        )
        dlg._model_combo.setCurrentIndex(1)  # user picks a bigger model, then says No
        assert store.config.stt.model == ids[0]
        assert calls == []
        assert dlg._model_combo.currentData() == ids[0]
    finally:
        dlg.deleteLater()


def test_empty_voice_combo_disabled_when_none_downloaded(qapp, tmp_path):
    store = _store(tmp_path)
    dm = _FakeDM(whisper=set())
    dlg = SettingsDialog(store, download_manager=dm)
    try:
        assert dlg._model_combo.count() == 0
        assert not dlg._model_combo.isEnabled()
    finally:
        dlg.deleteLater()


def test_model_fields_not_restart_gated():
    flat = set(_RESTART_FIELDS)
    assert ("stt", "model") not in flat
    assert ("translate", "model") not in flat
    assert ("translate", "enabled") not in flat


# -- Task 5: dialog fits a laptop screen, plain labels, presets, placeholder --


def test_dialog_min_height_fits_laptop_screen(qapp, tmp_path):
    """The tallest tab (Advanced) used to force the whole dialog past 1000px.
    Each page is now wrapped in a scroll area so it no longer drives the
    dialog's minimum size."""
    dlg, _ = _dlg(tmp_path)
    try:
        assert dlg.minimumSizeHint().height() <= 640
    finally:
        dlg.close()
        dlg.deleteLater()


def test_page_margins_are_24px_left_right(qapp, tmp_path):
    dlg, _ = _dlg(tmp_path)
    try:
        for i in range(dlg._tabs.count()):
            page = dlg._tabs.widget(i).widget()
            margins = page.layout().contentsMargins()
            assert margins.left() == 24
            assert margins.right() == 24
    finally:
        dlg.close()
        dlg.deleteLater()


def test_no_advanced_suffix_in_labels(qapp, tmp_path):
    dlg, _ = _dlg(tmp_path)
    try:
        for i in range(dlg._tabs.count()):
            page = dlg._tabs.widget(i).widget()
            for label in page.findChildren(QLabel):
                assert "(advanced)" not in label.text()
    finally:
        dlg.close()
        dlg.deleteLater()


def test_text_size_large_writes_1_2(qapp, tmp_path):
    dlg, store = _dlg(tmp_path)
    try:
        dlg._text_size.set_value("Large")
        assert store.config.gui.font_scale == 1.2
    finally:
        dlg.close()
        dlg.deleteLater()


def test_text_size_closest_match_for_odd_stored_value(qapp, tmp_path):
    """0.95 is equidistant from Small (0.9) and Normal (1.0); the mapping
    prefers Small (the first candidate at the smallest distance)."""
    store = _store(tmp_path)
    store.config.gui.font_scale = 0.95
    dlg = SettingsDialog(store)
    try:
        assert dlg._text_size.value() == "Small"
    finally:
        dlg.close()
        dlg.deleteLater()


def test_sensitivity_slider_has_low_high_anchors(qapp, tmp_path):
    dlg, _ = _dlg(tmp_path)
    try:
        assert dlg._sensitivity_low.text() == "Low"
        assert dlg._sensitivity_high.text() == "High"
    finally:
        dlg.close()
        dlg.deleteLater()


def test_noise_slider_has_anchors_and_muted_value_label(qapp, tmp_path):
    dlg, _ = _dlg(tmp_path)
    try:
        assert dlg._noise_low.text() == "Low"
        assert dlg._noise_high.text() == "High"
        assert "color" in dlg._noise_value_label.styleSheet()
    finally:
        dlg.close()
        dlg.deleteLater()


def test_voice_combo_shows_deleted_placeholder_when_configured_model_missing(qapp, tmp_path):
    ids = list(WHISPER_MODELS)
    store = _store(tmp_path)
    store.config.stt.model = ids[2]  # not among the downloaded ids below
    dm = _FakeDM(whisper=set(ids[:2]))
    dlg = SettingsDialog(store, download_manager=dm)
    try:
        combo = dlg._model_combo
        assert combo.currentIndex() == 0
        assert not combo.model().item(0).isEnabled()
        assert store.config.stt.model == ids[2]  # unchanged until a real pick

        # Picking a real model still behaves normally -- and retires the
        # placeholder from the item list (it has served its purpose).
        combo.setCurrentIndex(1)
        assert store.config.stt.model == ids[0]
        data = [combo.itemData(i) for i in range(combo.count())]
        assert None not in data  # placeholder gone
        assert combo.currentData() == ids[0]  # selection preserved
    finally:
        dlg.deleteLater()


def test_translation_combo_shows_deleted_placeholder_when_configured_model_missing(qapp, tmp_path):
    ids = list(MT_MODELS)
    store = _store(tmp_path)
    store.config.translate.model = ids[3]  # not among the downloaded ids below
    dm = _FakeDM(mt=set(ids[:2]))
    dlg = SettingsDialog(store, download_manager=dm)
    try:
        combo = dlg._translate_model_combo
        assert combo.currentIndex() == 0
        assert not combo.model().item(0).isEnabled()
        assert store.config.translate.model == ids[3]

        # A real pick writes config and retires the placeholder.
        combo.setCurrentIndex(1)
        assert store.config.translate.model == ids[0]
        assert None not in [combo.itemData(i) for i in range(combo.count())]
    finally:
        dlg.deleteLater()


def test_hint_style_scales_with_font_scale(qapp, tmp_path):
    # The 11px hint size must follow the text-size preset like caption_log
    # does, else Large renders hints smaller than body text.
    store = _store(tmp_path)
    store.config.gui.font_scale = 1.2
    dlg = SettingsDialog(store)
    try:
        assert f"font-size: {round(11 * 1.2)}px" in dlg._muted_style
    finally:
        dlg.close()
        dlg.deleteLater()


def test_light_theme_dialog_uses_no_hardcoded_dark_colors(qapp, tmp_path):
    """Hint labels and the restart/warning banners must be themed, so a
    light-theme dialog carries none of the old dark-only hex literals."""
    from PySide6.QtWidgets import QWidget

    store = _store(tmp_path)
    store.config.gui.theme = "light"
    dlg = SettingsDialog(store)
    try:
        for w in dlg.findChildren(QWidget):
            ss = w.styleSheet()
            assert "#98a2b3" not in ss  # dark muted
            assert "#fdf2e0" not in ss  # cream warn background
    finally:
        dlg.close()
        dlg.deleteLater()


def test_simple_tab_hides_original_words_option(qapp, tmp_path):
    store = _store(tmp_path)
    dlg = SettingsDialog(store)
    try:
        check = dlg._include_original_check
        assert check.isChecked() is True  # config default keeps originals shown
        check.setChecked(False)
        assert store.config.osc.include_original is False
        check.setChecked(True)
        assert store.config.osc.include_original is True
    finally:
        dlg.deleteLater()


# -- split-delay spin + friendly overflow labels ("Message pacing") --------

_VRCHAT_TAB_INDEX = 3


def test_split_delay_spin_exists_and_writes_config(qapp, tmp_path):
    dlg, store = _dlg(tmp_path)
    try:
        page = dlg._tabs.widget(_VRCHAT_TAB_INDEX).widget()
        spin = next(
            s
            for s in page.findChildren(QDoubleSpinBox)
            if s.toolTip()
            == "How long each part of a long caption stays visible before "
            "the next part replaces it."
        )
        spin.setValue(4.5)
        assert store.config.osc.split_delay_s == 4.5
    finally:
        dlg.close()
        dlg.deleteLater()


def test_overflow_combo_shows_friendly_labels_and_binds_raw_value(qapp, tmp_path):
    dlg, store = _dlg(tmp_path)
    try:
        page = dlg._tabs.widget(_VRCHAT_TAB_INDEX).widget()
        combo = next(
            c
            for c in page.findChildren(QComboBox)
            if [c.itemData(i) for i in range(c.count())] == ["split", "truncate", "send"]
        )
        assert [combo.itemText(i) for i in range(combo.count())] == [
            "Send in parts",
            "Shorten to fit",
            "Send full (may be cut off in VRChat)",
        ]
        assert combo.currentData() == store.config.osc.overflow  # default "split"

        combo.setCurrentIndex(1)  # "Shorten to fit" -> "truncate"
        assert store.config.osc.overflow == "truncate"
    finally:
        dlg.close()
        dlg.deleteLater()
