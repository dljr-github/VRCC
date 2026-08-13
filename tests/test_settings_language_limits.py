"""Offscreen GUI tests for the Settings language greying, both directions:
models with a language restriction (Parakeet's European set, the distil
English-only pair) grey out when the spoken language falls outside their set,
AND the spoken-language entries the active voice model cannot transcribe grey
out. While translation is on, "auto" and the onnx-asr models grey each other
too: the backend tags every auto result "en", which would hand the translator
the wrong source language. The directions must never deadlock: switching the
model (or turning translation off) re-enables the languages.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication

from vrcc.core.config import ConfigStore, default_paths
from vrcc.gui.widgets import set_combo_value
from vrcc.gui import settings as settings_mod
from vrcc.gui.settings import SettingsDialog


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


def _item_enabled(combo, model_id):
    idx = combo.findData(model_id)
    assert idx >= 0
    return combo.model().item(idx).isEnabled()


def _lang_enabled(combo, text):
    # By stored value: the auto entry renders translated, so findText misses it.
    idx = combo.findData(text)
    if idx < 0:
        idx = combo.findText(text)
    assert idx >= 0, text
    return combo.model().item(idx).isEnabled()


def test_language_limited_models_grey_with_source_language(qapp, tmp_path):
    store = _store(tmp_path)
    store.config.translate.enabled = False
    dlg = SettingsDialog(store)  # headless: all models offered
    try:
        combo = dlg._model_combo
        # English: everything is available.
        dlg._source_combo.setCurrentText("English")
        for mid in ("small", "distil-small.en", "parakeet-tdt-0.6b-v3"):
            assert _item_enabled(combo, mid), mid

        # Japanese: the European-set models and distil (English) grey out.
        dlg._source_combo.setCurrentText("Japanese")
        assert _item_enabled(combo, "small")
        assert not _item_enabled(combo, "distil-small.en")
        assert not _item_enabled(combo, "parakeet-tdt-0.6b-v3")

        # French: Parakeet supports it, distil still doesn't.
        dlg._source_combo.setCurrentText("French")
        assert _item_enabled(combo, "parakeet-tdt-0.6b-v3")
        assert not _item_enabled(combo, "distil-small.en")

        # auto without translation: models that detect the language within
        # their set stay enabled (Parakeet); models that can't detect at all
        # grey out (distil would force English).
        set_combo_value(dlg._source_combo, "auto")
        assert _item_enabled(combo, "parakeet-tdt-0.6b-v3")
        assert not _item_enabled(combo, "distil-small.en")
    finally:
        dlg.close()
        dlg.deleteLater()


def test_auto_source_greys_onnx_models_while_translating(qapp, tmp_path):
    # Parakeet detects the spoken language but the onnx-asr backend tags
    # every auto result "en", so with translation on the translator would be
    # told English regardless; the model is not offerable under "auto".
    store = _store(tmp_path)  # translation defaults on
    dlg = SettingsDialog(store)
    try:
        combo = dlg._model_combo
        set_combo_value(dlg._source_combo, "auto")
        assert not _item_enabled(combo, "parakeet-tdt-0.6b-v3")
        assert not _item_enabled(combo, "distil-small.en")
        assert _item_enabled(combo, "small")  # detects AND reports
    finally:
        dlg.close()
        dlg.deleteLater()


def test_language_limited_greying_survives_placeholder_removal(qapp, tmp_path):
    # A deleted-model placeholder shifts combo indices when it is removed;
    # the greying bookkeeping must shift with it.
    store = _store(tmp_path)
    store.config.stt.model = "medium"  # configured model NOT downloaded
    dm = _FakeDM(whisper={"small", "parakeet-tdt-0.6b-v3"})
    dlg = SettingsDialog(store, download_manager=dm)
    try:
        combo = dlg._model_combo
        assert combo.itemData(0) is None  # the placeholder
        combo.setCurrentIndex(combo.findData("small"))  # real pick removes it
        assert combo.findData(None) < 0

        dlg._source_combo.setCurrentText("Japanese")
        assert _item_enabled(combo, "small")
        assert not _item_enabled(combo, "parakeet-tdt-0.6b-v3")
    finally:
        dlg.close()
        dlg.deleteLater()


# -- reverse direction: the spoken-language combo greys against the model -----


def test_source_languages_grey_for_european_model(qapp, tmp_path):
    store = _store(tmp_path)
    store.config.stt.model = "parakeet-tdt-0.6b-v3"
    store.config.translate.enabled = False
    dlg = SettingsDialog(store)  # headless: source greying reads cfg.stt.model
    try:
        src = dlg._source_combo
        assert _lang_enabled(src, "French")        # inside Parakeet's set
        assert not _lang_enabled(src, "Japanese")  # outside it
        assert _lang_enabled(src, "auto")          # detects, and no translator to mislead
        # The disabled entry carries an explanatory tooltip naming the model.
        item = src.model().item(src.findData("Japanese"))
        assert item.toolTip().strip()
    finally:
        dlg.close()
        dlg.deleteLater()


def test_auto_entry_greys_for_onnx_model_while_translating(qapp, tmp_path):
    store = _store(tmp_path)  # translation defaults on
    store.config.stt.model = "parakeet-tdt-0.6b-v3"
    dlg = SettingsDialog(store)
    try:
        src = dlg._source_combo
        assert not _lang_enabled(src, "auto")
        # The tooltip explains the reporting gap and both ways out; it is not
        # the generic cannot-transcribe tip.
        tip = src.model().item(src.findData("auto")).toolTip()
        assert "cannot tell the translator" in tip
        assert "turn translation off" in tip
        assert _lang_enabled(src, "French")  # explicit picks stay open
    finally:
        dlg.close()
        dlg.deleteLater()


def test_translate_toggle_regreys_both_combos_live(qapp, tmp_path):
    store = _store(tmp_path)  # translation defaults on
    store.config.stt.model = "parakeet-tdt-0.6b-v3"
    store.config.stt.source_language = "auto"
    dlg = SettingsDialog(store)
    try:
        assert not _lang_enabled(dlg._source_combo, "auto")
        assert not _item_enabled(dlg._model_combo, "parakeet-tdt-0.6b-v3")

        dlg._translate_check.setChecked(False)
        assert store.config.translate.enabled is False
        assert _lang_enabled(dlg._source_combo, "auto")
        assert _item_enabled(dlg._model_combo, "parakeet-tdt-0.6b-v3")

        dlg._translate_check.setChecked(True)
        assert not _lang_enabled(dlg._source_combo, "auto")
        assert not _item_enabled(dlg._model_combo, "parakeet-tdt-0.6b-v3")
    finally:
        dlg.close()
        dlg.deleteLater()


def test_source_languages_grey_for_english_only_model(qapp, tmp_path):
    store = _store(tmp_path)
    store.config.stt.model = "distil-small.en"
    dlg = SettingsDialog(store)
    try:
        src = dlg._source_combo
        assert _lang_enabled(src, "English")
        assert not _lang_enabled(src, "French")
        assert not _lang_enabled(src, "Japanese")
        assert not _lang_enabled(src, "auto")  # cannot self-detect the language
    finally:
        dlg.close()
        dlg.deleteLater()


def test_unknown_model_id_restricts_no_language(qapp, tmp_path):
    store = _store(tmp_path)
    store.config.stt.model = "hand-edited-nonsense"
    dlg = SettingsDialog(store)
    try:
        src = dlg._source_combo
        for text in ("auto", "English", "Japanese", "French"):
            assert _lang_enabled(src, text), text
    finally:
        dlg.close()
        dlg.deleteLater()


def _item_tip(combo, model_id):
    idx = combo.findData(model_id)
    assert idx >= 0
    return combo.model().item(idx).toolTip()


def test_greyed_model_entries_say_why_and_how_out(qapp, tmp_path):
    # The mirror of the spoken-language greying, which explains every entry it
    # disables. A greyed model with no tooltip leaves the user guessing.
    store = _store(tmp_path)
    store.config.translate.enabled = False
    dlg = SettingsDialog(store)
    try:
        combo = dlg._model_combo
        dlg._source_combo.setCurrentText("Japanese")
        tip = _item_tip(combo, "parakeet-tdt-0.6b-v3")
        assert "Parakeet" in tip
        assert "Japanese" in tip
        assert "spoken language" in tip  # names the way out
        assert _item_tip(combo, "small") == ""  # an offered model explains nothing
    finally:
        dlg.close()
        dlg.deleteLater()


def test_auto_greying_tooltips_name_the_two_different_reasons(qapp, tmp_path):
    store = _store(tmp_path)  # translation defaults on
    store.config.stt.source_language = "auto"
    dlg = SettingsDialog(store)
    try:
        combo = dlg._model_combo
        set_combo_value(dlg._source_combo, "auto")
        # distil cannot detect the language at all...
        assert "cannot detect" in _item_tip(combo, "distil-small.en")
        # ...while Parakeet detects it but cannot tell the translator which.
        # That tooltip is borrowed verbatim from the spoken-language combo:
        # the ways out (a concrete language, translation off) are the same
        # whichever side the user is looking at.
        from vrcc.gui import model_prompts
        from vrcc.gui.model_labels import whisper_display_name

        assert _item_tip(combo, "parakeet-tdt-0.6b-v3") == model_prompts.tr(
            model_prompts.AUTO_LOCKED_TIP,
            name=whisper_display_name("parakeet-tdt-0.6b-v3"),
        )

        dlg._translate_check.setChecked(False)
        assert _item_tip(combo, "parakeet-tdt-0.6b-v3") == ""  # offered again
    finally:
        dlg.close()
        dlg.deleteLater()


def test_no_deadlock_switch_model_reenables_language(qapp, tmp_path, monkeypatch):
    # Parakeet can't transcribe Japanese, so Japanese is greyed while it is the
    # active model. Switching the voice model to one that can (small) re-enables
    # Japanese; the model combo then greys Parakeet against the new language.
    monkeypatch.setattr(settings_mod.model_fit, "vram_warning", lambda *a, **k: None)
    store = _store(tmp_path)
    store.config.stt.model = "parakeet-tdt-0.6b-v3"
    store.config.stt.source_language = "French"
    dm = _FakeDM(whisper={"parakeet-tdt-0.6b-v3", "small"})
    dlg = SettingsDialog(store, download_manager=dm)
    try:
        src = dlg._source_combo
        assert not _lang_enabled(src, "Japanese")

        dlg._model_combo.setCurrentIndex(dlg._model_combo.findData("small"))
        assert store.config.stt.model == "small"
        assert _lang_enabled(src, "Japanese")  # the switch broke the deadlock

        src.setCurrentText("Japanese")
        assert store.config.stt.source_language == "Japanese"
        assert not _item_enabled(dlg._model_combo, "parakeet-tdt-0.6b-v3")
        assert _item_enabled(dlg._model_combo, "small")
    finally:
        dlg.close()
        dlg.deleteLater()
