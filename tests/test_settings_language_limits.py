"""Offscreen GUI tests for the Settings language greying, both directions:
models with a language restriction (Parakeet's European set, the distil
English-only pair) grey out when the spoken language falls outside their set,
AND the spoken-language entries the active voice model cannot transcribe grey
out. The two directions must never deadlock: switching the model re-enables
the languages.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication

from vrcc.core.config import ConfigStore, default_paths
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
    idx = combo.findText(text)
    assert idx >= 0, text
    return combo.model().item(idx).isEnabled()


def test_language_limited_models_grey_with_source_language(qapp, tmp_path):
    store = _store(tmp_path)
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

        # auto: models that detect the language within their set stay enabled
        # (Parakeet); models that can't detect at all grey out (distil would
        # force English).
        dlg._source_combo.setCurrentText("auto")
        assert _item_enabled(combo, "parakeet-tdt-0.6b-v3")
        assert not _item_enabled(combo, "distil-small.en")
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
    dlg = SettingsDialog(store)  # headless: source greying reads cfg.stt.model
    try:
        src = dlg._source_combo
        assert _lang_enabled(src, "French")        # inside Parakeet's set
        assert not _lang_enabled(src, "Japanese")  # outside it
        assert _lang_enabled(src, "auto")          # Parakeet self-detects
        # The disabled entry carries an explanatory tooltip naming the model.
        item = src.model().item(src.findText("Japanese"))
        assert item.toolTip().strip()
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
