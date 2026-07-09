"""Offscreen GUI tests for the Settings voice-model greying: models with a
language restriction (Parakeet's European set, the distil English-only pair)
must grey out whenever the selected spoken language falls outside it.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication

from vrcc.core.config import ConfigStore, default_paths
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


def test_language_limited_models_grey_with_source_language(qapp, tmp_path):
    store = _store(tmp_path)
    dlg = SettingsDialog(store)  # headless: all models offered
    try:
        combo = dlg._model_combo
        # English: everything is available.
        dlg._source_combo.setCurrentText("English")
        for mid in ("small", "distil-small.en", "parakeet-tdt-0.6b-v3"):
            assert _item_enabled(combo, mid), mid

        # Japanese: Parakeet (European languages) and distil (English) grey out.
        dlg._source_combo.setCurrentText("Japanese")
        assert _item_enabled(combo, "small")
        assert not _item_enabled(combo, "distil-small.en")
        assert not _item_enabled(combo, "parakeet-tdt-0.6b-v3")

        # French: Parakeet supports it, distil still doesn't.
        dlg._source_combo.setCurrentText("French")
        assert _item_enabled(combo, "parakeet-tdt-0.6b-v3")
        assert not _item_enabled(combo, "distil-small.en")

        # auto: multilingual-but-limited models stay enabled (they detect
        # within their set); single-language distil models grey out.
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
