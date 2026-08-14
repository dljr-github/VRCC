"""Offscreen GUI tests for two Settings page states: the Translation page
greys out with the feature it configures, and the interface-language picker
says when the new language will actually show up.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication

from vrcc.core.config import ConfigStore, default_paths
from vrcc.gui.settings import SettingsDialog
from vrcc.translate.registry import MT_MODELS


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


class _FakeDM:
    def __init__(self, whisper=(), mt=()):
        self._w, self._m = set(whisper), set(mt)

    def is_whisper_downloaded(self, mid):
        return mid in self._w

    def is_mt_downloaded(self, spec):
        return spec.id in self._m


def _store(tmp_path):
    return ConfigStore(default_paths(portable=True, app_dir=tmp_path).config_file)


def test_translation_page_greys_out_while_translation_is_off(qapp, tmp_path):
    store = _store(tmp_path)
    store.config.translate.enabled = False
    dlg = SettingsDialog(store)
    try:
        assert not dlg._translate_model_combo.isEnabled()
        assert not dlg._mt_beam_spin.isEnabled()
        assert not dlg._mt_rep_spin.isEnabled()
        assert not dlg._mt_norepeat_spin.isEnabled()
    finally:
        dlg.close()
        dlg.deleteLater()


def test_translation_page_follows_the_simple_tab_toggle(qapp, tmp_path):
    store = _store(tmp_path)
    store.config.translate.enabled = False
    dlg = SettingsDialog(store)
    try:
        dlg._translate_check.setChecked(True)
        assert dlg._translate_model_combo.isEnabled()
        assert dlg._mt_beam_spin.isEnabled()

        dlg._translate_check.setChecked(False)
        assert not dlg._translate_model_combo.isEnabled()
        assert not dlg._mt_beam_spin.isEnabled()
    finally:
        dlg.close()
        dlg.deleteLater()


def test_model_combo_stays_disabled_when_nothing_is_downloaded(qapp, tmp_path):
    # Turning translation on must not offer a combo that has nothing in it.
    store = _store(tmp_path)
    store.config.translate.enabled = False
    dlg = SettingsDialog(store, download_manager=_FakeDM(mt=set()))
    try:
        dlg._translate_check.setChecked(True)
        assert dlg._translate_model_combo.count() == 0
        assert not dlg._translate_model_combo.isEnabled()
        assert dlg._mt_beam_spin.isEnabled()  # the tuning group still opens
    finally:
        dlg.close()
        dlg.deleteLater()


def test_downloaded_model_combo_reenables_with_the_feature(qapp, tmp_path):
    ids = list(MT_MODELS)
    store = _store(tmp_path)
    store.config.translate.enabled = False
    store.config.translate.model = ids[0]
    dlg = SettingsDialog(store, download_manager=_FakeDM(mt={ids[0]}))
    try:
        assert not dlg._translate_model_combo.isEnabled()
        dlg._translate_check.setChecked(True)
        assert dlg._translate_model_combo.isEnabled()
    finally:
        dlg.close()
        dlg.deleteLater()


def test_interface_language_says_when_it_applies(qapp, tmp_path):
    # tr() runs at construction, so the open dialog keeps the old language and
    # the picker looks broken without this.
    store = _store(tmp_path)
    dlg = SettingsDialog(store)
    try:
        hint = dlg._ui_language_hint.text().lower()
        assert "close" in hint
        assert "settings" in hint
        assert dlg._ui_language_hint.wordWrap() is True
    finally:
        dlg.close()
        dlg.deleteLater()
