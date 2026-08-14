"""The Models window's spoken-language picker.

Before this existed, the wizard was the only place that ever asked which
languages you speak, so anyone who skipped it (or who started speaking another
one) was stuck on the language-blind recommendation with nowhere to correct it.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from vrcc.core.bus import EventBus
from vrcc.core.config import ConfigStore, default_paths


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


class _FakeDM:
    def __init__(self, models_dir=None):
        self.downloaded = set()
        self.models_dir = models_dir

    def is_whisper_downloaded(self, mid):
        return mid in self.downloaded

    def is_mt_downloaded(self, spec):
        return spec.id in self.downloaded


def _dlg(tmp_path, spoken=None):
    from vrcc.gui.bridge import BusBridge
    from vrcc.gui.models_dialog import ModelsDialog

    store = ConfigStore(default_paths(portable=True, app_dir=tmp_path).config_file)
    store.config.stt.source_language = "auto"
    store.config.stt.spoken_languages = list(spoken or [])
    bridge = BusBridge(EventBus())
    dlg = ModelsDialog(_FakeDM(tmp_path / "models"), bridge, config_store=store)
    return dlg, store, bridge


def _tick(dlg, display, checked=True):
    picker = dlg._spoken_list
    for i in range(picker.count()):
        item = picker.item(i)
        if item.text() == display:
            item.setCheckState(
                Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked
            )
            return
    raise AssertionError(f"{display!r} not in the picker")


def test_picker_preticks_what_config_already_says(qapp, tmp_path):
    dlg, _store, bridge = _dlg(tmp_path, spoken=["Japanese"])
    try:
        from vrcc.gui import firstrun_languages

        assert firstrun_languages.checked_in(dlg._spoken_list) == ["Japanese"]
    finally:
        dlg.close()
        dlg.deleteLater()
        bridge.detach()


def test_ticking_a_language_persists_it(qapp, tmp_path):
    dlg, store, bridge = _dlg(tmp_path)
    try:
        _tick(dlg, "Japanese")
        assert store.config.stt.spoken_languages == ["Japanese"]
    finally:
        dlg.close()
        dlg.deleteLater()
        bridge.detach()


def test_ticking_a_language_moves_the_recommended_badge(qapp, tmp_path):
    # The whole point: the badge follows the answer instead of staying on the
    # language-blind pick the user could not otherwise escape. The two ids are
    # the benchmark-derived outcome, so a registry change should surface here
    # as a conscious diff rather than a silent reshuffle.
    dlg, store, bridge = _dlg(tmp_path)
    try:
        store.config.stt.device = "cpu"  # pin the tier so language drives the pick
        assert dlg._recommendation()[0] == "small"

        _tick(dlg, "Japanese")

        assert dlg._recommended_ids[0] == "small"
    finally:
        dlg.close()
        dlg.deleteLater()
        bridge.detach()


def test_badge_actually_rerenders_on_the_rows(qapp, tmp_path):
    # Recomputing _recommended_ids is not enough; the rows have to be told.
    dlg, store, bridge = _dlg(tmp_path)
    try:
        store.config.stt.device = "cpu"
        _tick(dlg, "Japanese")
        badged = [
            row.model_id for row in dlg._rows
            if row.kind == "whisper" and row._badge.isVisibleTo(dlg)
        ]
        assert badged == ["small"]
    finally:
        dlg.close()
        dlg.deleteLater()
        bridge.detach()


def test_single_tick_pins_the_source_language(qapp, tmp_path):
    dlg, store, bridge = _dlg(tmp_path)
    try:
        _tick(dlg, "Japanese")
        assert store.config.stt.source_language == "Japanese"
    finally:
        dlg.close()
        dlg.deleteLater()
        bridge.detach()


def test_unticking_everything_leaves_the_source_alone(qapp, tmp_path):
    # Declining to say is not the same as saying "auto", so the engines keep
    # the source they were running. The recommendation keeps following that
    # stored source (spoken_whisper_codes falls back to it), which is why
    # unticking is not a route back to the language-blind pick. The window says
    # which language that leaves in force (test_models_dialog_warnings), so an
    # empty picker no longer reads as "no language at all".
    dlg, store, bridge = _dlg(tmp_path, spoken=["Japanese"])
    try:
        store.config.stt.source_language = "Japanese"

        _tick(dlg, "Japanese", checked=False)

        assert store.config.stt.spoken_languages == []
        assert store.config.stt.source_language == "Japanese"
    finally:
        dlg.close()
        dlg.deleteLater()
        bridge.detach()


def test_dialog_without_a_config_store_has_no_picker(qapp, tmp_path):
    from vrcc.gui.bridge import BusBridge
    from vrcc.gui.models_dialog import ModelsDialog

    bridge = BusBridge(EventBus())
    dlg = ModelsDialog(_FakeDM(tmp_path / "models"), bridge)
    try:
        assert not hasattr(dlg, "_spoken_list")
    finally:
        dlg.close()
        dlg.deleteLater()
        bridge.detach()


def test_picker_never_sets_a_source_the_active_model_cannot_serve(qapp, tmp_path):
    # Settings and the main window grey out exactly these pairs, so a source
    # they refuse to offer must not be reachable from here either. This window
    # only badges a recommendation, it never installs one, so the active model
    # is still the English-only one when the tick lands.
    dlg, store, bridge = _dlg(tmp_path)
    try:
        store.config.stt.model = "distil-small.en"
        store.config.stt.source_language = "English"

        _tick(dlg, "Japanese")

        assert store.config.stt.spoken_languages == ["Japanese"]
        assert store.config.stt.source_language == "English"
    finally:
        dlg.close()
        dlg.deleteLater()
        bridge.detach()


def test_picker_does_not_go_auto_for_a_model_that_cannot_report(qapp, tmp_path):
    # Parakeet detects the language but tags every result "en". Judging "auto"
    # against the recommendation instead of the running model set it here for
    # a model that would mislabel the translator's source.
    dlg, store, bridge = _dlg(tmp_path)
    try:
        store.config.stt.model = "parakeet-tdt-0.6b-v3"
        store.config.stt.source_language = "German"
        store.config.translate.enabled = True

        _tick(dlg, "German")
        _tick(dlg, "French")

        assert store.config.stt.source_language == "German"
    finally:
        dlg.close()
        dlg.deleteLater()
        bridge.detach()
