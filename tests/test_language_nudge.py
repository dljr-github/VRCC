"""Offscreen GUI tests for the spoken-language / voice-model interaction.

Both language combos (Settings and the main window) grey the spoken languages
the active voice model cannot transcribe, so an unsupported language can't be
picked from the popup. The main window additionally keeps the model nudge: a
language set programmatically fires it at once, and a STORED language the
model cannot serve fires it once after construction/reload (queued via a
zero-delay shot), remembering a declined (model, language) pair so reloads do
not nag. Settings no longer nudges -- its greying makes the prompt
unreachable -- so ``maybe_switch_model_for_language`` is gone.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QMessageBox

from vrcc.core.bus import EventBus
from vrcc.core.config import ConfigStore, default_paths
from vrcc.gui import model_prompts
from vrcc.gui.bridge import BusBridge


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


class _FakeDM:
    """Presence checks only, like the real DownloadManager's is_*_downloaded."""

    def __init__(self, whisper=(), mt=()):
        self._w, self._m = set(whisper), set(mt)

    def is_whisper_downloaded(self, mid):
        return mid in self._w

    def is_mt_downloaded(self, spec):
        return spec.id in self._m


def _capture_question(monkeypatch, answer):
    asked: list[str] = []

    def question(parent, title, text, *args, **kwargs):
        asked.append(text)
        return answer

    monkeypatch.setattr(QMessageBox, "question", question)
    return asked


def _lang_enabled(combo, text):
    idx = combo.findText(text)
    assert idx >= 0, text
    return combo.model().item(idx).isEnabled()


# -- dead code removed --------------------------------------------------------


def test_settings_nudge_helper_is_gone():
    # Settings greys unsupported languages instead of nudging, so the settings
    # nudge entry point no longer exists.
    assert not hasattr(model_prompts, "maybe_switch_model_for_language")


# -- main window --------------------------------------------------------------


class _Pipeline:
    captioning_enabled = False

    def set_captioning(self, value):
        pass


def _window(tmp_path, downloaded, model_id, source=None):
    from vrcc.gui.main_window import MainWindow

    store = ConfigStore(tmp_path / "config.json")
    store.config.stt.model = model_id
    store.config.stt.device = "cpu"  # pins tier_for_config, machine-independent
    if source is not None:
        store.config.stt.source_language = source
    bridge = BusBridge(EventBus())
    swaps: list[str] = []
    window = MainWindow(
        bridge,
        store,
        _Pipeline(),
        lambda: None,
        lambda: None,
        download_manager=_FakeDM(whisper=downloaded),
        on_model_change=swaps.append,
    )
    return window, store, bridge, swaps


def _fire_scheduled_nudge(qapp):
    # The stored-language nudge queues a zero-delay singleShot; spin the event
    # loop so it fires (a bare construct/reload never shows the dialog).
    QTest.qWait(20)


def test_main_window_greys_source_languages_for_active_model(qapp, tmp_path):
    window, store, bridge, swaps = _window(
        tmp_path, {"parakeet-tdt-0.6b-v3"}, "parakeet-tdt-0.6b-v3"
    )
    try:
        src = window._source_combo
        assert _lang_enabled(src, "French")        # inside Parakeet's set
        assert not _lang_enabled(src, "Japanese")  # outside it
        assert _lang_enabled(src, "auto")          # Parakeet self-detects
    finally:
        window.close()
        window.deleteLater()
        bridge.detach()


def test_main_window_regreys_source_on_reload(qapp, tmp_path):
    # A model change made in Settings reaches the window via reload_from_config;
    # the spoken-language greying must re-run against the current model.
    window, store, bridge, swaps = _window(
        tmp_path, {"distil-small.en", "small"}, "distil-small.en"
    )
    try:
        src = window._source_combo
        assert not _lang_enabled(src, "Japanese")  # english-only
        assert not _lang_enabled(src, "auto")      # cannot self-detect

        store.config.stt.model = "small"
        window.reload_from_config()
        assert _lang_enabled(src, "Japanese")
        assert _lang_enabled(src, "auto")
    finally:
        window.close()
        window.deleteLater()
        bridge.detach()


def test_main_window_nudge_yes_switches_and_hotswaps(qapp, tmp_path, monkeypatch):
    # Greying blocks a popup pick, but a language set programmatically (or from
    # config) still reaches the nudge; on Yes it swaps to a compatible model.
    window, store, bridge, swaps = _window(
        tmp_path, {"parakeet-tdt-0.6b-v3", "small"}, "parakeet-tdt-0.6b-v3"
    )
    try:
        asked = _capture_question(monkeypatch, QMessageBox.StandardButton.Yes)
        window._source_combo.setCurrentText("Japanese")
        assert len(asked) == 1
        assert store.config.stt.model == "small"
        assert swaps == ["stt"]
    finally:
        window.close()
        window.deleteLater()
        bridge.detach()


def test_main_window_nudge_no_leaves_everything(qapp, tmp_path, monkeypatch):
    window, store, bridge, swaps = _window(
        tmp_path, {"parakeet-tdt-0.6b-v3", "small"}, "parakeet-tdt-0.6b-v3"
    )
    try:
        asked = _capture_question(monkeypatch, QMessageBox.StandardButton.No)
        window._source_combo.setCurrentText("Japanese")
        assert len(asked) == 1
        assert store.config.stt.model == "parakeet-tdt-0.6b-v3"
        assert store.config.stt.source_language == "Japanese"
        assert swaps == []
    finally:
        window.close()
        window.deleteLater()
        bridge.detach()


# -- stored-language nudge at load/reload --------------------------------------


def test_stored_unsupported_language_prompts_once_at_load(qapp, tmp_path, monkeypatch):
    # A config whose stored language the active model cannot transcribe never
    # passes through _on_source_changed (loads run under the _loading guard),
    # so the load path must fire the nudge itself -- exactly once, even when a
    # reload lands while the shot is still queued.
    asked = _capture_question(monkeypatch, QMessageBox.StandardButton.No)
    window, store, bridge, swaps = _window(
        tmp_path, {"parakeet-tdt-0.6b-v3", "small"}, "parakeet-tdt-0.6b-v3",
        source="Japanese",
    )
    try:
        assert asked == []  # queued, never shown mid-construction
        window.reload_from_config()  # re-schedules; the pending flag must coalesce
        _fire_scheduled_nudge(qapp)
        assert len(asked) == 1
    finally:
        window.close()
        window.deleteLater()
        bridge.detach()


def test_declined_stored_nudge_does_not_reprompt_on_reload(qapp, tmp_path, monkeypatch):
    asked = _capture_question(monkeypatch, QMessageBox.StandardButton.No)
    window, store, bridge, swaps = _window(
        tmp_path, {"parakeet-tdt-0.6b-v3", "small"}, "parakeet-tdt-0.6b-v3",
        source="Japanese",
    )
    try:
        _fire_scheduled_nudge(qapp)
        assert len(asked) == 1
        assert store.config.stt.model == "parakeet-tdt-0.6b-v3"
        assert swaps == []
        window.reload_from_config()
        _fire_scheduled_nudge(qapp)
        assert len(asked) == 1  # same (model, language) mismatch: stay quiet
    finally:
        window.close()
        window.deleteLater()
        bridge.detach()


def test_accepted_stored_nudge_routes_through_model_change(qapp, tmp_path, monkeypatch):
    asked = _capture_question(monkeypatch, QMessageBox.StandardButton.Yes)
    window, store, bridge, swaps = _window(
        tmp_path, {"parakeet-tdt-0.6b-v3", "small"}, "parakeet-tdt-0.6b-v3",
        source="Japanese",
    )
    try:
        _fire_scheduled_nudge(qapp)
        assert len(asked) == 1
        assert store.config.stt.model == "small"
        assert swaps == ["stt"]
        # The accepted model covers Japanese, so the greying re-enables it.
        assert _lang_enabled(window._source_combo, "Japanese")
    finally:
        window.close()
        window.deleteLater()
        bridge.detach()


def test_supported_stored_language_never_prompts(qapp, tmp_path, monkeypatch):
    asked = _capture_question(monkeypatch, QMessageBox.StandardButton.Yes)
    window, store, bridge, swaps = _window(
        tmp_path, {"parakeet-tdt-0.6b-v3", "small"}, "parakeet-tdt-0.6b-v3",
        source="French",  # inside Parakeet's set
    )
    try:
        _fire_scheduled_nudge(qapp)
        assert asked == []
        assert swaps == []
    finally:
        window.close()
        window.deleteLater()
        bridge.detach()
