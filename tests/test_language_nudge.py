"""Offscreen GUI tests for the spoken-language / voice-model interaction.

Both language combos (Settings and the main window) grey the spoken languages
the active voice model cannot transcribe, so an unsupported language can't be
picked from the popup. The main window additionally keeps the model nudge: a
language set programmatically fires it at once, and a STORED language the
model cannot serve fires it once after construction/reload (queued via a
zero-delay shot), remembering a declined (model, language) pair so reloads do
not nag. A stored "auto" joins in only for the translation mislabel case: an
onnx-asr model detects the language but tags the result "en", so with
translation on the nudge offers a model that reports its detection. Settings
no longer nudges -- its greying makes the prompt unreachable -- so
``maybe_switch_model_for_language`` is gone.
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


def _window(tmp_path, downloaded, model_id, source=None, translate=None):
    from vrcc.gui.main_window import MainWindow

    store = ConfigStore(tmp_path / "config.json")
    store.config.stt.model = model_id
    store.config.stt.device = "cpu"  # pins tier_for_config, machine-independent
    if source is not None:
        store.config.stt.source_language = source
    if translate is not None:
        store.config.translate.enabled = translate
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
        # Translation defaults on and Parakeet cannot report what it
        # detected, so "auto" is out until translation is off.
        assert not _lang_enabled(src, "auto")
    finally:
        window.close()
        window.deleteLater()
        bridge.detach()


def test_main_window_enables_auto_for_onnx_model_without_translation(qapp, tmp_path):
    window, store, bridge, swaps = _window(
        tmp_path, {"parakeet-tdt-0.6b-v3"}, "parakeet-tdt-0.6b-v3", translate=False
    )
    try:
        assert _lang_enabled(window._source_combo, "auto")
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


# -- stored "auto" under a model that cannot report its detection --------------


def _auto_cfg(model="parakeet-tdt-0.6b-v3", translating=True):
    from vrcc.core.config import AppConfig

    cfg = AppConfig()
    cfg.stt.model = model
    cfg.stt.device = "cpu"  # pins tier_for_config, machine-independent
    cfg.stt.source_language = "auto"
    cfg.translate.enabled = translating
    return cfg


def test_unsupported_stored_language_auto_truth_table():
    # Parakeet detects but tags every auto result "en", so "auto" only fails
    # it while translation is on; small detects and reports; distil cannot
    # detect at all, translation or not.
    cases = [
        ("parakeet-tdt-0.6b-v3", True, True),
        ("parakeet-tdt-0.6b-v3", False, False),
        ("small", True, False),
        ("small", False, False),
        ("distil-small.en", True, True),
        ("distil-small.en", False, True),
    ]
    for model, translating, expected in cases:
        cfg = _auto_cfg(model=model, translating=translating)
        got = model_prompts.unsupported_stored_language(cfg)
        assert got is expected, (model, translating)


def test_propose_auto_switch_offers_language_reporting_model():
    dm = _FakeDM(whisper={"parakeet-tdt-0.6b-v3", "small"})
    assert model_prompts.propose_language_switch(_auto_cfg(), dm, "auto") == "small"


def test_propose_auto_switch_none_when_nothing_suitable_downloaded():
    dm = _FakeDM(whisper={"parakeet-tdt-0.6b-v3"})
    assert model_prompts.propose_language_switch(_auto_cfg(), dm, "auto") is None


def test_propose_auto_switch_none_when_translation_off():
    dm = _FakeDM(whisper={"parakeet-tdt-0.6b-v3", "small"})
    cfg = _auto_cfg(translating=False)
    assert model_prompts.propose_language_switch(cfg, dm, "auto") is None


def test_propose_auto_switch_none_for_distil():
    # distil cannot detect at all; the auto offer stays scoped to the
    # translation mislabel case, so a stored distil + auto keeps returning
    # nothing even with a better model downloaded.
    dm = _FakeDM(whisper={"distil-small.en", "small"})
    cfg = _auto_cfg(model="distil-small.en")
    assert model_prompts.propose_language_switch(cfg, dm, "auto") is None


def test_stored_auto_nudge_offers_reporting_model(qapp, tmp_path, monkeypatch):
    asked = _capture_question(monkeypatch, QMessageBox.StandardButton.Yes)
    window, store, bridge, swaps = _window(
        tmp_path, {"parakeet-tdt-0.6b-v3", "small"}, "parakeet-tdt-0.6b-v3",
        source="auto",
    )
    try:
        _fire_scheduled_nudge(qapp)
        assert len(asked) == 1
        assert "cannot tell the translator" in asked[0]
        assert "{" not in asked[0]  # placeholders formatted
        assert store.config.stt.model == "small"
        assert swaps == ["stt"]
        # The accepted model reports its detection, so "auto" re-enables.
        assert _lang_enabled(window._source_combo, "auto")
    finally:
        window.close()
        window.deleteLater()
        bridge.detach()


def test_declined_auto_nudge_not_reasked_on_reload(qapp, tmp_path, monkeypatch):
    asked = _capture_question(monkeypatch, QMessageBox.StandardButton.No)
    window, store, bridge, swaps = _window(
        tmp_path, {"parakeet-tdt-0.6b-v3", "small"}, "parakeet-tdt-0.6b-v3",
        source="auto",
    )
    try:
        _fire_scheduled_nudge(qapp)
        assert len(asked) == 1
        assert store.config.stt.model == "parakeet-tdt-0.6b-v3"
        assert swaps == []
        window.reload_from_config()
        _fire_scheduled_nudge(qapp)
        assert len(asked) == 1  # same (model, "auto") mismatch: stay quiet
    finally:
        window.close()
        window.deleteLater()
        bridge.detach()


def test_stored_auto_without_translation_never_prompts(qapp, tmp_path, monkeypatch):
    asked = _capture_question(monkeypatch, QMessageBox.StandardButton.Yes)
    window, store, bridge, swaps = _window(
        tmp_path, {"parakeet-tdt-0.6b-v3", "small"}, "parakeet-tdt-0.6b-v3",
        source="auto", translate=False,
    )
    try:
        _fire_scheduled_nudge(qapp)
        assert asked == []
        assert swaps == []
        assert _lang_enabled(window._source_combo, "auto")
    finally:
        window.close()
        window.deleteLater()
        bridge.detach()


def test_stored_auto_with_distil_stays_quiet(qapp, tmp_path, monkeypatch):
    # distil + auto is a mismatch (no detection) but nothing is offered; the
    # scope stays deliberate.
    asked = _capture_question(monkeypatch, QMessageBox.StandardButton.Yes)
    window, store, bridge, swaps = _window(
        tmp_path, {"distil-small.en", "small"}, "distil-small.en", source="auto",
    )
    try:
        _fire_scheduled_nudge(qapp)
        assert asked == []
        assert store.config.stt.model == "distil-small.en"
        assert swaps == []
    finally:
        window.close()
        window.deleteLater()
        bridge.detach()
