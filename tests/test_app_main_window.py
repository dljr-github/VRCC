"""Offscreen Qt smoke tests for ``MainWindow`` wiring: profile toggle,
translation-active gating, send/capture status, and config reload.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path

import pytest

from vrcc.core.bus import EventBus
from vrcc.core.config import ConfigStore, default_paths


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _paths(tmp_path: Path):
    return default_paths(portable=True, app_dir=tmp_path)


def _store(tmp_path: Path) -> ConfigStore:
    return ConfigStore(_paths(tmp_path).config_file)


def _bridge():
    from vrcc.gui.bridge import BusBridge

    return BusBridge(EventBus())


class _FakePipeline:
    """Minimal pipeline surface MainWindow needs."""

    def __init__(self, accept_typed: bool = True) -> None:
        self.captioning_enabled = True
        self.typed: list[str] = []
        self._accept_typed = accept_typed

    def submit_typed(self, text: str) -> bool:
        self.typed.append(text)
        return self._accept_typed

    def set_captioning(self, enabled: bool) -> None:
        self.captioning_enabled = bool(enabled)


def _main_window(store, mt_available: bool = True):
    from vrcc.gui.main_window import MainWindow

    bridge = _bridge()
    window = MainWindow(
        bridge,
        store,
        _FakePipeline(),
        on_open_settings=lambda: None,
        on_open_models=lambda: None,
        mt_available=mt_available,
    )
    return window, bridge


def test_main_window_profile_toggle_applies_kwargs_bundle(qapp, tmp_path):
    # The Quality/Speed profile control moved off the main window into Settings
    # (Task 5); its handler stays on MainWindow, so drive it directly to keep
    # the kwargs-bundle behavior covered until the Settings control is wired.
    store = _store(tmp_path)
    window, bridge = _main_window(store)
    try:
        cfg = store.config
        assert cfg.stt.beam_size == 1  # latency default

        window._on_profile_toggled(True)  # Quality mode ON
        assert cfg.stt.beam_size == 5
        assert cfg.translate.beam_size == 3
        assert cfg.vad.speculative_silence_ms == 450
        assert cfg.vad.finalize_silence_ms == 800
        assert cfg.gui.profile == "quality"

        window._on_profile_toggled(False)  # back to Latency
        assert cfg.stt.beam_size == 1
        assert cfg.translate.beam_size == 1
        assert cfg.vad.speculative_silence_ms == 350
        assert cfg.vad.finalize_silence_ms == 600
        assert cfg.gui.profile == "latency"
    finally:
        window.close()
        window.deleteLater()
        bridge.detach()


def test_main_window_translation_active_reads_live_config_not_ctor_snapshot(qapp, tmp_path):
    # MT engines hot-swap in mid-session now, so ``mt_available`` at
    # construction must no longer gate `_translate_active` -- it reads live
    # config only. Constructing with mt_available=False (as if no MT engine
    # existed at launch) must not suppress translation once config says it's
    # on, otherwise a row would wrongly skip the "translating…" interim state.
    store = _store(tmp_path)
    store.config.translate.enabled = True
    window, bridge = _main_window(store, mt_available=False)
    try:
        assert window._translate_active() is True
    finally:
        window.close()
        window.deleteLater()
        bridge.detach()


def test_main_window_translation_inactive_when_toggle_off(qapp, tmp_path):
    store = _store(tmp_path)
    store.config.translate.enabled = False
    window, bridge = _main_window(store, mt_available=True)
    try:
        assert window._translate_active() is False
    finally:
        window.close()
        window.deleteLater()
        bridge.detach()


def test_send_clears_input_only_when_accepted(qapp, tmp_path):
    """Regression: the typed message must survive a refused send (pipeline not
    running) rather than being cleared and lost."""
    from vrcc.gui.main_window import MainWindow

    store = _store(tmp_path)
    for accept, expected in [(False, "hello"), (True, "")]:
        bridge = _bridge()
        window = MainWindow(
            bridge, store, _FakePipeline(accept_typed=accept),
            on_open_settings=lambda: None, on_open_models=lambda: None,
        )
        try:
            window._text_input.setText("hello")
            window._on_send_clicked()
            assert window._text_input.text() == expected
        finally:
            window.close()
            window.deleteLater()
            bridge.detach()


def test_capture_status_indicator_reflects_state(qapp, tmp_path):
    store = _store(tmp_path)
    window, bridge = _main_window(store)
    try:
        window.set_capture_status(True)
        assert "Listening" in window._capture_label.text()
        window.set_capture_status(False, "microphone unavailable")
        assert "Not listening" in window._capture_label.text()
        assert "microphone unavailable" in window._capture_label.text()
    finally:
        window.close()
        window.deleteLater()
        bridge.detach()


def test_captioning_toggle_pauses_without_closing_and_shows_paused(qapp, tmp_path):
    store = _store(tmp_path)
    window, bridge = _main_window(store)
    try:
        window.set_capture_status(True)  # pipeline running
        assert "Listening" in window._capture_label.text()

        window._captioning_btn.setChecked(False)  # pause captioning live
        assert window._pipeline.captioning_enabled is False
        assert "Paused" in window._capture_label.text()

        window._captioning_btn.setChecked(True)  # resume
        assert window._pipeline.captioning_enabled is True
        assert "Listening" in window._capture_label.text()
    finally:
        window.close()
        window.deleteLater()
        bridge.detach()


def test_chatbox_sent_marks_caption_row_sent(qapp, tmp_path):
    from types import SimpleNamespace

    store = _store(tmp_path)
    window, bridge = _main_window(store)
    try:
        window._on_phrase_recognized(
            SimpleNamespace(utterance_id=7, text="hello")
        )
        assert window._caption_model.rows()[0].status != "sent"
        window._on_chatbox_sent(
            SimpleNamespace(utterance_id=7, truncated=False)
        )
        assert window._caption_model.rows()[0].status == "sent"
    finally:
        window.close()
        window.deleteLater()
        bridge.detach()


def test_vrchat_status_chip_reflects_detection(qapp, tmp_path):
    from types import SimpleNamespace

    store = _store(tmp_path)
    window, bridge = _main_window(store)
    try:
        window._on_vrchat_detected(SimpleNamespace(detected=True))
        assert "connected" in window._vrchat_label.text()
        window._on_vrchat_detected(SimpleNamespace(detected=False))
        assert "not detected" in window._vrchat_label.text().lower()
    finally:
        window.close()
        window.deleteLater()
        bridge.detach()


def test_chatbox_sent_truncated_marks_row_truncated(qapp, tmp_path):
    from types import SimpleNamespace

    store = _store(tmp_path)
    window, bridge = _main_window(store)
    try:
        window._on_phrase_recognized(SimpleNamespace(utterance_id=9, text="hi"))
        window._on_chatbox_sent(SimpleNamespace(utterance_id=9, truncated=True))
        assert window._caption_model.rows()[0].status == "truncated"
    finally:
        window.close()
        window.deleteLater()
        bridge.detach()


def test_changing_source_drops_a_target_equal_to_it(qapp, tmp_path):
    store = _store(tmp_path)
    store.config.stt.source_language = "English"
    store.config.translate.targets = ["Japanese"]
    window, bridge = _main_window(store)
    try:
        # Point the source at the existing target: it must be dropped so we
        # don't translate a language into itself (double-send).
        window._source_combo.setCurrentText("Japanese")
        assert "Japanese" not in store.config.translate.targets
    finally:
        window.close()
        window.deleteLater()
        bridge.detach()


def test_reload_from_config_resyncs_toolbar(qapp, tmp_path):
    store = _store(tmp_path)
    window, bridge = _main_window(store)
    try:
        # Simulate the Settings dialog editing a shared field, then closing.
        store.config.stt.source_language = "auto"
        window.reload_from_config()
        assert window._source_combo.currentText() == "auto"
    finally:
        window.close()
        window.deleteLater()
        bridge.detach()


def test_fallback_cpu_engine_state_flashes_status(qapp, tmp_path):
    from types import SimpleNamespace

    store = _store(tmp_path)
    window, bridge = _main_window(store)
    try:
        window._on_engine_state(
            SimpleNamespace(engine="mt", state="fallback_cpu", detail="")
        )
        assert "CPU" in window.statusBar().currentMessage()
    finally:
        window.close()
        window.deleteLater()
        bridge.detach()
