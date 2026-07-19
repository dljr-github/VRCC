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
        self.mute_gate = False
        self.typed: list[str] = []
        self._accept_typed = accept_typed

    def submit_typed(self, text: str) -> bool:
        self.typed.append(text)
        return self._accept_typed

    def set_captioning(self, enabled: bool) -> None:
        self.captioning_enabled = bool(enabled)

    def mute_gated(self) -> bool:
        return self.mute_gate


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
        assert cfg.vad.speculative_silence_ms == 350
        assert cfg.vad.finalize_silence_ms == 800
        assert cfg.gui.profile == "quality"

        window._on_profile_toggled(False)  # back to Latency
        assert cfg.stt.beam_size == 1
        assert cfg.translate.beam_size == 1
        assert cfg.vad.speculative_silence_ms == 250
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


def test_capture_label_names_the_mute_pause_instead_of_listening(qapp, tmp_path):
    # Reported: mute sync in pause mode with the user muted in VRChat dropped
    # every caption while the label stayed green "Listening". The label folds
    # the pipeline's mute gate in and repaints on every MuteChanged.
    from types import SimpleNamespace

    store = _store(tmp_path)
    window, bridge = _main_window(store)
    try:
        window.set_capture_status(True)
        assert "Listening" in window._capture_label.text()

        window._pipeline.mute_gate = True
        window._on_mute_changed(SimpleNamespace(muted=True))
        assert window._capture_label.text() == "Paused - following your VRChat mute"
        assert window._mute_chip.text() == "MUTED"

        window._pipeline.mute_gate = False
        window._on_mute_changed(SimpleNamespace(muted=False))
        assert "Listening" in window._capture_label.text()
    finally:
        window.close()
        window.deleteLater()
        bridge.detach()


def test_reload_from_config_repaints_the_mute_pause_label(qapp, tmp_path):
    # A Settings mode change moves the gate through the shared config object
    # with no bus event; reload_from_config (run when Settings closes) must
    # re-derive the label from the live gate.
    store = _store(tmp_path)
    window, bridge = _main_window(store)
    try:
        window.set_capture_status(True)
        window._pipeline.mute_gate = True
        window.reload_from_config()
        assert window._capture_label.text() == "Paused - following your VRChat mute"
    finally:
        window.close()
        window.deleteLater()
        bridge.detach()


def test_master_toggle_pause_outranks_the_mute_gate_label(qapp, tmp_path):
    # With captioning toggled off the pipeline gates on the toggle first, so
    # naming the mute would send the user to the wrong control.
    store = _store(tmp_path)
    window, bridge = _main_window(store)
    try:
        window.set_capture_status(True)
        window._pipeline.mute_gate = True
        window._captioning_btn.setChecked(False)
        assert window._capture_label.text() == "Paused - not listening"
    finally:
        window.close()
        window.deleteLater()
        bridge.detach()


def test_mute_chip_clears_on_unknown_state(qapp, tmp_path):
    # MuteSync.stop() publishes MuteChanged(None): the chip must hide rather
    # than keep the stopped session's MUTED/LIVE.
    from types import SimpleNamespace

    store = _store(tmp_path)
    window, bridge = _main_window(store)
    try:
        window._on_mute_changed(SimpleNamespace(muted=True))
        assert not window._mute_chip.isHidden()
        window._on_mute_changed(SimpleNamespace(muted=None))
        assert window._mute_chip.isHidden()
    finally:
        window.close()
        window.deleteLater()
        bridge.detach()


def _rebuild_env(tmp_path):
    """Bridge, detector, pipeline and window factory for driving
    app._swap_main_window (the UI-language change path) for real."""
    from vrcc.gui.bridge import BusBridge
    from vrcc.gui.main_window import MainWindow
    from vrcc.osc.vrchat_detect import VrchatDetector

    class _Zc:
        def close(self) -> None:
            pass

    class _Browser:
        def cancel(self) -> None:
            pass

    bus = EventBus()
    bridge = BusBridge(bus)
    detector = VrchatDetector(
        bus, zeroconf_factory=_Zc, browser_factory=lambda zc, st, listener: _Browser()
    )
    store = _store(tmp_path)
    pipeline = _FakePipeline()

    def make_window():
        return MainWindow(
            bridge, store, pipeline,
            on_open_settings=lambda: None, on_open_models=lambda: None,
        )

    return bridge, detector, pipeline, make_window


def test_rebuilt_window_is_repushed_vrchat_state_and_capture_ok(qapp, tmp_path):
    # The detector publishes VrchatDetected only on transitions, so the fresh
    # window is told the current state via republish(); the capture label
    # carries over from the old window, and paused-vs-listening re-derives
    # from the captioning toggle, so a merely paused run renders amber
    # Paused instead of red failure or gray Starting.
    from vrcc.app import _swap_main_window

    bridge, detector, pipeline, make_window = _rebuild_env(tmp_path)
    pipeline.captioning_enabled = False  # the user paused before the rebuild

    old = make_window()
    fresh = None
    try:
        detector.start()
        detector.add_service(
            None, "_oscjson._tcp.local.", "VRChat-Client-7._oscjson._tcp.local."
        )
        assert "connected" in old._vrchat_label.text()
        old.set_capture_status(True)  # the pipeline started and is healthy

        fresh = _swap_main_window(old, make_window, detector)

        assert "connected" in fresh._vrchat_label.text()
        assert fresh._capture_label.text() == "Paused - not listening"
    finally:
        for w in (old, fresh):
            if w is not None:
                w.close()
                w.deleteLater()
        bridge.detach()


def test_rebuilt_window_keeps_a_capture_failure_red(qapp, tmp_path):
    # A failed engine paints red with a reason, whether from a mid-session
    # swap (pipeline started) or before capture ever started; the rebuild
    # must carry that verbatim rather than repaint green "Listening" over a
    # pipeline that is not captioning, or reset to gray "Starting".
    from vrcc.app import _swap_main_window

    bridge, detector, _pipeline, make_window = _rebuild_env(tmp_path)

    old = make_window()
    fresh = None
    try:
        old.set_capture_status(False, "a model failed to load")

        fresh = _swap_main_window(old, make_window, detector)

        assert fresh._capture_label.text() == (
            "Not listening - a model failed to load"
        )
    finally:
        for w in (old, fresh):
            if w is not None:
                w.close()
                w.deleteLater()
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


def test_stt_ready_clears_the_loading_message_without_a_caption(qapp, tmp_path):
    # The empty-state text is chosen from _engine_states, and only a caption
    # event would otherwise redraw the feed: a ready engine must clear the
    # "getting the voice model ready" line by itself.
    from types import SimpleNamespace

    store = _store(tmp_path)
    window, bridge = _main_window(store)
    try:
        window._on_engine_state(
            SimpleNamespace(engine="stt", state="loading", detail="")
        )
        assert "ready" in window._log.toPlainText().lower()

        window._on_engine_state(
            SimpleNamespace(engine="stt", state="ready", detail="cpu:int8")
        )

        text = window._log.toPlainText()
        assert "Say something" in text
        assert "ready" not in text.lower()
    finally:
        window.close()
        window.deleteLater()
        bridge.detach()
