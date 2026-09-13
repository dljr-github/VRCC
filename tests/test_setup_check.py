"""The setup check controller: does the panel auto-open on the right
condition, and does each row tick only on the real evidence setup_steps.py
requires (not a typed Send, not any engine's ready, not a live send toggle
gone stale). Lifecycle concerns (the poll timer, threading, teardown, the
window-rebuild path) are in test_setup_check_lifecycle.py, split out to stay
under the repo's 500-line-per-file cap; both share the fixtures and fakes
defined here.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path

import pytest
from PySide6.QtWidgets import QApplication, QMessageBox

from vrcc.core.bus import EventBus
from vrcc.core.config import ConfigStore, default_paths
from vrcc.core.events import (
    ChatboxSent,
    EngineStateChanged,
    MicLevel,
    PhraseRecognized,
    VrchatDetected,
)
from vrcc.gui import model_prompts
from vrcc.gui.bridge import BusBridge
from vrcc.gui.main_window import MainWindow
from vrcc.gui.setup_check import start


@pytest.fixture(scope="module")
def qapp():
    # QApplication, never a bare QGuiApplication: the process-wide singleton
    # cannot be upgraded once claimed, and a bare one poisons every widget
    # test that runs later in this process.
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture(autouse=True)
def no_language_nudge(monkeypatch):
    # schedule_language_nudge no-ops without a download manager (never passed
    # below), so this is belt and braces rather than load-bearing here. Any
    # test that pumps the event loop must not risk a queued modal.
    monkeypatch.setattr(model_prompts, "run_language_nudge", lambda window: None)


class _FakePipeline:
    """The one surface SetupCheck reads from a pipeline: the toggle Pipeline
    itself never publishes (setup_steps.py's own docstring)."""

    def __init__(self) -> None:
        self.captioning_enabled = False
        # MainWindow._translate_active() reads this whenever a bridged event
        # repaints the caption log (translate.enabled defaults True).
        self.mt_active = False

    def set_captioning(self, enabled: bool) -> None:
        self.captioning_enabled = bool(enabled)

    def submit_typed(self, text: str) -> bool:
        return True

    def mute_gated(self) -> bool:
        return False


class _FakeDetector:
    """Stands in for VrchatDetector: SetupCheck reads only `.detected`, and
    window_swap.py's own swap path calls `.republish()`."""

    def __init__(self, detected: bool = False) -> None:
        self.detected = detected

    def republish(self) -> None:
        pass


def _store(tmp_path: Path) -> ConfigStore:
    paths = default_paths(portable=True, app_dir=tmp_path)
    return ConfigStore(paths.config_file)


def _window(bridge: BusBridge, store: ConfigStore, pipeline) -> MainWindow:
    window = MainWindow(
        bridge, store, pipeline,
        on_open_settings=lambda: None,
        on_open_models=lambda: None,
    )
    # place_beside reads frameGeometry(), which under-reports on Windows
    # before a window is ever shown; both window and panel must be shown
    # first (setup_panel.py's own docstring), matching app.py's real order.
    window.show()
    return window


def _phrase(utterance_id: int, text: str = "hello"):
    return PhraseRecognized(
        utterance_id=utterance_id, text=text, language="en",
        avg_logprob=0.0, no_speech_prob=0.0,
    )


# -- auto-open -----------------------------------------------------------


def test_panel_opens_when_setup_check_not_done(qapp, tmp_path):
    store = _store(tmp_path)
    bus = EventBus()
    pipeline = _FakePipeline()
    bridge = BusBridge(bus)
    window = _window(bridge, store, pipeline)
    check = start(bus, store, window, _FakeDetector())
    try:
        assert check._panel.isVisible()
    finally:
        check.stop()
        window.close()
        window.deleteLater()
        bridge.detach()


def test_panel_stays_hidden_when_already_done(qapp, tmp_path):
    store = _store(tmp_path)
    store.config.gui.setup_check_done = True
    bus = EventBus()
    pipeline = _FakePipeline()
    bridge = BusBridge(bus)
    window = _window(bridge, store, pipeline)
    check = start(bus, store, window, _FakeDetector())
    try:
        assert not check._panel.isVisible()
    finally:
        check.stop()
        window.close()
        window.deleteLater()
        bridge.detach()


# -- per-row evidence, and its traps -------------------------------------


def test_mic_level_at_zero_rms_does_not_tick_the_row(qapp, tmp_path):
    store = _store(tmp_path)
    bus = EventBus()
    pipeline = _FakePipeline()
    bridge = BusBridge(bus)
    window = _window(bridge, store, pipeline)
    check = start(bus, store, window, _FakeDetector())
    try:
        bus.publish(MicLevel(rms=0.0, vad_prob=0.0))
        qapp.processEvents()
        assert check._facts.mic_seen is False

        bus.publish(MicLevel(rms=0.02, vad_prob=0.9))
        qapp.processEvents()
        assert check._facts.mic_seen is True
    finally:
        check.stop()
        window.close()
        window.deleteLater()
        bridge.detach()


def test_typed_send_does_not_satisfy_voice_or_chatbox_rows(qapp, tmp_path):
    # The home of the typed-text filter: _on_event's utterance_id > 0 check,
    # not the evaluator, which only ever sees the already-filtered fields.
    # pipeline_typed.py gives a typed Send a NEGATIVE utterance id, and a
    # user must not complete the check by typing rather than speaking, so
    # dropping that check flips both facts below to True.
    store = _store(tmp_path)
    bus = EventBus()
    pipeline = _FakePipeline()
    bridge = BusBridge(bus)
    window = _window(bridge, store, pipeline)
    check = start(bus, store, window, _FakeDetector())
    try:
        bus.publish(_phrase(-3))
        bus.publish(ChatboxSent(text="hi", utterance_id=-3))
        qapp.processEvents()
        assert check._facts.spoken_utterance is False
        assert check._facts.spoken_chatbox_send is False
        # Tracked but never read by the evaluator; a decoy that must stay inert.
        assert check._facts.any_chatbox_send is True

        bus.publish(_phrase(4))
        bus.publish(ChatboxSent(text="hi", utterance_id=4))
        qapp.processEvents()
        assert check._facts.spoken_utterance is True
        assert check._facts.spoken_chatbox_send is True
    finally:
        check.stop()
        window.close()
        window.deleteLater()
        bridge.detach()


def test_model_row_reads_only_the_stt_engine_state(qapp, tmp_path, monkeypatch):
    # EngineStateChanged carries the translator on the same signal as the
    # voice model; a row keyed on "any engine ready" would go green when
    # only the translator loaded.
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: QMessageBox.StandardButton.Ok)
    from vrcc.gui.setup_steps import evaluate

    store = _store(tmp_path)
    bus = EventBus()
    pipeline = _FakePipeline()
    bridge = BusBridge(bus)
    window = _window(bridge, store, pipeline)
    check = start(bus, store, window, _FakeDetector())
    try:
        bus.publish(EngineStateChanged("mt", "ready"))
        qapp.processEvents()
        assert evaluate(check._facts)["model"] == "pending"

        bus.publish(EngineStateChanged("stt", "failed", "boom"))
        qapp.processEvents()
        assert evaluate(check._facts)["model"] == "attention"

        bus.publish(EngineStateChanged("stt", "ready"))
        qapp.processEvents()
        assert evaluate(check._facts)["model"] == "pass"
    finally:
        check.stop()
        window.close()
        window.deleteLater()
        bridge.detach()


def test_vrchat_detected_tracks_the_latest_publish(qapp, tmp_path):
    store = _store(tmp_path)
    bus = EventBus()
    pipeline = _FakePipeline()
    bridge = BusBridge(bus)
    window = _window(bridge, store, pipeline)
    check = start(bus, store, window, _FakeDetector(detected=False))
    try:
        assert check._facts.vrchat_found is False
        bus.publish(VrchatDetected(True))
        qapp.processEvents()
        assert check._facts.vrchat_found is True
    finally:
        check.stop()
        window.close()
        window.deleteLater()
        bridge.detach()


# -- captioning and send_enabled: read live, no bus event ----------------


def test_captioning_is_read_from_the_live_pipeline(qapp, tmp_path):
    from vrcc.gui.setup_steps import evaluate

    store = _store(tmp_path)
    bus = EventBus()
    pipeline = _FakePipeline()
    bridge = BusBridge(bus)
    window = _window(bridge, store, pipeline)
    check = start(bus, store, window, _FakeDetector())
    try:
        assert evaluate(check._facts)["captioning"] == "attention"
        pipeline.set_captioning(True)
        check._recompute()
        assert evaluate(check._facts)["captioning"] == "pass"
    finally:
        check.stop()
        window.close()
        window.deleteLater()
        bridge.detach()


def test_chatbox_row_drops_out_of_required_when_sending_is_off(qapp, tmp_path, monkeypatch):
    store = _store(tmp_path)
    store.config.osc.send_to_vrchat = False
    monkeypatch.setattr(store, "save_soon", lambda: None)
    bus = EventBus()
    pipeline = _FakePipeline()
    bridge = BusBridge(bus)
    window = _window(bridge, store, pipeline)
    check = start(bus, store, window, _FakeDetector())
    try:
        pipeline.set_captioning(True)
        bus.publish(EngineStateChanged("stt", "ready"))
        bus.publish(_phrase(1))
        bus.publish(VrchatDetected(True))
        qapp.processEvents()
        check._recompute()

        assert check._facts.spoken_chatbox_send is False
        assert check._facts.heard_seen is False  # optional row, never gates either
        assert store.config.gui.setup_check_done is True
    finally:
        check.stop()
        window.close()
        window.deleteLater()
        bridge.detach()
