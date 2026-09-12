"""The setup check controller: does it track only real bus evidence, marshal
worker-thread events onto the GUI thread, persist completion exactly once,
and survive a window rebuild the same run() gives it.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import threading
import time
from pathlib import Path

import pytest
from PySide6.QtCore import QThread
from PySide6.QtWidgets import QApplication, QMessageBox

from vrcc.core.bus import EventBus
from vrcc.core.config import ConfigStore, default_paths
from vrcc.core.events import (
    ChatboxSent,
    EngineStateChanged,
    HeardLevel,
    MicLevel,
    PhraseRecognized,
    VrchatDetected,
)
from vrcc.gui import model_prompts
from vrcc.gui.bridge import BusBridge
from vrcc.gui.main_window import MainWindow
from vrcc.gui.setup_check import start
from vrcc.gui.window_swap import _swap_main_window


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


def _pump(app, done, timeout: float = 1.0) -> bool:
    """Process events until `done()` holds or `timeout` passes. A bare
    processEvents loop can outrun a queued cross-thread emit before Qt has
    posted it, so this needs real wall-clock time between polls."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if done():
            return True
        app.processEvents()
        time.sleep(0.001)
    return bool(done())


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


def test_mic_level_only_ticks_on_real_sound(qapp, tmp_path):
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
    # pipeline_typed.py gives a typed Send a NEGATIVE utterance id; a user
    # must not complete the check by typing rather than speaking.
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


def test_chatbox_row_drops_out_of_required_when_sending_is_off(qapp, tmp_path):
    store = _store(tmp_path)
    store.config.osc.send_to_vrchat = False
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


# -- completion persists exactly once -------------------------------------


def test_completion_sets_and_persists_the_flag_exactly_once(qapp, tmp_path, monkeypatch):
    store = _store(tmp_path)
    bus = EventBus()
    pipeline = _FakePipeline()
    bridge = BusBridge(bus)
    window = _window(bridge, store, pipeline)
    check = start(bus, store, window, _FakeDetector())
    calls = []
    monkeypatch.setattr(store, "save_soon", lambda: calls.append(1))
    try:
        assert store.config.gui.setup_check_done is False

        pipeline.set_captioning(True)
        bus.publish(EngineStateChanged("stt", "ready"))
        bus.publish(_phrase(1))
        bus.publish(VrchatDetected(True))
        bus.publish(ChatboxSent(text="hi", utterance_id=1))
        qapp.processEvents()
        check._recompute()

        assert store.config.gui.setup_check_done is True
        assert calls == [1]

        # A later event must not persist a second time.
        bus.publish(EngineStateChanged("stt", "ready"))
        qapp.processEvents()
        check._recompute()
        assert calls == [1]
    finally:
        check.stop()
        window.close()
        window.deleteLater()
        bridge.detach()


# -- threading: never touch a widget from a bus callback -------------------


def test_worker_thread_publishes_are_marshalled_to_the_gui_thread(qapp, tmp_path, monkeypatch):
    import vrcc.gui.setup_check as setup_check_mod

    store = _store(tmp_path)
    bus = EventBus()
    pipeline = _FakePipeline()
    bridge = BusBridge(bus)
    window = _window(bridge, store, pipeline)
    check = start(bus, store, window, _FakeDetector())

    seen_threads = []
    original_evaluate = setup_check_mod.evaluate

    def spy_evaluate(facts):
        seen_threads.append(QThread.currentThread())
        return original_evaluate(facts)

    monkeypatch.setattr(setup_check_mod, "evaluate", spy_evaluate)
    try:
        t = threading.Thread(target=lambda: bus.publish(MicLevel(rms=0.5, vad_prob=0.9)))
        t.start()
        t.join()
        assert _pump(qapp, lambda: len(seen_threads) > 0)
        assert seen_threads[0] is qapp.thread()
    finally:
        check.stop()
        window.close()
        window.deleteLater()
        bridge.detach()


# -- teardown --------------------------------------------------------------


def test_stop_unsubscribes_and_is_idempotent(qapp, tmp_path):
    store = _store(tmp_path)
    bus = EventBus()
    pipeline = _FakePipeline()
    bridge = BusBridge(bus)
    window = _window(bridge, store, pipeline)
    check = start(bus, store, window, _FakeDetector())
    try:
        check.stop()
        check.stop()  # must not raise

        bus.publish(MicLevel(rms=0.4, vad_prob=0.9))
        bus.publish(VrchatDetected(True))
        bus.publish(EngineStateChanged("stt", "ready"))
        qapp.processEvents()

        assert check._facts.mic_seen is False
        assert check._facts.vrchat_found is False
        assert check._facts.engine_states == {}
    finally:
        window.close()
        window.deleteLater()
        bridge.detach()


# -- survives the window-rebuild path used on a UI-language change ---------


def test_controller_survives_app_swap_main_window(qapp, tmp_path):
    store = _store(tmp_path)
    bus = EventBus()
    pipeline = _FakePipeline()
    bridge = BusBridge(bus)
    old = _window(bridge, store, pipeline)
    check = start(bus, store, old, _FakeDetector())

    fresh = None
    try:
        bus.publish(MicLevel(rms=0.3, vad_prob=0.9))
        qapp.processEvents()
        assert check._facts.mic_seen is True

        def make_window():
            # Same bridge and pipeline as `old`: app.py's own make_window()
            # closes over one BusBridge and one Pipeline for the whole run,
            # only the window is rebuilt.
            return _window(bridge, store, pipeline)

        fresh = _swap_main_window(old, make_window, _FakeDetector(), None)
        qapp.processEvents()

        # A fact recorded before the rebuild must not have been reset: a
        # controller owned by the window would have lost it here.
        assert check._facts.mic_seen is True

        # Bus events published AFTER the rebuild must still reach the
        # controller: it was never torn down by the swap.
        bus.publish(VrchatDetected(True))
        qapp.processEvents()
        assert check._facts.vrchat_found is True

        # Captioning is read off the shared pipeline, not off either
        # window's button, so it survives the rebuild too.
        pipeline.set_captioning(True)
        check._recompute()
        assert check._facts.captioning is True
    finally:
        check.stop()
        if fresh is not None:
            fresh.close()
            fresh.deleteLater()
        bridge.detach()
