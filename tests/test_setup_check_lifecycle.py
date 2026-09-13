"""The setup check controller's lifecycle: completion persists the flag
exactly once and does not re-fire on a manual re-arm, the 250 ms poll for
captioning/send_enabled actually fires and actually stops, worker-thread
events are marshalled onto the GUI thread rather than touched from a
callback, stop() is idempotent even across an event-loop turn, and the
controller survives the window rebuild app._swap_main_window performs on a
UI-language change.

Split out of test_setup_check.py to stay under the repo's 500-line-per-file
cap; the fixtures and fakes it needs are defined there.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import threading
import time

from PySide6.QtCore import QThread, QTimer

from vrcc.core.bus import EventBus
from vrcc.core.events import ChatboxSent, EngineStateChanged, MicLevel, VrchatDetected
from vrcc.gui.bridge import BusBridge
from vrcc.gui.setup_check import start
from vrcc.gui.window_swap import _swap_main_window
from tests.test_setup_check import (  # noqa: F401 -- shared fixtures/fakes
    _FakeDetector,
    _FakePipeline,
    _phrase,
    _store,
    _window,
    no_language_nudge,
    qapp,
)


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


def _run_loop_once(app) -> None:
    """A genuine (short) event-loop turn. Plain processEvents() does not
    honor QObject.deleteLater() under offscreen QPA without an actual
    exec()/quit() cycle, so pumping this way is what a test needs to prove
    a stop() this controller can double-call really meets a since-deleted
    widget rather than one merely scheduled for deletion."""
    QTimer.singleShot(0, app.quit)
    app.exec()


# -- completion persists exactly once, and only on a fresh transition -----


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


def test_reset_flag_does_not_flip_back_while_facts_stay_passed(qapp, tmp_path, monkeypatch):
    """A later task reopens the check from Settings by resetting
    gui.setup_check_done to False, and the common case for doing that is
    "already finished, I just want to see it again": the facts are still all
    passed. A level-triggered completion check (write whenever
    required_passed(facts) holds) would flip the flag straight back to True
    before the reopened panel is ever seen. Only a fresh False -> True
    transition may write it."""
    store = _store(tmp_path)
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
        bus.publish(ChatboxSent(text="hi", utterance_id=1))
        qapp.processEvents()
        check._recompute()
        assert store.config.gui.setup_check_done is True

        store.config.gui.setup_check_done = False
        check._recompute()
        assert store.config.gui.setup_check_done is False
    finally:
        check.stop()
        window.close()
        window.deleteLater()
        bridge.detach()


# -- the 250 ms poll for captioning / send_enabled -------------------------


def test_poll_timer_updates_captioning_with_no_manual_recompute(qapp, tmp_path):
    store = _store(tmp_path)
    bus = EventBus()
    pipeline = _FakePipeline()
    bridge = BusBridge(bus)
    window = _window(bridge, store, pipeline)
    check = start(bus, store, window, _FakeDetector())
    try:
        assert check._facts.captioning is False
        pipeline.set_captioning(True)
        # No _recompute() call here: only the timer itself may pick this up.
        assert _pump(qapp, lambda: check._facts.captioning is True)
    finally:
        check.stop()
        window.close()
        window.deleteLater()
        bridge.detach()


def test_stop_stops_the_poll_timer(qapp, tmp_path):
    # Asserted on the timer's own isActive(), not on a fact staying put:
    # _recompute() also short-circuits once _panel is None (the same guard
    # that protects a late-delivered event), so a fact-only assertion here
    # would pass even if stop() forgot to stop the timer at all.
    store = _store(tmp_path)
    bus = EventBus()
    pipeline = _FakePipeline()
    bridge = BusBridge(bus)
    window = _window(bridge, store, pipeline)
    check = start(bus, store, window, _FakeDetector())
    try:
        assert check._timer.isActive()
        check.stop()
        assert not check._timer.isActive()
    finally:
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
        # evaluate() also runs from the 250 ms poll timer, itself native to
        # the GUI thread with no marshalling involved; the thread assertion
        # above would hold even if the cross-thread emit above were silently
        # dropped. mic_seen is set only inside _on_event, so pinning it too
        # proves the emitted MicLevel itself was what arrived.
        assert check._facts.mic_seen is True
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


def test_stop_is_idempotent_across_an_event_loop_turn(qapp, tmp_path):
    # A back-to-back double stop() never gives Qt a chance to actually
    # process the panel's deleteLater(); pumping between the two calls does,
    # which is the only way the second call could reach an already-deleted
    # C++ widget.
    store = _store(tmp_path)
    bus = EventBus()
    pipeline = _FakePipeline()
    bridge = BusBridge(bus)
    window = _window(bridge, store, pipeline)
    check = start(bus, store, window, _FakeDetector())
    try:
        check.stop()
        _run_loop_once(qapp)
        check.stop()  # must not raise on the by-now-deleted panel
    finally:
        window.close()
        window.deleteLater()
        bridge.detach()


def test_stop_absorbs_an_event_delivered_after_it(qapp, tmp_path):
    """A cross-thread emit queued before stop() runs is delivered by Qt as a
    plain queued slot call; PySide prints and swallows an exception raised
    inside one rather than letting it reach app.exec()'s caller, which would
    make a thread-and-pump version of this test pass whether or not the
    guard exists. Calling _on_event directly exercises the identical branch
    (isinstance dispatch, then _recompute()) as a normal Python call, whose
    exception (if the guard were missing) propagates to pytest like any
    other. VrchatDetected unconditionally flips the vrchat row's evaluated
    state, which is what drives _recompute() into calling panel.apply()
    rather than short-circuiting on an unchanged states dict."""
    store = _store(tmp_path)
    bus = EventBus()
    pipeline = _FakePipeline()
    bridge = BusBridge(bus)
    window = _window(bridge, store, pipeline)
    check = start(bus, store, window, _FakeDetector())
    try:
        check.stop()
        check._on_event(VrchatDetected(True))  # must not raise
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
