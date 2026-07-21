"""Tests for the Qt bridge that fans EventBus events out as Qt signals,
run headless via the offscreen platform with cross-thread queued delivery
(each assertion pumps the event loop before checking).
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import threading
import time

import pytest

from PySide6.QtCore import QCoreApplication, QObject
from PySide6.QtWidgets import QApplication

from vrcc.core.bus import EventBus
from vrcc.core.events import (
    AppError,
    ChatboxSent,
    DownloadProgress,
    EngineStateChanged,
    MicLevel,
    MuteChanged,
    PhraseRecognized,
    PhraseTranslated,
    UpdateCheckResult,
)
from vrcc.gui.bridge import BusBridge


@pytest.fixture(scope="module")
def qapp():
    """One QApplication for the whole module (Qt allows only one). Reuse an
    existing instance if some other test already created it."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


class _Collector(QObject):
    """Records signal emissions. Lives on the GUI thread so cross-thread
    emits queue onto it and are delivered on processEvents()."""

    def __init__(self, bridge: BusBridge) -> None:
        super().__init__()
        self.mic: list[tuple[float, float]] = []
        self.recognized: list[object] = []
        self.translated: list[object] = []
        self.chatbox: list[object] = []
        self.mute: list[object] = []
        self.download: list[object] = []
        self.engine: list[object] = []
        self.error: list[object] = []
        self.update_result: list[object] = []
        bridge.mic_level.connect(self._on_mic)
        bridge.phrase_recognized.connect(self.recognized.append)
        bridge.phrase_translated.connect(self.translated.append)
        bridge.chatbox_sent.connect(self.chatbox.append)
        bridge.mute_changed.connect(self.mute.append)
        bridge.download_progress.connect(self.download.append)
        bridge.engine_state.connect(self.engine.append)
        bridge.app_error.connect(self.error.append)
        bridge.update_result.connect(self.update_result.append)

    def _on_mic(self, rms: float, vad: float) -> None:
        self.mic.append((rms, vad))


class _FakeClock:
    """Deterministic monotonic clock. Returns each queued time once, then
    repeats the last value (so an unexpected extra read can't raise)."""

    def __init__(self, times: list[float]) -> None:
        self._times = list(times)
        self._i = 0

    def __call__(self) -> float:
        t = self._times[min(self._i, len(self._times) - 1)]
        self._i += 1
        return t


def _publish_all(bus: EventBus, events: list[object]) -> None:
    """Publish `events` synchronously on a worker thread and wait for it to
    finish, so signal emits happen off the GUI thread."""
    t = threading.Thread(target=lambda: [bus.publish(e) for e in events])
    t.start()
    t.join()


def _pump_until(predicate, timeout: float = 2.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        QCoreApplication.processEvents()
        if predicate():
            return True
        time.sleep(0.005)
    QCoreApplication.processEvents()
    return predicate()


# -- one delivery test per event type --------------------------------------


def test_mic_level_delivered_from_worker_thread(qapp):
    bus = EventBus()
    bridge = BusBridge(bus)
    c = _Collector(bridge)

    _publish_all(bus, [MicLevel(rms=0.03, vad_prob=0.8)])

    assert _pump_until(lambda: len(c.mic) == 1)
    assert c.mic[0] == pytest.approx((0.03, 0.8))


def test_phrase_recognized_delivered(qapp):
    bus = EventBus()
    bridge = BusBridge(bus)
    c = _Collector(bridge)
    event = PhraseRecognized(
        utterance_id=1, text="hello", language="en", avg_logprob=-0.1, no_speech_prob=0.0
    )

    _publish_all(bus, [event])

    assert _pump_until(lambda: c.recognized == [event])


def test_phrase_translated_delivered(qapp):
    bus = EventBus()
    bridge = BusBridge(bus)
    c = _Collector(bridge)
    event = PhraseTranslated(
        utterance_id=1,
        original="hello",
        source_lang="English",
        translations=(("Japanese", "こんにちは"),),
    )

    _publish_all(bus, [event])

    assert _pump_until(lambda: c.translated == [event])


def test_chatbox_sent_delivered(qapp):
    bus = EventBus()
    bridge = BusBridge(bus)
    c = _Collector(bridge)
    event = ChatboxSent(text="hi", utterance_id=2)

    _publish_all(bus, [event])

    assert _pump_until(lambda: c.chatbox == [event])


def test_mute_changed_delivered(qapp):
    bus = EventBus()
    bridge = BusBridge(bus)
    c = _Collector(bridge)
    event = MuteChanged(muted=True)

    _publish_all(bus, [event])

    assert _pump_until(lambda: c.mute == [event])


def test_download_progress_delivered(qapp):
    bus = EventBus()
    bridge = BusBridge(bus)
    c = _Collector(bridge)
    event = DownloadProgress(model_id="small", downloaded=50, total=100)

    _publish_all(bus, [event])

    assert _pump_until(lambda: c.download == [event])


def test_engine_state_delivered(qapp):
    bus = EventBus()
    bridge = BusBridge(bus)
    c = _Collector(bridge)
    event = EngineStateChanged(engine="stt", state="ready")

    _publish_all(bus, [event])

    assert _pump_until(lambda: c.engine == [event])


def test_app_error_delivered(qapp):
    bus = EventBus()
    bridge = BusBridge(bus)
    c = _Collector(bridge)
    event = AppError(code="BOOM", message="broke")

    _publish_all(bus, [event])

    assert _pump_until(lambda: c.error == [event])


def test_update_result_delivered(qapp):
    bus = EventBus()
    bridge = BusBridge(bus)
    c = _Collector(bridge)
    event = UpdateCheckResult(available=True, latest="1.2.0", url="https://example.com")

    _publish_all(bus, [event])

    assert _pump_until(lambda: c.update_result == [event])


# -- MicLevel throttle ------------------------------------------------------


def test_mic_level_throttled_drops_bursts(qapp):
    """Five MicLevel events published within one throttle window collapse to
    a single emission (the first; the rest are dropped)."""
    bus = EventBus()
    bridge = BusBridge(bus, clock=_FakeClock([1000.0]))
    c = _Collector(bridge)

    events = [MicLevel(rms=0.01 * i, vad_prob=0.5) for i in range(5)]
    _publish_all(bus, events)

    # Give the loop a chance to deliver everything that WILL be delivered.
    _pump_until(lambda: len(c.mic) >= 1)
    QCoreApplication.processEvents()
    assert 0 < len(c.mic) < len(events)
    assert len(c.mic) == 1


def test_mic_level_emits_again_after_interval(qapp):
    """Once the ~10 Hz gate has elapsed, a later MicLevel emits again."""
    bus = EventBus()
    # First event at t=1000.0 emits; next two within 100ms drop; the fourth
    # at t=1000.2 (200ms later) clears the gate and emits again.
    clock = _FakeClock([1000.0, 1000.0, 1000.0, 1000.2, 1000.2])
    bridge = BusBridge(bus, clock=clock)
    c = _Collector(bridge)

    events = [MicLevel(rms=0.02, vad_prob=0.5) for _ in range(5)]
    _publish_all(bus, events)

    assert _pump_until(lambda: len(c.mic) == 2)


# -- detach -----------------------------------------------------------------


def test_detach_stops_delivery(qapp):
    bus = EventBus()
    bridge = BusBridge(bus)
    c = _Collector(bridge)

    bridge.detach()
    _publish_all(
        bus,
        [
            MicLevel(rms=0.03, vad_prob=0.8),
            PhraseRecognized(
                utterance_id=1, text="x", language="en", avg_logprob=0.0, no_speech_prob=0.0
            ),
            AppError(code="X", message="y"),
        ],
    )

    # Nothing should arrive; a short negative wait is enough.
    assert not _pump_until(
        lambda: bool(c.mic or c.recognized or c.error), timeout=0.3
    )


# -- MainWindow smoke construct (visual verification is Task 17) -------------


class _FakePipeline:
    def __init__(self) -> None:
        self.captioning_enabled = True
        self.typed: list[str] = []
        self.captioning_calls: list[bool] = []

    def submit_typed(self, text: str) -> bool:
        self.typed.append(text)
        return True

    def set_captioning(self, enabled: bool) -> None:
        self.captioning_calls.append(enabled)
        self.captioning_enabled = enabled


def test_main_window_constructs(qapp, tmp_path):
    from vrcc.core.config import ConfigStore
    from vrcc.gui.main_window import MainWindow

    bus = EventBus()
    bridge = BusBridge(bus)
    store = ConfigStore(tmp_path / "config.json")
    pipeline = _FakePipeline()

    window = MainWindow(
        bridge,
        store,
        pipeline,
        on_open_settings=lambda: None,
        on_open_models=lambda: None,
    )
    try:
        assert window.windowTitle() == "VRCC"
    finally:
        window.close()
        window.deleteLater()
        QCoreApplication.processEvents()
