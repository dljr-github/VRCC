"""Tests for `ChatboxSender.submit_partial`: sends a tentative live-caption
chunk through the same fit/coalesce/token-bucket path as `submit_message`,
but never publishes `ChatboxSent` (a partial must not mark the caption log
row delivered). Split out of test_chatbox.py to stay under the line cap.
"""

from __future__ import annotations

import time

from vrcc.core.bus import EventBus
from vrcc.core.events import ChatboxSent


def _wait_until(predicate, timeout=2.0, interval=0.01):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()


class FakeClock:
    def __init__(self, start: float = 0.0) -> None:
        self.now = start

    def time(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


class FakeClient:
    def __init__(self, clock=None) -> None:
        self.calls: list[tuple[str, list, float]] = []
        self._clock = clock or (lambda: 0.0)

    def send_message(self, address, value) -> None:
        self.calls.append((address, list(value), self._clock()))


def make_cfg(**overrides):
    from vrcc.core.config import OscConfig

    return OscConfig(**overrides)


def make_sender(cfg, bus, clock, client=None):
    from vrcc.osc.chatbox import ChatboxSender

    client = client if client is not None else FakeClient(clock=clock.time)
    sender = ChatboxSender(
        cfg,
        bus,
        client_factory=lambda ip, port: client,
        clock=clock.time,
        sleep=clock.sleep,
    )
    return sender, client


def test_submit_partial_sends_but_publishes_no_chatboxsent():
    cfg = make_cfg(coalesce_latest_wins=False, overflow="send")
    bus = EventBus()
    clock = FakeClock()
    sender, client = make_sender(cfg, bus, clock)

    received = []
    bus.subscribe(ChatboxSent, received.append)

    sender.submit_partial("partial text")
    sender.start()
    assert _wait_until(lambda: any(c[0] == "/chatbox/input" for c in client.calls))
    time.sleep(0.02)  # give the (absent) publish a chance to land
    sender.stop()

    sent = [c for c in client.calls if c[0] == "/chatbox/input"]
    assert sent == [("/chatbox/input", ["partial text", True, cfg.notification_sfx], 0.0)]
    assert received == []  # a partial never marks the log row delivered


def test_submit_message_still_publishes_chatboxsent():
    # Regression: submit_partial must not have broken the normal path.
    cfg = make_cfg(coalesce_latest_wins=False, overflow="send")
    bus = EventBus()
    clock = FakeClock()
    sender, client = make_sender(cfg, bus, clock)

    received = []
    bus.subscribe(ChatboxSent, received.append)

    sender.submit_message("hello", [], 1)
    sender.start()
    assert _wait_until(lambda: len(received) == 1)
    sender.stop()

    assert received[0].text == "hello"
    assert received[0].utterance_id == 1


def test_submit_partial_ignores_empty_and_whitespace_text():
    cfg = make_cfg()
    bus = EventBus()
    clock = FakeClock()
    sender, client = make_sender(cfg, bus, clock)

    sender.submit_partial("")
    sender.submit_partial("   ")
    sender.start()
    sender.stop()

    assert [c for c in client.calls if c[0] == "/chatbox/input"] == []
