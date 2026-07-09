"""Tests for live rate reconfiguration of the OSC chatbox throttle:
``TokenBucket.reconfigure`` (clamp, no free burst, keep _last_refill) and
``ChatboxSender.reconfigure_rate`` (retune the bucket while draining).
"""

from __future__ import annotations

import time

from vrcc.core.bus import EventBus
from vrcc.core.config import OscConfig
from vrcc.osc.chatbox import ChatboxSender, TokenBucket


class FakeClock:
    """Monotonic-like fake clock: `sleep(s)` advances instead of blocking, so
    throttling is driven at CPU speed with no real waiting."""

    def __init__(self, start: float = 0.0) -> None:
        self.now = start

    def time(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


def make_cfg(**overrides) -> OscConfig:
    return OscConfig(**overrides)


def make_sender(cfg, bus, clock) -> ChatboxSender:
    return ChatboxSender(
        cfg,
        bus,
        client_factory=lambda ip, port: object(),
        clock=clock.time,
        sleep=clock.sleep,
    )


# -- TokenBucket.reconfigure ------------------------------------------------


def test_reconfigure_clamps_tokens_to_smaller_capacity():
    clock = FakeClock()
    bucket = TokenBucket(capacity=5, refill_interval_s=1.3, clock=clock.time)
    bucket.reconfigure(capacity=2, refill_interval_s=1.3)  # 5 full -> clamp to 2
    assert [bucket.try_acquire() for _ in range(3)] == [True, True, False]


def test_reconfigure_larger_capacity_grants_no_free_burst():
    clock = FakeClock()
    bucket = TokenBucket(capacity=2, refill_interval_s=1.3, clock=clock.time)
    bucket.try_acquire()
    bucket.try_acquire()
    assert bucket.try_acquire() is False  # drained
    bucket.reconfigure(capacity=10, refill_interval_s=1.3)
    assert bucket.try_acquire() is False  # bigger bucket must not refill tokens


def test_reconfigure_keeps_last_refill_no_phantom_credit():
    # Reconfigure must keep _last_refill: the half-interval already accrued is
    # preserved (not reset), so another half-interval yields a full token.
    clock = FakeClock()
    bucket = TokenBucket(capacity=5, refill_interval_s=1.3, clock=clock.time)
    for _ in range(5):
        bucket.try_acquire()  # drained; _last_refill == 0.0
    clock.now += 0.65  # half an interval accrues but is not realized yet
    bucket.reconfigure(capacity=5, refill_interval_s=1.3)
    clock.now += 0.65  # the other half: a full interval since _last_refill
    assert bucket.try_acquire() is True


def test_reconfigure_applies_new_refill_interval():
    clock = FakeClock()
    bucket = TokenBucket(capacity=1, refill_interval_s=1.3, clock=clock.time)
    bucket.try_acquire()  # drained
    bucket.reconfigure(capacity=1, refill_interval_s=0.5)  # faster refill
    clock.now += 0.5
    assert bucket.try_acquire() is True


# -- ChatboxSender.reconfigure_rate -----------------------------------------


def test_reconfigure_rate_retunes_bucket_live_without_free_burst():
    cfg = make_cfg(burst=2, min_interval_s=1.3)
    bus = EventBus()
    clock = FakeClock()
    sender = make_sender(cfg, bus, clock)

    # Drain the initial 2-token burst directly on the sender's bucket.
    assert sender._bucket.try_acquire() and sender._bucket.try_acquire()
    assert sender._bucket.try_acquire() is False

    sender.reconfigure_rate(burst=5, min_interval_s=0.5)
    assert sender._bucket.try_acquire() is False  # bigger burst: no free tokens
    clock.now += 0.5  # the new, faster interval is in effect
    assert sender._bucket.try_acquire() is True


class _NullClient:
    def send_message(self, address, value) -> None:
        pass


def test_reconfigure_rate_is_safe_while_worker_drains():
    # Smoke test: retune while the real worker thread is running and draining.
    cfg = make_cfg(burst=1, min_interval_s=0.01, overflow="send")
    bus = EventBus()
    sender = ChatboxSender(cfg, bus, client_factory=lambda ip, port: _NullClient())
    sender.start()
    try:
        for _ in range(20):
            sender.submit("hi", 1)
            sender.reconfigure_rate(burst=3, min_interval_s=0.02)
            time.sleep(0)
    finally:
        sender.stop()
