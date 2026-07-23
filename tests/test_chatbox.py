"""Tests for the OSC chatbox sender: token-bucket throttling, queueing,
coalescing, typing indicator, and send-failure handling.
"""

import logging
import time

import pytest

from vrcc.core.bus import EventBus
from vrcc.core.config import OscConfig
from vrcc.core.events import ChatboxSent, TypingStateChanged
from vrcc.osc.chatbox import CHATBOX_LIMIT, ChatboxSender, TokenBucket, fit_chatbox


def _wait_until(predicate, timeout=2.0, interval=0.01):
    """Poll `predicate` until it's truthy or `timeout` (real seconds)
    elapses. Only used to synchronize with the sender's background thread
    (e.g. "has it published N events yet?") -- the interval is test-harness
    plumbing, never a stand-in for the token-bucket's own timing, which is
    always driven by the fake clock/sleep instead."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()


class FakeClock:
    """Monotonic-like fake clock. `sleep(s)` advances the clock instead of
    blocking, so a background worker polling this clock via injected
    `sleep`/`clock` callables races through waits at CPU speed instead of
    wall-clock speed -- no real waiting is needed to exercise throttling."""

    def __init__(self, start: float = 0.0) -> None:
        self.now = start

    def time(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


class FakeClient:
    """Fake OSC client recording every `send_message` call as
    `(address, args, clock_time_at_call)`. `fail_next` makes exactly the
    next call raise OSError, to test send-failure handling."""

    def __init__(self, clock=None) -> None:
        self.calls: list[tuple[str, list, float]] = []
        self._clock = clock or (lambda: 0.0)
        self.fail_next = False

    def send_message(self, address, value) -> None:
        if self.fail_next:
            self.fail_next = False
            raise OSError("simulated send failure")
        self.calls.append((address, list(value), self._clock()))


def make_cfg(**overrides) -> OscConfig:
    return OscConfig(**overrides)


def make_sender(cfg, bus, clock, client=None):
    client = client if client is not None else FakeClient(clock=clock.time)
    sender = ChatboxSender(
        cfg,
        bus,
        client_factory=lambda ip, port: client,
        clock=clock.time,
        sleep=clock.sleep,
    )
    return sender, client


# -- TokenBucket -----------------------------------------------------------


def test_token_bucket_starts_full_allows_capacity_immediate_acquires():
    clock = FakeClock()
    bucket = TokenBucket(capacity=5, refill_interval_s=1.3, clock=clock.time)
    assert [bucket.try_acquire() for _ in range(5)] == [True] * 5


def test_token_bucket_denies_when_empty():
    clock = FakeClock()
    bucket = TokenBucket(capacity=5, refill_interval_s=1.3, clock=clock.time)
    for _ in range(5):
        bucket.try_acquire()
    assert bucket.try_acquire() is False


def test_token_bucket_seconds_until_token_is_zero_when_available():
    clock = FakeClock()
    bucket = TokenBucket(capacity=5, refill_interval_s=1.3, clock=clock.time)
    assert bucket.seconds_until_token() == 0.0


def test_token_bucket_refills_one_token_after_one_interval():
    clock = FakeClock()
    bucket = TokenBucket(capacity=5, refill_interval_s=1.3, clock=clock.time)
    for _ in range(5):
        bucket.try_acquire()
    assert bucket.try_acquire() is False

    clock.now += 1.3
    assert bucket.try_acquire() is True
    assert bucket.try_acquire() is False  # only one token refilled


def test_token_bucket_fractional_accumulation():
    clock = FakeClock()
    bucket = TokenBucket(capacity=5, refill_interval_s=1.3, clock=clock.time)
    for _ in range(5):
        bucket.try_acquire()

    clock.now += 0.65  # half an interval: not enough for a full token yet
    assert bucket.try_acquire() is False
    assert bucket.seconds_until_token() == pytest.approx(0.65)

    clock.now += 0.65  # the other half: now a full interval has accumulated
    assert bucket.try_acquire() is True


def test_token_bucket_refill_caps_at_capacity():
    clock = FakeClock()
    bucket = TokenBucket(capacity=5, refill_interval_s=1.3, clock=clock.time)
    clock.now += 100  # idle for a long time; must not over-accumulate
    acquired = [bucket.try_acquire() for _ in range(6)]
    assert acquired == [True, True, True, True, True, False]


# -- ChatboxSender ---------------------------------------------------------


def test_submit_ignores_empty_and_whitespace_text():
    cfg = make_cfg()
    bus = EventBus()
    clock = FakeClock()
    sender, client = make_sender(cfg, bus, clock)

    sender.submit("", 1)
    sender.submit("   ", 2)
    sender.start()
    sender.stop()

    # No /chatbox/input sends at all; stop()'s best-effort typing-False is
    # the only call.
    assert [c for c in client.calls if c[0] == "/chatbox/input"] == []


def test_burst_of_capacity_sends_immediately_then_throttles():
    cfg = make_cfg(
        burst=5, min_interval_s=1.3, coalesce_latest_wins=False, overflow="send"
    )
    bus = EventBus()
    clock = FakeClock()
    sender, client = make_sender(cfg, bus, clock)

    received = []
    bus.subscribe(ChatboxSent, received.append)

    for i in range(6):
        sender.submit(f"message {i}", i)

    sender.start()
    assert _wait_until(lambda: len(received) == 6)
    sender.stop()

    assert [e.text for e in received] == [f"message {i}" for i in range(6)]
    timestamps = [t for _, _, t in client.calls]
    assert timestamps[:5] == [0.0] * 5  # burst: no waiting needed
    assert timestamps[5] >= 1.3  # 6th needed one refill interval


def test_truncated_message_sets_truncated_flag_on_event():
    cfg = make_cfg(overflow="truncate", coalesce_latest_wins=False)
    bus = EventBus()
    clock = FakeClock()
    sender, _client = make_sender(cfg, bus, clock)
    received = []
    bus.subscribe(ChatboxSent, received.append)

    sender.submit("x" * (CHATBOX_LIMIT + 50), 1)  # over the 144-char limit
    sender.start()
    assert _wait_until(lambda: len(received) == 1)
    sender.stop()

    assert received[0].truncated is True


def test_within_limit_message_is_not_flagged_truncated():
    cfg = make_cfg(overflow="truncate", coalesce_latest_wins=False)
    bus = EventBus()
    clock = FakeClock()
    sender, _client = make_sender(cfg, bus, clock)
    received = []
    bus.subscribe(ChatboxSent, received.append)

    sender.submit("short message", 1)
    sender.start()
    assert _wait_until(lambda: len(received) == 1)
    sender.stop()

    assert received[0].truncated is False


def test_coalesce_latest_wins_replaces_unsent_queued_message():
    cfg = make_cfg(coalesce_latest_wins=True, overflow="send")
    bus = EventBus()
    clock = FakeClock()
    sender, client = make_sender(cfg, bus, clock)

    received = []
    bus.subscribe(ChatboxSent, received.append)

    # Both submits land before the worker thread ever runs, so this
    # deterministically exercises "replace queued-unsent" rather than racing
    # a live worker.
    sender.submit("first", 1)
    sender.submit("second", 2)

    sender.start()
    assert _wait_until(lambda: len(received) == 1)
    sender.stop()

    assert [e.text for e in received] == ["second"]
    assert [e.utterance_id for e in received] == [2]


def test_coalesce_disabled_keeps_both_queued_messages():
    cfg = make_cfg(coalesce_latest_wins=False, overflow="send")
    bus = EventBus()
    clock = FakeClock()
    sender, client = make_sender(cfg, bus, clock)

    received = []
    bus.subscribe(ChatboxSent, received.append)

    sender.submit("first", 1)
    sender.submit("second", 2)

    sender.start()
    assert _wait_until(lambda: len(received) == 2)
    sender.stop()

    assert [e.text for e in received] == ["first", "second"]


def test_queue_is_capped_dropping_oldest_and_logs_once(caplog):
    # With coalesce disabled the queue is bounded at 64 pending chunks:
    # overflow drops the OLDEST (newest speech wins) and warns exactly once.
    cfg = make_cfg(coalesce_latest_wins=False, overflow="send")
    bus = EventBus()
    clock = FakeClock()
    sender, client = make_sender(cfg, bus, clock)

    with caplog.at_level(logging.WARNING, logger="vrcc.osc"):
        for i in range(70):  # worker not started: nothing drains
            sender.submit(f"msg{i}", i)

    assert len(sender._queue) == 64
    texts = [item[0] for item in sender._queue]
    assert texts[0] == "msg6"  # msg0..msg5 dropped (oldest)
    assert texts[-1] == "msg69"  # newest retained
    assert sum("dropping oldest" in r.message for r in caplog.records) == 1


def test_split_message_chunks_sequence_through_bucket_in_order():
    # split_delay_s is real wall-clock time (see _run's stop_flag.wait), so
    # keep it tiny here -- this test cares about chunk order, not pacing.
    cfg = make_cfg(overflow="split", burst=2, min_interval_s=1.3, split_delay_s=0.01)
    bus = EventBus()
    clock = FakeClock()
    sender, client = make_sender(cfg, bus, clock)

    received = []
    bus.subscribe(ChatboxSent, received.append)

    long_text = " ".join(f"word{i}" for i in range(60))
    expected_chunks = fit_chatbox(long_text, "split")
    assert len(expected_chunks) >= 3  # sanity: fixture actually splits

    sender.submit(long_text, 42)
    sender.start()
    assert _wait_until(lambda: len(received) == len(expected_chunks))
    sender.stop()

    assert [e.text for e in received] == expected_chunks
    assert all(e.utterance_id == 42 for e in received)


def test_split_delay_waits_between_chunks_via_stop_flag_wait():
    cfg = make_cfg(overflow="split", coalesce_latest_wins=False, split_delay_s=0.05)
    bus = EventBus()
    clock = FakeClock()
    sender, _client = make_sender(cfg, bus, clock)

    received = []
    bus.subscribe(ChatboxSent, received.append)

    recorded_timeouts = []
    real_wait = sender._stop_flag.wait

    def recording_wait(timeout=None):
        recorded_timeouts.append(timeout)
        return real_wait(timeout)

    sender._stop_flag.wait = recording_wait

    long_text = " ".join(f"word{i}" for i in range(60))
    expected_chunks = fit_chatbox(long_text, "split")
    assert len(expected_chunks) >= 3  # sanity: fixture actually splits

    sender.submit(long_text, 1)
    sender.start()
    assert _wait_until(lambda: len(received) == len(expected_chunks))
    sender.stop()

    non_last_delay_waits = [t for t in recorded_timeouts if t == 0.05]
    assert len(non_last_delay_waits) == len(expected_chunks) - 1


def test_stop_returns_promptly_while_waiting_on_split_delay():
    cfg = make_cfg(overflow="split", coalesce_latest_wins=False, split_delay_s=5.0)
    bus = EventBus()
    clock = FakeClock()
    sender, _client = make_sender(cfg, bus, clock)

    received = []
    bus.subscribe(ChatboxSent, received.append)

    long_text = " ".join(f"word{i}" for i in range(60))
    sender.submit(long_text, 1)
    sender.start()
    assert _wait_until(lambda: len(received) >= 1)  # inside the post-send delay now
    worker_thread = sender._thread

    start = time.monotonic()
    sender.stop()
    elapsed = time.monotonic() - start

    assert elapsed < 1.0  # well under both the 5s delay and the 2s join timeout
    assert not worker_thread.is_alive()


def test_a_newer_submit_replaces_the_whole_remaining_split_chunk_group():
    # coalesce applies at message-group granularity: replacing mid-group
    # must drop the *rest* of the old group's chunks, not interleave them
    # with the new message's chunks.
    # split_delay_s is real wall-clock time; keep it tiny so the test stays
    # fast -- it's asserting coalescing behavior, not pacing.
    cfg = make_cfg(overflow="split", coalesce_latest_wins=True, split_delay_s=0.01)
    bus = EventBus()
    clock = FakeClock()
    sender, client = make_sender(cfg, bus, clock)

    received = []
    bus.subscribe(ChatboxSent, received.append)

    old_text = " ".join(f"old{i}" for i in range(60))
    new_text = " ".join(f"new{i}" for i in range(60))
    expected_new_chunks = fit_chatbox(new_text, "split")

    # Both submits land before start(), so the old group never begins
    # sending -- this deterministically proves whole-group replacement.
    sender.submit(old_text, 1)
    sender.submit(new_text, 2)

    sender.start()
    assert _wait_until(lambda: len(received) == len(expected_new_chunks))
    sender.stop()

    assert [e.text for e in received] == expected_new_chunks
    assert all(e.utterance_id == 2 for e in received)


def test_set_typing_sends_immediately_and_dedupes_repeats():
    cfg = make_cfg()
    bus = EventBus()
    clock = FakeClock()
    sender, client = make_sender(cfg, bus, clock)

    typing_events = []
    bus.subscribe(TypingStateChanged, typing_events.append)

    sender.set_typing(True)
    sender.set_typing(True)  # repeat: deduped, no second send
    sender.set_typing(False)

    assert [(addr, args) for addr, args, _ in client.calls] == [
        ("/chatbox/typing", [True]),
        ("/chatbox/typing", [False]),
    ]
    assert [e.typing for e in typing_events] == [True, False]


def test_set_typing_publishes_while_typing_lock_still_held():
    # The publish must happen INSIDE set_typing's `with self._typing_lock`
    # block (mirroring the OSC send), so two threads' publish order can't
    # invert relative to their actual send order. A non-blocking acquire
    # from inside the subscriber proves the lock is still held at publish
    # time (a plain Lock, not reentrant: acquire() only succeeds if free).
    cfg = make_cfg()
    bus = EventBus()
    clock = FakeClock()
    sender, _client = make_sender(cfg, bus, clock)

    lock_was_held = []

    def on_typing_changed(_event):
        acquired = sender._typing_lock.acquire(blocking=False)
        lock_was_held.append(not acquired)
        if acquired:
            sender._typing_lock.release()

    bus.subscribe(TypingStateChanged, on_typing_changed)
    sender.set_typing(True)

    assert lock_was_held == [True]


def test_reconfigure_swaps_client_atomically():
    cfg = make_cfg()
    bus = EventBus()
    clock = FakeClock()
    old_client = FakeClient(clock=clock.time)
    new_client = FakeClient(clock=clock.time)
    clients = iter([old_client, new_client])
    sender = ChatboxSender(
        cfg,
        bus,
        client_factory=lambda ip, port: next(clients),
        clock=clock.time,
        sleep=clock.sleep,
    )

    sender.set_typing(True)  # -> old_client
    sender.reconfigure("10.0.0.5", 9001)
    sender.set_typing(False)  # -> new_client (different value, not deduped)

    assert len(old_client.calls) == 1
    assert len(new_client.calls) == 1


def test_send_failure_is_logged_and_dropped_without_crashing(caplog):
    cfg = make_cfg(overflow="send", coalesce_latest_wins=False)
    bus = EventBus()
    clock = FakeClock()
    sender, client = make_sender(cfg, bus, clock)

    received = []
    bus.subscribe(ChatboxSent, received.append)

    client.fail_next = True
    sender.submit("will fail", 1)
    sender.submit("will succeed", 2)

    with caplog.at_level(logging.WARNING, logger="vrcc.osc"):
        sender.start()
        assert _wait_until(lambda: len(received) == 1)
        sender.stop()

    assert [e.text for e in received] == ["will succeed"]
    assert any("failed to send chatbox message" in r.message for r in caplog.records)


def test_stop_is_idempotent_and_safe_before_start():
    cfg = make_cfg()
    bus = EventBus()
    clock = FakeClock()
    sender, client = make_sender(cfg, bus, clock)

    sender.stop()  # never started: must not raise
    sender.stop()  # double-stop: must not raise

    sender.start()
    sender.stop()
    sender.stop()  # double-stop after a real start/stop: must not raise

    # stop() sends "/chatbox/typing False" best-effort.
    assert ("/chatbox/typing", [False]) in [(a, v) for a, v, _ in client.calls]


def test_start_is_idempotent():
    cfg = make_cfg()
    bus = EventBus()
    clock = FakeClock()
    sender, client = make_sender(cfg, bus, clock)

    sender.start()
    first_thread = sender._thread
    sender.start()  # must not spawn a second thread
    assert sender._thread is first_thread
    sender.stop()
