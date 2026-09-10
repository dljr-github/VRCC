"""Tests for the OSC chatbox sender's inter-chunk pacing: the readability
pause `split_delay_s` inserts between parts of one split message, and its
interruption by a newer caption that coalesces the remaining parts away.

Split out of tests/test_chatbox.py for the file-length cap; this file owns
everything that waits on real wall-clock time (the delay is a plain
`threading.Event.wait`, not driven by the fake clock), while the rest of
`test_chatbox.py` stays on the fake clock throughout.
"""

from __future__ import annotations

import time

from vrcc.core.bus import EventBus
from vrcc.core.events import ChatboxSent
from vrcc.osc.chatbox import fit_chatbox
from tests.test_chatbox import FakeClock, _wait_until, make_cfg, make_sender


def test_split_message_chunks_sequence_through_bucket_in_order():
    # split_delay_s is a plain Event.wait, real wall-clock time, so keep it
    # tiny here -- this test cares about chunk order, not pacing.
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


def test_split_delay_separates_chunks_by_at_least_split_delay_s():
    # With coalescing off nothing preempts the pause, so consecutive parts
    # of one message must still be separated by split_delay_s of real wall
    # time (the pause is a plain Event.wait, not driven by the fake clock).
    split_delay_s = 0.08
    cfg = make_cfg(
        overflow="split", coalesce_latest_wins=False, split_delay_s=split_delay_s
    )
    bus = EventBus()
    clock = FakeClock()
    sender, _client = make_sender(cfg, bus, clock)

    send_times = []
    bus.subscribe(ChatboxSent, lambda _e: send_times.append(time.monotonic()))

    long_text = " ".join(f"word{i}" for i in range(60))
    expected_chunks = fit_chatbox(long_text, "split")
    assert len(expected_chunks) >= 3  # sanity: fixture actually splits

    sender.submit(long_text, 1)
    sender.start()
    assert _wait_until(lambda: len(send_times) == len(expected_chunks))
    sender.stop()

    # A couple of ms of tolerance for the host timer's own resolution, not
    # for anything the sender does.
    gaps = [later - earlier for earlier, later in zip(send_times, send_times[1:])]
    assert all(gap >= split_delay_s - 0.01 for gap in gaps)


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


def test_a_newer_submit_preempts_the_split_delay_of_the_old_group():
    # A newer caption that coalesces away the rest of the old group must not
    # wait out the pause meant for the parts it just discarded.
    cfg = make_cfg(overflow="split", coalesce_latest_wins=True, split_delay_s=2.0)
    bus = EventBus()
    clock = FakeClock()
    sender, client = make_sender(cfg, bus, clock)

    received = []
    bus.subscribe(ChatboxSent, received.append)

    long_text = " ".join(f"word{i}" for i in range(60))
    expected_chunks = fit_chatbox(long_text, "split")
    assert len(expected_chunks) == 3  # sanity: fixture splits into exactly 3

    sender.submit(long_text, 1)
    sender.start()
    assert _wait_until(lambda: len(client.calls) == 1)

    start = time.monotonic()
    sender.submit("a different message", 2)

    assert _wait_until(
        lambda: any(e.utterance_id == 2 for e in received), timeout=3.0
    )
    elapsed = time.monotonic() - start
    sender.stop()

    assert elapsed < 0.2
