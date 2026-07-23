"""Tests for :class:`vrcc.core.pipeline_state.TypingTracker`'s atomic
begin/resolve callback (concurrency fix).

Split out from test_pipeline.py to stay under the line cap.
"""

from __future__ import annotations

from vrcc.core.pipeline_state import TypingTracker


# -- begin/resolve hold the tracker lock across the callback --------------


def test_begin_and_resolve_invoke_callback_while_lock_held():
    # A concurrent begin() can only interleave with a resolve()'s check if
    # the lock is released before the callback runs. A non-blocking acquire
    # from inside the callback proves it is still held (plain Lock, not
    # reentrant): acquire() only succeeds if nobody else holds it.
    tracker = TypingTracker()
    seen: list[tuple[bool, bool]] = []

    def on_change(value: bool) -> None:
        acquired = tracker._lock.acquire(blocking=False)
        seen.append((value, not acquired))
        if acquired:
            tracker._lock.release()

    tracker.begin(1, on_change)
    tracker.resolve(1, on_change)

    assert seen == [(True, True), (False, True)]


def test_resolve_skips_callback_while_another_utterance_in_flight():
    tracker = TypingTracker()
    calls: list[bool] = []
    tracker.begin(1, calls.append)
    tracker.begin(2, calls.append)
    tracker.resolve(1, calls.append)  # utterance 2 still in flight: no False
    assert calls == [True, True]
    tracker.resolve(2, calls.append)
    assert calls == [True, True, False]
