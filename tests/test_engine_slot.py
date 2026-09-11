"""EngineSlot: swap/borrow/current, and the invariant every caller of the
pipeline's engines depends on (Pipeline, pipeline_jobs, HeardStream all share
one slot instance per engine kind)."""

from __future__ import annotations

import threading
import time

from vrcc.core.engine_slot import EngineSlot


def test_swap_installs_and_returns_the_previous_engine():
    slot = EngineSlot("old")
    old = slot.swap("new")
    assert old == "old"
    assert slot.current == "new"


def test_swap_from_empty_returns_none():
    slot = EngineSlot()
    assert slot.current is None
    assert slot.swap("first") is None
    assert slot.current == "first"


def test_borrow_yields_the_current_engine():
    slot = EngineSlot("engine")
    with slot.borrow() as engine:
        assert engine == "engine"


def test_borrow_yields_none_when_empty():
    slot = EngineSlot()
    with slot.borrow() as engine:
        assert engine is None


def test_borrow_holds_the_lock_across_the_call():
    slot = EngineSlot("engine")
    entered = threading.Event()
    release = threading.Event()
    swapped_at = []

    def swapper():
        entered.wait(1.0)
        slot.swap("new")
        swapped_at.append(time.monotonic())

    t = threading.Thread(target=swapper)
    t.start()
    with slot.borrow():
        entered.set()
        time.sleep(0.1)
        released_at = time.monotonic()
        release.set()
    t.join(2.0)

    assert swapped_at, "the swap never completed"
    assert swapped_at[0] >= released_at, "swap ran before borrow released the lock"
    assert slot.current == "new"


def test_current_does_not_block_while_borrow_is_held():
    slot = EngineSlot("engine")
    with slot.borrow():
        # The unlocked read must return immediately: a caller that only
        # tests truthiness (e.g. the GUI thread reading mt_active) must
        # never queue behind an in-flight call.
        assert slot.current == "engine"


def test_swap_waits_for_an_in_flight_borrow_to_finish():
    slot = EngineSlot("old")
    entered = threading.Event()
    hold_s = 0.15

    def holder():
        with slot.borrow():
            entered.set()
            time.sleep(hold_s)

    t = threading.Thread(target=holder)
    t.start()
    entered.wait(1.0)
    start = time.monotonic()
    old = slot.swap("new")
    elapsed = time.monotonic() - start
    t.join(2.0)

    assert old == "old"
    assert slot.current == "new"
    assert elapsed >= hold_s * 0.5, "swap did not wait for the in-flight borrow"
