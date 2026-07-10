"""Tests for the ``MuteSync`` state machine: should_caption modes, start
gating, and MuteChanged transitions, driven with a fake OSCQuery server.
"""

import threading
import time

import pytest

from vrcc.core.bus import EventBus
from vrcc.core.config import MuteSyncConfig
from vrcc.core.events import AppError, MuteChanged
from vrcc.osc.mutesync import MuteSync


def _wait_until(predicate, timeout=2.0, interval=0.01):
    """Poll `predicate` until truthy or `timeout` real seconds elapse.

    Only used to synchronize with background daemon threads (HTTP/OSC
    servers, the initial-fetch thread) -- never as a stand-in for logic
    timing.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()


# -- fakes -----------------------------------------------------------------


class FakeServer:
    """Stand-in for :class:`OscQueryServer` used by ``MuteSync`` tests.

    Captures the ``on_mute`` callback so a test can simulate a VRChat push
    by calling it directly; ``start`` returns the preset ``mdns_ok``.
    """

    def __init__(self, name, on_mute, mdns_ok=True) -> None:
        self.name = name
        self.on_mute = on_mute
        self.mdns_ok = mdns_ok
        self.started = False
        self.stopped = False
        self.http_port = 9001
        self.osc_port = 9002

    def start(self) -> bool:
        self.started = True
        return self.mdns_ok

    def stop(self) -> None:
        self.stopped = True


def _factory(holder, mdns_ok=True):
    def make(name, on_mute):
        srv = FakeServer(name, on_mute, mdns_ok)
        holder.append(srv)
        return srv

    return make


def _collect(bus):
    events: list = []
    bus.subscribe(AppError, events.append)
    bus.subscribe(MuteChanged, events.append)
    return events


# -- should_caption truth table (3 modes x True/False/None) ----------------


@pytest.mark.parametrize(
    "mode,muted,expected",
    [
        ("ignore", True, True),
        ("ignore", False, True),
        ("ignore", None, True),
        ("pause", True, False),
        ("pause", False, True),
        ("pause", None, True),
        ("invert", True, True),
        ("invert", False, False),
        ("invert", None, False),
    ],
)
def test_should_caption_truth_table(mode, muted, expected):
    bus = EventBus()
    holder: list = []
    mute = MuteSync(
        MuteSyncConfig(mode=mode),
        "127.0.0.1",
        bus,
        server_factory=_factory(holder),
        initial_fetch=lambda: None,
    )
    mute._active = True  # the truth table describes behavior while sync is running
    if muted is not None:
        mute._on_mute(muted)
    assert mute.should_caption() is expected


@pytest.mark.parametrize("mode", ["ignore", "pause", "invert"])
def test_should_caption_fails_open_when_not_active(mode):
    # Regression: with the sync server not running (non-localhost OSC, disabled,
    # or after stop) _muted is None; 'invert' would return None-is-True = False
    # forever, silently disabling ALL captioning. Fail open instead.
    bus = EventBus()
    holder: list = []
    mute = MuteSync(
        MuteSyncConfig(mode=mode),
        "127.0.0.1",
        bus,
        server_factory=_factory(holder),
        initial_fetch=lambda: None,
    )
    assert mute.active is False
    assert mute.should_caption() is True


# -- MuteSync.start gating -------------------------------------------------


def test_non_localhost_publishes_error_and_stays_stopped():
    bus = EventBus()
    events = _collect(bus)
    holder: list = []
    mute = MuteSync(
        MuteSyncConfig(),
        "192.168.1.50",
        bus,
        server_factory=_factory(holder),
        initial_fetch=lambda: None,
    )
    mute.start()

    assert mute.active is False
    assert holder == []  # server factory never called
    errs = [e for e in events if isinstance(e, AppError)]
    assert len(errs) == 1
    assert errs[0].code == "MUTE_SYNC_REQUIRES_LOCALHOST"
    assert errs[0].detail == "192.168.1.50"


def test_disabled_is_noop():
    bus = EventBus()
    events = _collect(bus)
    holder: list = []
    mute = MuteSync(
        MuteSyncConfig(enabled=False),
        "127.0.0.1",
        bus,
        server_factory=_factory(holder),
        initial_fetch=lambda: None,
    )
    mute.start()

    assert mute.active is False
    assert holder == []
    assert events == []


def test_mdns_failed_still_active_and_publishes_error():
    bus = EventBus()
    events = _collect(bus)
    holder: list = []
    mute = MuteSync(
        MuteSyncConfig(),
        "127.0.0.1",
        bus,
        server_factory=_factory(holder, mdns_ok=False),
        initial_fetch=lambda: None,
    )
    mute.start()

    assert mute.active is True
    errs = [e for e in events if isinstance(e, AppError)]
    assert len(errs) == 1
    assert errs[0].code == "MUTE_SYNC_MDNS_FAILED"

    mute.stop()
    assert holder[0].stopped is True
    assert mute.active is False


def test_localhost_variant_starts():
    bus = EventBus()
    holder: list = []
    mute = MuteSync(
        MuteSyncConfig(),
        "localhost",
        bus,
        server_factory=_factory(holder),
        initial_fetch=lambda: None,
    )
    mute.start()
    assert mute.active is True
    assert holder[0].started is True
    mute.stop()


# -- MuteChanged transitions -----------------------------------------------


def test_mute_changed_only_on_transitions():
    bus = EventBus()
    events = _collect(bus)
    holder: list = []
    mute = MuteSync(
        MuteSyncConfig(),
        "127.0.0.1",
        bus,
        server_factory=_factory(holder),
        initial_fetch=lambda: None,
    )
    mute.start()
    srv = holder[0]

    srv.on_mute(True)
    srv.on_mute(True)  # repeat -> no second event
    changes = [e for e in events if isinstance(e, MuteChanged)]
    assert changes == [MuteChanged(True)]
    assert mute.muted is True

    srv.on_mute(False)  # real transition
    changes = [e for e in events if isinstance(e, MuteChanged)]
    assert changes == [MuteChanged(True), MuteChanged(False)]
    assert mute.muted is False

    mute.stop()


def test_initial_fetch_applies_when_no_push():
    bus = EventBus()
    events = _collect(bus)
    holder: list = []
    mute = MuteSync(
        MuteSyncConfig(),
        "127.0.0.1",
        bus,
        server_factory=_factory(holder),
        initial_fetch=lambda: True,
    )
    mute.start()

    assert _wait_until(lambda: mute.muted is True)
    changes = [e for e in events if isinstance(e, MuteChanged)]
    assert changes == [MuteChanged(True)]
    mute.stop()


def test_initial_fetch_does_not_override_push():
    bus = EventBus()
    events = _collect(bus)
    holder: list = []
    release = threading.Event()

    def slow_fetch():
        release.wait(2.0)
        return False

    mute = MuteSync(
        MuteSyncConfig(),
        "127.0.0.1",
        bus,
        server_factory=_factory(holder),
        initial_fetch=slow_fetch,
    )
    mute.start()
    srv = holder[0]

    srv.on_mute(True)  # push arrives before the slow fetch resolves
    assert mute.muted is True

    release.set()  # fetch now returns False -- must NOT override the push
    assert _wait_until(lambda: not mute._fetch_thread.is_alive())

    assert mute.muted is True
    changes = [e for e in events if isinstance(e, MuteChanged)]
    assert changes == [MuteChanged(True)]
    mute.stop()


def test_superseded_update_never_publishes():
    """Generation guard: if a push lands between another update's state
    mutation and its publish step, the superseded publish is suppressed --
    subscribers never see an event ordering that contradicts final state.

    Exercises the two halves of ``_update`` directly to pin the race
    deterministically: the fetch-apply mutates state first, the push mutates
    (and publishes) second, then the fetch's stale publish step runs last.
    """
    bus = EventBus()
    events = _collect(bus)
    holder: list = []
    mute = MuteSync(
        MuteSyncConfig(),
        "127.0.0.1",
        bus,
        server_factory=_factory(holder),
        initial_fetch=lambda: None,
    )
    mute.start()

    gen_fetch = mute._apply(False, is_push=False)  # fetch applies False first
    gen_push = mute._apply(True, is_push=True)  # push applies True second
    assert gen_fetch is not None
    assert gen_push is not None

    mute._publish_if_current(gen_push, True)  # push publishes (current)
    mute._publish_if_current(gen_fetch, False)  # stale fetch: suppressed

    assert mute.muted is True
    changes = [e for e in events if isinstance(e, MuteChanged)]
    assert changes == [MuteChanged(True)]
    mute.stop()


def test_update_after_stop_publishes_nothing():
    bus = EventBus()
    events = _collect(bus)
    holder: list = []
    mute = MuteSync(
        MuteSyncConfig(),
        "127.0.0.1",
        bus,
        server_factory=_factory(holder),
        initial_fetch=lambda: None,
    )
    mute.start()
    srv = holder[0]
    mute.stop()
    after_stop = [e for e in events if isinstance(e, MuteChanged)]

    srv.on_mute(True)  # late push after stop: no state change, no event

    assert mute.muted is None
    assert [e for e in events if isinstance(e, MuteChanged)] == after_stop


def test_late_initial_fetch_after_stop_publishes_nothing():
    bus = EventBus()
    events = _collect(bus)
    holder: list = []
    release = threading.Event()

    def slow_fetch():
        release.wait(2.0)
        return True

    mute = MuteSync(
        MuteSyncConfig(),
        "127.0.0.1",
        bus,
        server_factory=_factory(holder),
        initial_fetch=slow_fetch,
    )
    mute.start()
    mute.stop()  # stop while the fetch is still in flight
    after_stop = [e for e in events if isinstance(e, MuteChanged)]

    release.set()
    assert _wait_until(lambda: not mute._fetch_thread.is_alive())

    assert mute.muted is None
    assert [e for e in events if isinstance(e, MuteChanged)] == after_stop


# -- stop/restart: per-session state must not survive ------------------------


def test_stop_clears_state_so_a_restart_trusts_its_fresh_fetch():
    # Regression: a push in one session left _muted/_push_received behind, so
    # the next session's initial fetch was dropped by the fetch-after-push
    # guard and the coordinator resumed on a value VRChat may have changed
    # while sync was off (Settings enable off/on wedged the GUI on stale mute).
    bus = EventBus()
    events = _collect(bus)
    holder: list = []
    truth = [True]  # what VRChat would answer to the initial fetch
    mute = MuteSync(
        MuteSyncConfig(),
        "127.0.0.1",
        bus,
        server_factory=_factory(holder),
        initial_fetch=lambda: truth[0],
    )
    mute.start()
    holder[0].on_mute(True)  # a real push arms the fetch-after-push guard
    assert mute.muted is True
    mute.stop()
    assert mute.muted is None

    truth[0] = False  # user unmuted while sync was off; the push was lost
    mute.start()
    assert _wait_until(lambda: mute.muted is False)
    assert mute.should_caption() is True  # default "pause" mode, unmuted
    changes = [e.muted for e in events if isinstance(e, MuteChanged)]
    assert changes == [True, None, False]
    mute.stop()


def test_stop_of_an_active_session_publishes_unknown_state():
    # The GUI's mute chip has no other way to learn the state is no longer
    # knowable; MuteChanged(None) tells subscribers to drop the last value.
    bus = EventBus()
    events = _collect(bus)
    holder: list = []
    mute = MuteSync(
        MuteSyncConfig(),
        "127.0.0.1",
        bus,
        server_factory=_factory(holder),
        initial_fetch=lambda: None,
    )
    mute.start()
    holder[0].on_mute(True)
    mute.stop()
    changes = [e.muted for e in events if isinstance(e, MuteChanged)]
    assert changes == [True, None]


def test_stop_without_an_active_session_publishes_nothing():
    # stop() is also the shutdown hook for a coordinator that never started
    # (disabled config, non-localhost OSC); no session, no event.
    bus = EventBus()
    events = _collect(bus)
    holder: list = []
    mute = MuteSync(
        MuteSyncConfig(enabled=False),
        "127.0.0.1",
        bus,
        server_factory=_factory(holder),
        initial_fetch=lambda: None,
    )
    mute.start()  # disabled: never becomes active
    mute.stop()
    mute.stop()  # idempotent
    assert [e for e in events if isinstance(e, MuteChanged)] == []
