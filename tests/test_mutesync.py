"""Tests for the ``MuteSync`` state machine: should_caption modes and start
gating, driven with a fake OSCQuery server. MuteChanged transitions and
stop/restart state handling live in test_mutesync_transitions.py.
"""

import pytest

from vrcc.core.bus import EventBus
from vrcc.core.config import MuteSyncConfig
from vrcc.core.events import AppError, MuteChanged
from vrcc.osc.mutesync import MuteSync


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


def test_set_ip_updates_the_localhost_gate_for_the_next_start():
    # Regression: MuteSync cached _osc_ip at construction, so after changing
    # osc.ip and toggling mute sync off/on the gate in start() re-checked the
    # stale address. set_ip() must retarget the gate itself.
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

    mute.set_ip("192.168.1.50")
    mute.start()

    assert mute.active is False  # now-remote address is correctly refused
    errs = [e for e in events if isinstance(e, AppError)]
    assert len(errs) == 1
    assert errs[0].code == "MUTE_SYNC_REQUIRES_LOCALHOST"

    events.clear()
    mute.set_ip("127.0.0.1")
    mute.start()

    assert mute.active is True  # now-local address is correctly accepted
    assert [e for e in events if isinstance(e, AppError)] == []
    mute.stop()


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
