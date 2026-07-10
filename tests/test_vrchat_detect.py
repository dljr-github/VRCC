"""Tests for the VRChat OSCQuery/mDNS presence detector (state machine only;
the real zeroconf browser is injected)."""

from __future__ import annotations

from vrcc.core.bus import EventBus
from vrcc.core.events import VrchatDetected
from vrcc.osc.vrchat_detect import VrchatDetector


class _FakeBrowser:
    def __init__(self, zc, service_type, listener) -> None:
        self.listener = listener
        self.cancelled = False

    def cancel(self) -> None:
        self.cancelled = True


class _FakeZeroconf:
    def close(self) -> None:
        pass


def _detector(bus):
    holder = {}

    def browser_factory(zc, service_type, listener):
        holder["browser"] = _FakeBrowser(zc, service_type, listener)
        return holder["browser"]

    det = VrchatDetector(
        bus, zeroconf_factory=_FakeZeroconf, browser_factory=browser_factory
    )
    return det, holder


def test_publishes_not_detected_on_start_then_detected_when_vrchat_appears():
    bus = EventBus()
    events: list[VrchatDetected] = []
    bus.subscribe(VrchatDetected, events.append)

    det, _ = _detector(bus)
    det.start()
    assert events[-1].detected is False  # initial UI state

    det.add_service(None, "_oscjson._tcp.local.", "VRChat-Client-1234._oscjson._tcp.local.")
    assert det.detected is True
    assert events[-1].detected is True


def test_non_vrchat_services_are_ignored():
    bus = EventBus()
    events: list[VrchatDetected] = []
    bus.subscribe(VrchatDetected, events.append)
    det, _ = _detector(bus)
    det.start()
    before = len(events)
    det.add_service(None, "_oscjson._tcp.local.", "VRCC-9999._oscjson._tcp.local.")
    assert det.detected is False
    assert len(events) == before  # no transition, no event


def test_removal_transitions_back_to_not_detected_once():
    bus = EventBus()
    events: list[VrchatDetected] = []
    bus.subscribe(VrchatDetected, events.append)
    det, _ = _detector(bus)
    det.start()
    name = "VRChat-Client-1._oscjson._tcp.local."
    det.add_service(None, "_oscjson._tcp.local.", name)
    det.add_service(None, "_oscjson._tcp.local.", name)  # duplicate: no extra event
    det.remove_service(None, "_oscjson._tcp.local.", name)
    assert det.detected is False
    # events: initial False, True (first add), False (remove) -- add dupe is a no-op
    assert [e.detected for e in events] == [False, True, False]


def test_republish_reannounces_current_state_without_a_transition():
    # A window rebuilt on a UI-language change subscribes after the last
    # transition, and VrchatDetected fires only on transitions; republish()
    # lets the composition root feed the current state to a late subscriber.
    bus = EventBus()
    events: list[VrchatDetected] = []
    bus.subscribe(VrchatDetected, events.append)
    det, _ = _detector(bus)
    det.start()
    name = "VRChat-Client-2._oscjson._tcp.local."
    det.add_service(None, "_oscjson._tcp.local.", name)
    assert [e.detected for e in events] == [False, True]

    det.republish()
    assert [e.detected for e in events] == [False, True, True]

    det.remove_service(None, "_oscjson._tcp.local.", name)
    det.republish()
    assert [e.detected for e in events][-2:] == [False, False]


def test_stop_cancels_browser_and_is_safe_before_start():
    bus = EventBus()
    det, holder = _detector(bus)
    det.stop()  # before start: no browser, must not raise
    det.start()
    det.stop()
    assert holder["browser"].cancelled is True
