"""Detect whether a VRChat client with OSC is on the local network.

VRChat advertises an OSCQuery service (``VRChat-Client-*``) over mDNS when OSC is
on -- the closest proxy for "is the chatbox reachable?". :class:`VrchatDetector`
browses via ``zeroconf`` (lazy, injectable) and publishes on presence changes.
"""

from __future__ import annotations

import logging
import threading

from vrcc.core.bus import EventBus
from vrcc.core.events import VrchatDetected

logger = logging.getLogger("vrcc.osc.vrchat_detect")

_VRCHAT_PREFIX = "VRChat-Client"
_SERVICE_TYPE = "_oscjson._tcp.local."


class VrchatDetector:
    """Publishes ``VrchatDetected`` as VRChat's OSCQuery service comes and goes.

    ``browser_factory``/``zeroconf_factory`` are injectable for tests (browser
    needs ``cancel()``, Zeroconf needs ``close()``).
    """

    def __init__(
        self, bus: EventBus, zeroconf_factory=None, browser_factory=None
    ) -> None:
        self._bus = bus
        self._zeroconf_factory = zeroconf_factory
        self._browser_factory = browser_factory
        self._present: set[str] = set()
        self._detected = False
        self._lock = threading.Lock()
        self._zc = None
        self._browser = None

    @property
    def detected(self) -> bool:
        return self._detected

    def start(self) -> None:
        """Begin browsing. Publishes the initial (not-detected) state so the UI
        can render immediately, then transitions as services appear/disappear.
        Never raises: an mDNS failure just means detection stays off."""
        self._publish(False)  # initial UI state
        try:
            if self._zeroconf_factory is not None:
                self._zc = self._zeroconf_factory()
                make_browser = self._browser_factory
            else:  # pragma: no cover - real mDNS path, exercised manually
                from zeroconf import ServiceBrowser, Zeroconf

                self._zc = Zeroconf()
                make_browser = self._browser_factory or ServiceBrowser
            self._browser = make_browser(self._zc, _SERVICE_TYPE, self)
        except Exception:  # noqa: BLE001 -- detection is best-effort
            logger.warning("VRChat detection unavailable (mDNS failed)", exc_info=True)

    def republish(self) -> None:
        """Re-announce the current state. ``VrchatDetected`` fires only on
        presence transitions, so a subscriber attached after the last one (a
        main window rebuilt on a UI-language change) would otherwise wait on
        "checking" until VRChat's mDNS record genuinely cycles."""
        self._publish(self._detected)

    def stop(self) -> None:
        for closer in (
            lambda: self._browser and self._browser.cancel(),
            lambda: self._zc and self._zc.close(),
        ):
            try:
                closer()
            except Exception:  # noqa: BLE001 -- teardown must never raise
                logger.debug("VRChat detector teardown error", exc_info=True)
        self._browser = None
        self._zc = None
        # Reset presence so a later start() re-publishes cleanly (a stale
        # _detected=True would suppress the next add_service transition and
        # wedge the UI).
        with self._lock:
            self._present.clear()
            self._detected = False

    # -- zeroconf ServiceListener interface --------------------------------

    def add_service(self, zeroconf, service_type, name) -> None:  # noqa: N802 (Qt/zc API)
        self._update(name, present=True)

    def remove_service(self, zeroconf, service_type, name) -> None:  # noqa: N802
        self._update(name, present=False)

    def update_service(self, zeroconf, service_type, name) -> None:  # noqa: N802
        # A record refresh doesn't change presence; nothing to do.
        pass

    # -- internals ---------------------------------------------------------

    def _update(self, name: str, present: bool) -> None:
        if not name.startswith(_VRCHAT_PREFIX):
            return
        with self._lock:
            if present:
                self._present.add(name)
            else:
                self._present.discard(name)
            detected = bool(self._present)
            changed = detected != self._detected
            self._detected = detected
        if changed:
            self._publish(detected)

    def _publish(self, detected: bool) -> None:
        self._bus.publish(VrchatDetected(detected))
