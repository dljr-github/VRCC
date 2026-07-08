"""Mute sync: the ``MuteSelf`` state machine.

Owns an :class:`~vrcc.osc.mutesync_server.OscQueryServer` (started only for
localhost OSC, since OSCQuery/mDNS is local-only) and tracks the latest
``MuteSelf`` value, publishing :class:`MuteChanged` on real transitions --
debouncing pushes vs. the initial fetch and guarding against races so
subscribers never see an event ordering that contradicts final state. The
OSCQuery wire/discovery layer itself (HTTP+OSC servers, mDNS browse) lives in
:mod:`vrcc.osc.mutesync_server`, which this module composes.
"""

from __future__ import annotations

import os
import threading
from typing import Callable

from vrcc.core.bus import EventBus
from vrcc.core.config import MuteSyncConfig
from vrcc.core.events import AppError, MuteChanged
from vrcc.osc.mutesync_server import OscQueryServer, fetch_initial_mute


def _default_server_factory(
    name: str, on_mute: Callable[[bool], None]
) -> OscQueryServer:
    return OscQueryServer(name, on_mute)


class MuteSync:
    """High-level mute-sync coordinator.

    Owns an :class:`OscQueryServer` (started only for localhost OSC, since
    OSCQuery/mDNS is local-only) and tracks the latest ``MuteSelf``, publishing
    :class:`MuteChanged` on real transitions. ``server_factory``/
    ``initial_fetch`` are injectable for tests.
    """

    def __init__(
        self,
        cfg: MuteSyncConfig,
        osc_ip: str,
        bus: EventBus,
        server_factory: Callable[[str, Callable[[bool], None]], object] | None = None,
        initial_fetch: Callable[[], "bool | None"] | None = None,
    ) -> None:
        self._cfg = cfg
        self._osc_ip = osc_ip
        self._bus = bus
        self._server_factory = server_factory or _default_server_factory
        self._initial_fetch = initial_fetch or fetch_initial_mute

        self._server = None
        self._active = False
        self._muted: bool | None = None
        self._push_received = False
        self._stopped = False
        # Bumped on every accepted state change; a publish only goes out if
        # its update is still the newest one (see _publish_if_current).
        self._generation = 0
        self._state_lock = threading.Lock()
        # Serializes recheck+publish so MuteChanged events reach subscribers in
        # the order state changed. RLock: a synchronous bus handler on the
        # publishing thread may call back in without self-deadlocking.
        self._publish_lock = threading.RLock()
        self._lifecycle_lock = threading.Lock()
        self._fetch_thread: threading.Thread | None = None

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        """Start syncing, subject to config + localhost gating. No-op if
        disabled or active. Non-localhost OSC publishes
        ``MUTE_SYNC_REQUIRES_LOCALHOST`` and stays inactive; otherwise start the
        server + a daemon initial-fetch, and a failed mDNS advertise publishes
        ``MUTE_SYNC_MDNS_FAILED`` but stays active.
        """
        with self._lifecycle_lock:
            if not self._cfg.enabled or self._active:
                return
            if self._osc_ip not in ("127.0.0.1", "localhost"):
                self._bus.publish(
                    AppError(
                        code="MUTE_SYNC_REQUIRES_LOCALHOST",
                        message=(
                            "Mute sync needs VRChat's OSC output on this "
                            "machine (127.0.0.1); it is off because OSC is "
                            "pointed at a remote host."
                        ),
                        detail=self._osc_ip,
                    )
                )
                return

            with self._state_lock:
                self._stopped = False

            server = self._server_factory(f"VRCC-{os.getpid()}", self._on_mute)
            mdns_ok = server.start()
            self._server = server
            self._active = True

            if not mdns_ok:
                self._bus.publish(
                    AppError(
                        code="MUTE_SYNC_MDNS_FAILED",
                        message=(
                            "Mute sync could not advertise itself on the "
                            "local network (mDNS/zeroconf); VRChat may not "
                            "detect it, so mute state might not update."
                        ),
                        detail="",
                    )
                )

            self._fetch_thread = threading.Thread(
                target=self._run_initial_fetch,
                name="MuteSyncInitialFetch",
                daemon=True,
            )
            self._fetch_thread.start()

    def stop(self) -> None:
        """Stop the OSCQuery server if running; idempotent. Flips the stopped
        flag so late updates (a straggler push or a resolving initial fetch)
        are dropped instead of mutating state or publishing post-stop.
        """
        with self._lifecycle_lock:
            with self._state_lock:
                self._stopped = True
            server = self._server
            self._server = None
            self._active = False
            if server is not None:
                server.stop()

    # -- state -------------------------------------------------------------

    def _run_initial_fetch(self) -> None:
        result = self._initial_fetch()
        if result is None:
            return
        self._update(bool(result), is_push=False)

    def _on_mute(self, value: bool) -> None:
        self._update(value, is_push=True)

    def _update(self, value: bool, is_push: bool) -> None:
        """Apply a new mute value, publishing :class:`MuteChanged` only on a
        real transition. An initial-fetch result (``is_push=False``) never
        overrides a push that already arrived, and updates arriving after
        :meth:`stop` are dropped entirely."""
        gen = self._apply(value, is_push)
        if gen is not None:
            self._publish_if_current(gen, value)

    def _apply(self, value: bool, is_push: bool) -> int | None:
        """State-mutation half of an update, under ``_state_lock``.

        Returns the new generation number if the value was accepted (a real
        transition that should be published), or None if the update was
        dropped (post-stop, fetch-after-push, or no-op repeat)."""
        with self._state_lock:
            if self._stopped:
                return None
            if is_push:
                self._push_received = True
            elif self._push_received:
                return None
            if self._muted == value:
                return None
            self._muted = value
            self._generation += 1
            return self._generation

    def _publish_if_current(self, gen: int, value: bool) -> None:
        """Publish half of an update: emit :class:`MuteChanged` only if ``gen``
        is still the newest applied state (else two updates could apply in one
        order but publish in the other, desyncing cached streams from
        :attr:`muted`). ``_publish_lock`` serializes recheck+publish; dispatch
        runs outside ``_state_lock`` since a handler may call back in.
        """
        with self._publish_lock:
            with self._state_lock:
                if gen != self._generation:
                    return  # superseded by a newer update: never publish
            self._bus.publish(MuteChanged(value))

    @property
    def muted(self) -> bool | None:
        return self._muted

    @property
    def active(self) -> bool:
        return self._active

    def should_caption(self) -> bool:
        """Whether captioning should run given mute state + mode: ``ignore``
        always True; ``pause`` captions while unmuted (``None`` -> unmuted);
        ``invert`` captions only while muted (``None`` -> False). Fail-open when
        the server isn't running (state unknowable) so ``invert`` can't
        silently kill all captioning.
        """
        if not self._active:
            return True
        mode = self._cfg.mode
        if mode == "ignore":
            return True
        muted = self._muted
        if mode == "invert":
            return muted is True
        # "pause" (and any unexpected mode): caption unless known-muted.
        return not muted
