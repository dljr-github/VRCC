"""Format captions/translations for VRChat's chatbox and send over OSC,
self-throttled. VRChat truncates the display at 144 chars and rate-limits
``/chatbox/input``, so :class:`ChatboxSender` self-throttles via a
:class:`TokenBucket` to stay inside it. Qt-free, lazy ``pythonosc``; the
caller gates on ``send_to_vrchat`` (every ``submit()`` here is sent).
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from typing import Callable, Protocol

from vrcc.core.bus import EventBus
from vrcc.core.config import OscConfig
from vrcc.core.events import ChatboxSent, TypingStateChanged
from vrcc.osc.chatbox_format import (  # noqa: F401 -- re-exported, see below
    CHATBOX_LIMIT,
    fit_chatbox,
    fit_message,
    format_message,
)

logger = logging.getLogger("vrcc.osc")

# fit_chatbox/fit_message/format_message/CHATBOX_LIMIT live in
# chatbox_format.py (pure text shaping, no OSC/threads) and are re-exported
# here so existing `from vrcc.osc.chatbox import ...` call sites keep working.

# Cap on each wait-for-token sleep so stop() stays responsive instead of
# committing to one long sleep.
_MAX_POLL_SLICE_S = 0.1
# Idle poll: the worker wakes early via an Event on submit()/stop(), so this
# is just a safety-net re-check period.
_IDLE_POLL_S = 0.05
_JOIN_TIMEOUT_S = 2.0
# Hard cap on queued chunks: with coalescing off, sustained speech outpaces
# the drain rate, so depth/lag would grow unbounded. Overflow drops the
# OLDEST (deque maxlen) -- newest speech is most useful.
_QUEUE_MAX = 64


class _OscClient(Protocol):
    def send_message(self, address: str, value) -> None: ...


def _default_client_factory(ip: str, port: int) -> _OscClient:
    from pythonosc.udp_client import SimpleUDPClient

    return SimpleUDPClient(ip, port)


class TokenBucket:
    """Continuous-refill token bucket for client-side rate limiting.

    Starts full (first burst never waits), refills one token per
    ``refill_interval_s`` (fractional, capped at ``capacity``). ``clock``
    defaults to `time.monotonic`; tests inject a fake to drive refill.
    """

    def __init__(
        self,
        capacity: int,
        refill_interval_s: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._capacity = capacity
        self._refill_interval_s = refill_interval_s
        self._clock = clock
        self._tokens = float(capacity)
        self._last_refill = clock()
        # try_acquire/seconds_until_token run on the sender's worker thread
        # while reconfigure() runs on the GUI thread, so all token + rate state
        # is guarded by one lock (the token math is a read-modify-write, not a
        # single atomic int, so the GIL alone is not enough here).
        self._lock = threading.Lock()

    def _refill(self) -> None:  # caller holds self._lock
        now = self._clock()
        elapsed = now - self._last_refill
        if elapsed <= 0:
            return
        self._last_refill = now
        if self._refill_interval_s > 0:
            self._tokens = min(
                self._capacity, self._tokens + elapsed / self._refill_interval_s
            )
        else:
            self._tokens = float(self._capacity)

    def try_acquire(self) -> bool:
        """Consume one token if available now. Returns whether it did."""
        with self._lock:
            self._refill()
            if self._tokens >= 1.0:
                self._tokens -= 1.0
                return True
            return False

    def seconds_until_token(self) -> float:
        """How much longer until `try_acquire()` would succeed; ``0.0`` if
        it would succeed right now."""
        with self._lock:
            self._refill()
            if self._tokens >= 1.0:
                return 0.0
            return (1.0 - self._tokens) * self._refill_interval_s

    def reconfigure(self, capacity: int, refill_interval_s: float) -> None:
        """Retune capacity/refill live without handing out a free burst.

        Time accrued so far is materialized at the OLD rate first: switching
        the rate with a stale ``_last_refill`` would re-price the elapsed span
        at the new rate, and a shrink of the interval could mint the whole
        burst instantly. Tokens then clamp down to the new capacity (never
        up), so a larger capacity does not grant tokens and a smaller one
        takes them away."""
        with self._lock:
            self._refill()
            self._capacity = capacity
            self._refill_interval_s = refill_interval_s
            if self._tokens > capacity:
                self._tokens = float(capacity)


class ChatboxSender:
    """Daemon worker draining a queue of chatbox chunks through a
    :class:`TokenBucket`, sending each over OSC and publishing
    :class:`~vrcc.core.events.ChatboxSent`. ``client_factory``/``clock``/
    ``sleep`` are injectable for tests. All public methods are thread-safe.
    """

    def __init__(
        self,
        cfg: OscConfig,
        bus: EventBus,
        client_factory: Callable[[str, int], _OscClient] | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._cfg = cfg
        self._bus = bus
        self._client_factory = client_factory or _default_client_factory
        self._clock = clock
        self._sleep = sleep

        self._client: _OscClient = self._client_factory(cfg.ip, cfg.port)
        self._client_lock = threading.Lock()

        self._bucket = TokenBucket(cfg.burst, cfg.min_interval_s, clock=clock)

        # Bounded: on overflow the deque's maxlen drops the oldest chunks
        # (logged once per sender; see submit()). Last field: partial (a
        # tentative live-caption chunk, see submit_partial()).
        self._queue: deque[tuple[str, int, bool, float, bool]] = deque(maxlen=_QUEUE_MAX)
        self._overflow_logged = False
        self._queue_lock = threading.Lock()
        self._wake = threading.Event()

        self._typing_lock = threading.Lock()
        self._last_typing: bool | None = None

        self._lifecycle_lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop_flag = threading.Event()

    # -- lifecycle -----------------------------------------------------

    def start(self) -> None:
        """Start the worker thread. A no-op if already running."""
        with self._lifecycle_lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop_flag.clear()
            thread = threading.Thread(
                target=self._run, name="ChatboxSender", daemon=True
            )
            self._thread = thread
            thread.start()

    def stop(self) -> None:
        """Stop the worker thread (join, 2 s) and best-effort turn typing off.
        Safe to call more than once and before `start()`."""
        with self._lifecycle_lock:
            thread = self._thread
            self._thread = None
            if thread is not None:
                self._stop_flag.set()
                self._wake.set()
                thread.join(timeout=_JOIN_TIMEOUT_S)
        self.set_typing(False)

    # -- producing messages ----------------------------------------------

    def submit(self, text: str, utterance_id: int) -> None:
        """Queue `text` (already-formatted) for sending; empty is ignored. It's
        fit to the chatbox limit via `fit_chatbox` (may split into several
        chunks sharing `utterance_id`). ``cfg.coalesce_latest_wins`` replaces
        whatever is queued and unsent; otherwise chunks are appended FIFO,
        dropping the oldest past ``_QUEUE_MAX`` (logged once). When a message
        splits into multiple chunks, every chunk but the last carries
        ``cfg.split_delay_s`` as its post-send pause (read live, not cached)
        so each part is actually readable before VRChat's chatbox replaces it
        with the next one; a single chunk never delays.
        """
        if not text or not text.strip():
            return
        chunks = fit_chatbox(text, self._cfg.overflow)
        if not chunks:
            return
        # Flag over-limit messages (unless "split", which loses nothing) so the
        # caption log can badge them as truncated.
        truncated = len(text) > CHATBOX_LIMIT and self._cfg.overflow != "split"
        self._enqueue(chunks, utterance_id, truncated)

    def submit_message(
        self, original: str, translations: list[tuple[str, str]], utterance_id: int
    ) -> None:
        """Queue a caption plus its ``[(name, text), ...]`` translations,
        shaped by `fit_message`: in "split" mode each queued part carries a
        balanced slice of EVERY language rather than a greedy cut of the
        joined string. Queueing semantics (coalescing, capping, per-chunk
        delay, truncated badging) match `submit()` exactly.
        """
        chunks = fit_message(original, translations, self._cfg)
        if not chunks:
            return
        joined = format_message(original, translations, self._cfg)
        truncated = self._cfg.overflow != "split" and len(joined) > CHATBOX_LIMIT
        self._enqueue(chunks, utterance_id, truncated)

    def submit_partial(self, text: str) -> None:
        """Queue a tentative, in-progress transcription: same fit-to-144 +
        coalesce-latest-wins + token-bucket path as `submit`, but the queued
        chunk is tagged so `_send_chunk` sends it over OSC WITHOUT publishing
        `ChatboxSent` -- a partial never marks the caption log row delivered.
        A later firmed-up `submit_message`/`submit` for the same utterance
        coalesces over it exactly like any other queued chunk.
        """
        if not text or not text.strip():
            return
        chunks = fit_chatbox(text, self._cfg.overflow)
        if not chunks:
            return
        truncated = len(text) > CHATBOX_LIMIT and self._cfg.overflow != "split"
        self._enqueue(chunks, 0, truncated, partial=True)

    def _enqueue(
        self,
        chunks: list[str],
        utterance_id: int,
        truncated: bool,
        partial: bool = False,
    ) -> None:
        split_delay_s = self._cfg.split_delay_s
        last = len(chunks) - 1
        items = [
            (chunk, utterance_id, truncated, split_delay_s if i < last else 0.0, partial)
            for i, chunk in enumerate(chunks)
        ]
        with self._queue_lock:
            if self._cfg.coalesce_latest_wins:
                self._queue.clear()
            overflowing = len(self._queue) + len(items) > _QUEUE_MAX
            self._queue.extend(items)
        if overflowing and not self._overflow_logged:
            self._overflow_logged = True
            logger.warning(
                "chatbox queue exceeded %d pending chunks; dropping oldest "
                "(sends can't keep up -- further drops are not logged)",
                _QUEUE_MAX,
            )
        self._wake.set()

    def set_typing(self, typing: bool) -> None:
        """Send ``/chatbox/typing [typing]`` immediately (bypasses the queue).
        Repeated same-value calls are deduped; publishes `TypingStateChanged`
        on a state change."""
        with self._typing_lock:
            if self._last_typing == typing:
                return
            client = self._get_client()
            try:
                client.send_message("/chatbox/typing", [typing])
            except OSError:
                logger.warning(
                    "failed to send chatbox typing state (VRChat likely not "
                    "running); dropping",
                    exc_info=True,
                )
                return
            self._last_typing = typing
        self._bus.publish(TypingStateChanged(typing))

    def reconfigure(self, ip: str, port: int) -> None:
        """Atomically swap the OSC client for one pointed at `ip`/`port`.
        Safe to call while the worker thread is running."""
        new_client = self._client_factory(ip, port)
        with self._client_lock:
            self._client = new_client

    def reconfigure_rate(self, burst: int, min_interval_s: float) -> None:
        """Retune the send throttle live to ``burst``/``min_interval_s`` without
        granting a fresh burst. The bucket is internally locked, so this is safe
        while the worker thread is draining."""
        self._bucket.reconfigure(burst, min_interval_s)

    # -- worker ------------------------------------------------------------

    def _get_client(self) -> _OscClient:
        with self._client_lock:
            return self._client

    def _pop_next(self) -> tuple[str, int, bool, float, bool] | None:
        with self._queue_lock:
            if self._queue:
                return self._queue.popleft()
            return None

    def _run(self) -> None:
        while not self._stop_flag.is_set():
            item = self._pop_next()
            if item is None:
                self._wake.wait(timeout=_IDLE_POLL_S)
                self._wake.clear()
                continue
            if not self._wait_for_token():
                continue  # stop requested while waiting; drop this item
            text, utterance_id, truncated, delay_after, partial = item
            self._send_chunk(text, utterance_id, truncated, partial)
            # Wait after the send attempt regardless of whether it actually
            # succeeded (VRChat may be offline -- rare, and the pacing goal is
            # about readability, not about the ack we don't get over OSC
            # anyway). `Event.wait` both provides the pause and makes stop()
            # responsive: it returns True immediately if stop() sets the flag
            # mid-wait, so we bail out of the loop instead of popping again.
            if delay_after > 0 and self._stop_flag.wait(delay_after):
                break

    def _wait_for_token(self) -> bool:
        """Block (via the injected `sleep`) until a token is available.
        Returns False if `stop()` was requested before one became
        available."""
        while True:
            if self._bucket.try_acquire():
                return True
            if self._stop_flag.is_set():
                return False
            remaining = self._bucket.seconds_until_token()
            slice_s = (
                min(remaining, _MAX_POLL_SLICE_S) if remaining > 0 else _MAX_POLL_SLICE_S
            )
            self._sleep(slice_s)

    def _send_chunk(
        self, text: str, utterance_id: int, truncated: bool = False, partial: bool = False
    ) -> None:
        client = self._get_client()
        try:
            client.send_message(
                "/chatbox/input", [text, True, self._cfg.notification_sfx]
            )
        except OSError:
            # UDP is fire-and-forget; VRChat may simply not be running.
            # Best-effort by design: log and drop, no retry loop.
            logger.warning(
                "failed to send chatbox message (VRChat likely not running); "
                "dropping",
                exc_info=True,
            )
            return
        if not partial:
            # A partial is tentative: it must never mark the caption log row
            # delivered, so it skips this publish entirely.
            self._bus.publish(ChatboxSent(text, utterance_id, truncated))
