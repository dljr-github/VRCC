"""One swappable engine behind one lock, shared by every caller that needs it.

The invariant an engine slot enforces: a swap can never land mid-call, and a
caller that only wants to know whether an engine is installed right now must
never be made to wait behind one that is already running (the GUI thread
polling mt_active is the case that made this a hard rule, not a nicety).
"""

from __future__ import annotations

import contextlib
import threading
from typing import Generic, Iterator, TypeVar

_T = TypeVar("_T")


class EngineSlot(Generic[_T]):
    """Holds one engine (or ``None``). ``swap``/``borrow`` serialise on the
    same lock, so an engine is never unloaded while a call on it is in
    flight; ``current`` reads without the lock for a caller that only tests
    truthiness and must not queue behind an in-flight call."""

    def __init__(self, engine: "_T | None" = None) -> None:
        self._engine = engine
        self._lock = threading.Lock()

    def swap(self, new: "_T | None") -> "_T | None":
        with self._lock:
            old, self._engine = self._engine, new
            return old

    @contextlib.contextmanager
    def borrow(self, *, blocking: bool = True) -> Iterator["_T | None"]:
        """Yield the current engine with the lock held for the caller's
        whole call, so a concurrent swap waits for it to return.

        Optional work may use ``blocking=False``: an occupied slot yields
        None immediately, just like an absent engine, without releasing the
        owner's lock. Foreground callers keep the blocking default.
        """
        if not self._lock.acquire(blocking=blocking):
            yield None
            return
        try:
            yield self._engine
        finally:
            self._lock.release()

    @property
    def busy(self) -> bool:
        """Advisory admission check; nonblocking borrow closes the race."""
        return self._lock.locked()

    @property
    def current(self) -> "_T | None":
        return self._engine
