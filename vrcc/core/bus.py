"""Synchronous, thread-safe publish/subscribe event bus.

Components publish events; subscribers match on the exact event type (no
isinstance). Dispatch is synchronous on the publishing thread.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Callable

from vrcc.core.events import AppError

logger = logging.getLogger("vrcc.bus")


class EventBus:
    def __init__(self) -> None:
        self._handlers: dict[type, list[Callable[[Any], None]]] = {}
        self._lock = threading.RLock()

    def subscribe(self, event_type: type, handler: Callable[[Any], None]) -> Callable[[], None]:
        """Register `handler` for `event_type`. Returns a zero-arg callable
        that unsubscribes it; safe to call more than once."""
        with self._lock:
            self._handlers.setdefault(event_type, []).append(handler)

        def unsubscribe() -> None:
            with self._lock:
                handlers = self._handlers.get(event_type)
                if handlers is not None and handler in handlers:
                    handlers.remove(handler)

        return unsubscribe

    def publish(self, event: Any) -> None:
        """Dispatch `event` synchronously to handlers of its exact type. A
        raising handler doesn't stop the rest; the error is logged and
        re-published as an `AppError`, unless `event` is itself an `AppError`
        (no recursion)."""
        event_type = type(event)
        with self._lock:
            handlers = list(self._handlers.get(event_type, ()))

        for handler in handlers:
            try:
                handler(event)
            except Exception as exc:
                logger.exception(
                    "Handler %r raised while handling %r", handler, event
                )
                if not isinstance(event, AppError):
                    self.publish(
                        AppError(
                            code="HANDLER_ERROR",
                            message=str(exc),
                            detail=(
                                f"{event_type.__name__} handler {handler!r} "
                                f"raised {type(exc).__name__}"
                            ),
                        )
                    )
