"""Single-instance guard: one VRCC per logon session.

A named mutex is the authority and its existence is the whole signal; the
process never takes ownership, so there is nothing to release but a handle.
Windows destroys the object with its last handle, so a crash or the os._exit
path at vrcc/app.py:481-488 cannot leave the guard stuck. That is why this is a
kernel object and not a lock file.

A separate auto-reset event is the doorbell. A refused launch signals it and
exits; the running copy polls it and raises its window. Auto-reset means a
burst of clicks raises the window once rather than queueing a stack of raises.

ctypes rather than a dependency: this module is imported before anything heavy,
and its cost has to stay near zero.
"""

from __future__ import annotations

import ctypes
import logging
import os
import sys

logger = logging.getLogger("vrcc.core.instance")

ERROR_ALREADY_EXISTS = 183
_WAIT_OBJECT_0 = 0

_DEFAULT_NAME = "VRCC.SingleInstance.1"

# ctypes itself is stdlib everywhere; only WinDLL and wintypes are Windows only.
# Keeping the plain import at module scope means _create_mutex and _create_event
# resolve ctypes.get_last_error on any platform, even though acquire() returns
# before calling them off Windows.
if sys.platform == "win32":  # pragma: no branch - the app is Windows only
    from ctypes import wintypes

    _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

    _kernel32.CreateMutexW.argtypes = (
        wintypes.LPVOID,
        wintypes.BOOL,
        wintypes.LPCWSTR,
    )
    _kernel32.CreateMutexW.restype = wintypes.HANDLE
    _kernel32.CreateEventW.argtypes = (
        wintypes.LPVOID,
        wintypes.BOOL,
        wintypes.BOOL,
        wintypes.LPCWSTR,
    )
    _kernel32.CreateEventW.restype = wintypes.HANDLE
    _kernel32.SetEvent.argtypes = (wintypes.HANDLE,)
    _kernel32.SetEvent.restype = wintypes.BOOL
    _kernel32.WaitForSingleObject.argtypes = (wintypes.HANDLE, wintypes.DWORD)
    _kernel32.WaitForSingleObject.restype = wintypes.DWORD
    _kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
    _kernel32.CloseHandle.restype = wintypes.BOOL
else:  # pragma: no cover - only reached off Windows
    _kernel32 = None


def allow_multiple() -> bool:
    """Escape hatch for running two copies deliberately. Documented in
    DEVELOPING.md only: it exists for development, not for users."""
    return os.environ.get("VRCC_ALLOW_MULTIPLE", "") == "1"


def _create_mutex(name: str):
    """Returns (handle, last_error). Split out so a test can force the
    fail-open branch without a second process."""
    handle = _kernel32.CreateMutexW(None, False, name)
    return handle, ctypes.get_last_error()


def _create_event(name: str):
    # Auto-reset, initially clear: one ring is consumed by one poll.
    handle = _kernel32.CreateEventW(None, False, False, name)
    return handle, ctypes.get_last_error()


def _close_handle(handle) -> None:
    if handle:
        _kernel32.CloseHandle(handle)


class InstanceGuard:
    """Decides whether this process may run, and carries the doorbell."""

    def __init__(self, name: str = _DEFAULT_NAME) -> None:
        # Local\ scopes the objects to the logon session, so two users on one
        # machine do not block each other.
        self.mutex_name = f"Local\\{name}.mutex"
        self.event_name = f"Local\\{name}.doorbell"
        self._mutex = None
        self._event = None

    def acquire(self) -> bool:
        """True when this process may run.

        Only ERROR_ALREADY_EXISTS refuses. Any other failure, including
        ERROR_ACCESS_DENIED under a locked-down policy, runs the app: a guard
        that cannot be read must never be the reason VRCC will not start.
        """
        if _kernel32 is None or allow_multiple():
            return True
        # Doorbell first, so a process that loses the mutex race still has
        # somewhere to ring.
        self._event, _ = _create_event(self.event_name)
        handle, err = _create_mutex(self.mutex_name)
        if err == ERROR_ALREADY_EXISTS:
            _close_handle(handle)
            return False
        if not handle:
            logger.warning("single-instance guard unavailable (error %s)", err)
            return True
        self._mutex = handle
        return True

    def ring(self) -> bool:
        """Ask the copy already running to show itself. True when the request
        was delivered; the raise itself is the other process's business."""
        if self._event is None:
            return False
        return bool(_kernel32.SetEvent(self._event))

    def doorbell_rang(self) -> bool:
        """Non-blocking poll. Consumes the ring, because the event is
        auto-reset."""
        if self._event is None:
            return False
        return _kernel32.WaitForSingleObject(self._event, 0) == _WAIT_OBJECT_0

    def release(self) -> None:
        """Safe before acquire and safe twice. The kernel would do this at
        process exit anyway; doing it explicitly frees the name the moment a
        clean shutdown finishes."""
        _close_handle(self._mutex)
        _close_handle(self._event)
        self._mutex = None
        self._event = None
