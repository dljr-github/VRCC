"""Bring the running copy forward when a second launch rings the doorbell.

Kept apart from vrcc.core.instance because that module must stay importable
before Qt exists; this half is Qt only and never touches a kernel handle.
"""

from __future__ import annotations

import logging

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import QMainWindow

logger = logging.getLogger("vrcc.gui.raise_window")


def pick_raise_target(app):
    """The window a ring should bring forward, or None when nothing is up yet.

    A modal beats the main window: during the first-run wizard the main window
    does not exist, and once it does, raising it from behind a modal hands the
    user a window they cannot click. The last fallback covers the boot panel
    and any other bare top level.
    """
    modal = app.activeModalWidget()
    if modal is not None and modal.isVisible():
        return modal
    for widget in app.topLevelWidgets():
        if isinstance(widget, QMainWindow) and widget.isVisible():
            return widget
    for widget in app.topLevelWidgets():
        if widget.isVisible():
            return widget
    return None


def raise_window(widget) -> None:
    """Un-minimise, stack on top, ask for focus.

    activateWindow can be refused: Windows revokes foreground rights once the
    user has touched another application, and the taskbar button flashes
    instead. Nothing here may promise the window comes forward.
    """
    widget.setWindowState(
        (widget.windowState() & ~Qt.WindowState.WindowMinimized)
        | Qt.WindowState.WindowActive
    )
    widget.show()
    widget.raise_()
    widget.activateWindow()


def install_raise_watch(app, guard, interval_ms: int = 250):
    """Poll the doorbell and raise on a ring. Returns the timer so a caller can
    stop it; the app owns it for the process lifetime otherwise.

    A poll rather than a waiter thread: the raise has to happen on the Qt
    thread, and a 250 ms tick costs nothing next to what it is waiting for.
    """

    def _tick() -> None:
        try:
            if not guard.doorbell_rang():
                return
            target = pick_raise_target(app)
            if target is None:
                # Rung during the import walk, before anything is on screen.
                return
            raise_window(target)
        except Exception:  # noqa: BLE001 -- a timer must never kill the app
            logger.debug("raise watch tick failed", exc_info=True)

    timer = QTimer(app)
    timer.setInterval(interval_ms)
    timer.timeout.connect(_tick)
    timer.start()
    return timer
