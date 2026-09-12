"""Tests for doorbell-driven window raising, offscreen."""

from __future__ import annotations

import os
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication, QDialog, QMainWindow, QWidget

from vrcc.gui.raise_window import (
    install_raise_watch,
    pick_raise_target,
    raise_window,
)


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


class _Guard:
    """Stands in for InstanceGuard: the watch only ever asks it one question."""

    def __init__(self, rings: int = 0) -> None:
        self.rings = rings
        self.polls = 0

    def doorbell_rang(self) -> bool:
        self.polls += 1
        if self.rings:
            self.rings -= 1
            return True
        return False


def _pump(app, done, timeout: float = 1.0) -> None:
    """Process events until done() holds or timeout passes.

    A bare processEvents loop with no sleep can finish 50 iterations in a
    fraction of a millisecond, faster than a 1 ms QTimer interval can become
    due, so the timer never fires within a fixed iteration count. Spacing
    the calls with a short sleep gives real wall clock time a chance to pass.
    """
    deadline = time.monotonic() + timeout
    while not done() and time.monotonic() < deadline:
        time.sleep(0.001)
        app.processEvents()


def test_no_visible_window_picks_nothing(qapp):
    assert pick_raise_target(qapp) is None


def test_picks_the_visible_main_window(qapp):
    w = QMainWindow()
    w.show()
    try:
        assert pick_raise_target(qapp) is w
    finally:
        w.close()
        w.deleteLater()


def test_skips_a_hidden_main_window(qapp):
    w = QMainWindow()
    try:
        assert pick_raise_target(qapp) is None
    finally:
        w.deleteLater()


def test_falls_back_to_any_visible_top_level(qapp):
    """During the first-run wizard there is no QMainWindow at all, so the
    fallback is what makes a second click do anything."""
    w = QWidget()
    w.show()
    try:
        assert pick_raise_target(qapp) is w
    finally:
        w.close()
        w.deleteLater()


def test_modal_wins_over_the_main_window(qapp):
    """Raising the window behind a modal would hand the user something they
    cannot click."""
    main = QMainWindow()
    main.show()
    dlg = QDialog()
    dlg.setModal(True)
    dlg.show()
    qapp.processEvents()
    try:
        # Asserted flatly, not as a disjunction with activeModalWidget(). If
        # offscreen QPA does not register modality, this must fail loudly here
        # rather than pass while proving nothing about the modal case.
        assert pick_raise_target(qapp) is dlg
    finally:
        dlg.close()
        dlg.deleteLater()
        main.close()
        main.deleteLater()


def test_raise_window_clears_minimised_and_does_not_raise(qapp):
    from PySide6.QtCore import Qt

    w = QMainWindow()
    w.show()
    w.setWindowState(Qt.WindowState.WindowMinimized)
    try:
        raise_window(w)
        assert not (w.windowState() & Qt.WindowState.WindowMinimized)
    finally:
        w.close()
        w.deleteLater()


def test_watch_polls_the_guard(qapp):
    guard = _Guard()
    timer = install_raise_watch(qapp, guard, interval_ms=1)
    try:
        _pump(qapp, lambda: guard.polls > 0)
        assert guard.polls > 0
    finally:
        timer.stop()
        timer.deleteLater()


def test_watch_raises_on_a_ring(qapp):
    guard = _Guard(rings=1)
    w = QMainWindow()
    w.show()
    timer = install_raise_watch(qapp, guard, interval_ms=1)
    try:
        _pump(qapp, lambda: guard.rings == 0)
        assert guard.rings == 0
    finally:
        timer.stop()
        timer.deleteLater()
        w.close()
        w.deleteLater()


def test_watch_survives_a_ring_with_no_window(qapp):
    """A ring during the import walk, before any window exists, must not
    raise an exception on the timer thread."""
    guard = _Guard(rings=1)
    timer = install_raise_watch(qapp, guard, interval_ms=1)
    try:
        _pump(qapp, lambda: guard.rings == 0)
        assert guard.rings == 0
    finally:
        timer.stop()
        timer.deleteLater()
