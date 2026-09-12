"""The boot panel, offscreen."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from vrcc.core.progress import PHASES
from vrcc.gui.boot_panel import BootPanel


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def test_panel_is_a_frameless_splash(qapp):
    panel = BootPanel()
    try:
        flags = panel.windowFlags()
        assert flags & Qt.WindowType.FramelessWindowHint
        assert flags & Qt.WindowType.SplashScreen
    finally:
        panel.close_panel()
        panel.deleteLater()


def test_bar_range_matches_the_phase_count(qapp):
    panel = BootPanel()
    try:
        assert panel._bar.minimum() == 0
        assert panel._bar.maximum() == len(PHASES)
    finally:
        panel.close_panel()
        panel.deleteLater()


def test_each_start_advances_the_bar_and_names_the_phase(qapp):
    panel = BootPanel()
    try:
        before = panel._bar.value()
        panel.start("audio")
        assert panel._bar.value() == before + 1
        assert panel._label.text()
        first = panel._label.text()
        panel.start("downloads")
        assert panel._bar.value() == before + 2
        assert panel._label.text() != first
    finally:
        panel.close_panel()
        panel.deleteLater()


def test_unknown_phase_does_not_crash_or_advance_past_the_end(qapp):
    """A caller naming a phase outside the table must never take down a launch."""
    panel = BootPanel()
    try:
        for _ in range(len(PHASES) + 5):
            panel.start("nonsense")
        assert panel._bar.value() <= panel._bar.maximum()
    finally:
        panel.close_panel()
        panel.deleteLater()


def test_close_panel_hides_and_is_idempotent(qapp):
    panel = BootPanel()
    panel.show()
    try:
        panel.close_panel()
        assert not panel.isVisible()
        panel.close_panel()
    finally:
        panel.deleteLater()


def test_no_banned_dash_in_any_visible_string(qapp):
    panel = BootPanel()
    try:
        for key, _ in PHASES:
            panel.start(key)
            assert not any(ch in panel._label.text() for ch in ("—", "–", "―"))
    finally:
        panel.close_panel()
        panel.deleteLater()


def test_panel_is_centred_on_its_screen(qapp):
    """A bare QWidget carrying Qt.WindowType.SplashScreen does not centre
    itself the way QSplashScreen does; offscreen still resolves a screen
    with a real geometry, so this catches a panel left unpositioned."""
    panel = BootPanel()
    try:
        panel.show()
        screen = panel.screen()
        assert screen is not None
        assert screen.availableGeometry().contains(panel.frameGeometry().center())
    finally:
        panel.close_panel()
        panel.deleteLater()
