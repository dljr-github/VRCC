"""When the setup check panel opens, and when it stays shut.

Two separate questions drive it, and they used to share one boolean:
``gui.setup_check_done`` answers "should this open by itself at launch?",
while ``gui.setup_check_requests`` answers "the user just asked for it". The
flag alone could not express the second, because anyone who dismissed the
panel before finishing setup already holds it at False, so the Settings
button wrote nothing and the panel never came back. Every test here asserts
the panel's real visibility rather than the config field behind it.

Split out of test_setup_check_lifecycle.py, which is at the repo's
500-line-per-file cap; the fixtures and fakes both need are defined in
test_setup_check.py.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from contextlib import contextmanager

import shiboken6

from vrcc.core.bus import EventBus
from vrcc.gui.bridge import BusBridge
from vrcc.gui.setup_check import start
from tests.test_setup_check import (  # noqa: F401 -- shared fixtures/fakes
    _FakeDetector,
    _FakePipeline,
    _store,
    _window,
    no_language_nudge,
    qapp,
)
from tests.test_setup_check_lifecycle import _run_loop_once


@contextmanager
def _running(tmp_path, *, done: bool = False, requests: int = 0):
    """A live controller over a real shown MainWindow, torn down in order.
    `done` and `requests` are written before construction, so they stand in
    for what a previous session persisted."""
    store = _store(tmp_path)
    store.config.gui.setup_check_done = done
    store.config.gui.setup_check_requests = requests
    bus = EventBus()
    pipeline = _FakePipeline()
    bridge = BusBridge(bus)
    window = _window(bridge, store, pipeline)
    check = start(bus, store, window, _FakeDetector())
    try:
        yield check, store, window
    finally:
        check.stop()
        # A test may have destroyed the window on purpose.
        if shiboken6.isValid(window):
            window.close()
            window.deleteLater()
        bridge.detach()


def _press_the_button(store) -> None:
    """What settings_simple.on_show_setup writes. Spelled out rather than
    driven through the dialog so the assertions below stay about the
    controller; test_the_settings_button_reopens_a_dismissed_panel drives
    the real button."""
    store.config.gui.setup_check_done = False
    store.config.gui.setup_check_requests += 1


# -- a request reopens the panel, from any starting state ------------------


def test_a_request_reshows_the_panel_after_the_check_was_completed(qapp, tmp_path):
    """The Settings button never touches this controller or the panel
    (test_setup_settings.py's own design constraint). The poll is the only
    thing that can notice a request and act on it."""
    with _running(tmp_path, done=True) as (check, store, _):
        assert not check._panel.isVisible()

        _press_the_button(store)
        check._recompute()
        assert check._panel.isVisible()


def test_a_request_reshows_a_panel_dismissed_before_the_check_was_finished(qapp, tmp_path):
    """The state the old edge on setup_check_done could not see, and the one
    the Troubleshooting bullet in README.md is written for: setup was never
    completed, so the flag is already False, and clearing it again changes
    nothing. Only the request counter moves here."""
    with _running(tmp_path, done=False) as (check, store, _):
        assert check._panel.isVisible(), "an unfinished check opens the panel at launch"

        check._panel.close_panel()
        assert not check._panel.isVisible()

        _press_the_button(store)
        check._recompute()
        assert check._panel.isVisible()


def test_two_requests_in_one_session_each_reshow_the_panel(qapp, tmp_path):
    # A request deliberately leaves setup_check_done False, so a second press
    # in the same session starts from exactly the state the first ended in.
    with _running(tmp_path, done=True) as (check, store, _):
        _press_the_button(store)
        check._recompute()
        assert check._panel.isVisible()

        check._panel.close_panel()
        assert not check._panel.isVisible()

        _press_the_button(store)
        check._recompute()
        assert check._panel.isVisible()


def test_a_stored_request_does_not_reshow_the_panel_at_launch(qapp, tmp_path):
    """The counter persists, so a session that starts with a completed check
    and a request left over from last time must not read that request as
    fresh. Seeding _last_request from the stored value is what prevents it."""
    with _running(tmp_path, done=True, requests=7) as (check, _, _w):
        assert not check._panel.isVisible()

        check._recompute()
        check._recompute()
        assert not check._panel.isVisible()


def test_the_settings_button_reopens_a_dismissed_panel(qapp, tmp_path, monkeypatch):
    """End to end over the real button, on the store a real controller polls:
    the two halves agree on what a press means. Asserting only that the
    dialog wrote a config field is what let a dead button ship."""
    from vrcc.gui.settings import SettingsDialog

    with _running(tmp_path, done=False) as (check, store, _):
        monkeypatch.setattr(store, "save_soon", lambda: None)
        check._panel.close_panel()
        assert not check._panel.isVisible()

        dlg = SettingsDialog(store)
        try:
            dlg._show_setup_btn.click()
        finally:
            dlg.close()
            dlg.deleteLater()

        check._recompute()
        assert check._panel.isVisible()


# -- and what must NOT reopen it -------------------------------------------


def test_recompute_does_not_reshow_an_already_visible_panel(qapp, tmp_path, monkeypatch):
    # The poll runs four times a second; re-showing and re-placing an
    # already-open panel on every tick would be wasted work at best and
    # visible flicker at worst.
    with _running(tmp_path) as (check, _store_, _w):
        assert check._panel.isVisible()

        show_calls = []
        place_calls = []
        monkeypatch.setattr(check._panel, "show", lambda: show_calls.append(1))
        monkeypatch.setattr(check._panel, "place_beside", lambda w: place_calls.append(1))

        check._recompute()
        check._recompute()

        assert show_calls == []
        assert place_calls == []


def test_closing_the_panel_stays_closed_across_later_polls(qapp, tmp_path):
    """The regression a level-triggered re-show ("unmet and hidden, so show")
    produced: the timer forces the panel back within one poll of the user
    closing it, and there is no dismiss control to fall back on
    (setup_panel.py has none). Watching the request counter avoids it for
    free, since closing the panel never writes a request."""
    with _running(tmp_path) as (check, _store_, _w):
        assert check._panel.isVisible()

        check._panel.close_panel()
        assert not check._panel.isVisible()

        check._recompute()
        check._recompute()
        assert not check._panel.isVisible()


def test_reshow_waits_for_a_modal_dialog_to_close(qapp, tmp_path):
    # Settings is application-modal (SettingsDialog.exec()); a modeless
    # panel appearing on top of it, or behind it unnoticed, would make
    # settings_simple.py's tooltip ("come back once you close Settings")
    # false. The request is remembered rather than dropped, so it still
    # takes effect once the modal is gone, and it fires exactly once.
    from PySide6.QtWidgets import QDialog

    with _running(tmp_path, done=True) as (check, store, _):
        dlg = QDialog()
        dlg.setModal(True)
        dlg.show()
        qapp.processEvents()
        try:
            assert not check._panel.isVisible()

            _press_the_button(store)
            check._recompute()
            assert not check._panel.isVisible(), "must not appear while a modal is active"

            dlg.close()
            qapp.processEvents()
            check._recompute()
            assert check._panel.isVisible()

            check._panel.close_panel()
            check._recompute()
            assert not check._panel.isVisible(), "one request, one show"
        finally:
            dlg.deleteLater()


# -- a request that arrives with no window left to sit beside --------------


def test_recompute_shows_the_panel_with_no_valid_window(qapp, tmp_path):
    """A UI-language change destroys the old MainWindow (window_swap.py)
    while this controller deliberately survives it, so _window can point at
    a C++ object that no longer exists by the time a re-show is due. The
    panel must still appear; only the repositioning may be skipped.

    Drives the real destruction path rather than patching shiboken6.isValid:
    close() + deleteLater() + a genuine exec()/quit() cycle (plain
    processEvents() does not honor deleteLater() under offscreen QPA,
    confirmed separately) reliably invalidates a real MainWindow here."""
    with _running(tmp_path, done=True) as (check, store, window):
        window.close()
        window.deleteLater()
        _run_loop_once(qapp)
        assert not shiboken6.isValid(window), "setup did not actually destroy the window"

        _press_the_button(store)
        check._recompute()  # must not raise on the stale reference
        assert check._panel.isVisible()
