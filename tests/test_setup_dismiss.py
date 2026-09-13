"""The setup panel's permanent dismiss: a footer button distinct from
closing the panel, which only silences it for the session (that behaviour
is covered elsewhere and reasserted here as a regression guard). Shares the
``_running`` fixture and ``_press_the_button`` helper with
test_setup_check_reshow.py, which already exercises the Settings side of
the request-counter path this feature must leave alone.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from vrcc.gui.setup_steps import required_passed
from tests.test_setup_check import no_language_nudge, qapp  # noqa: F401 -- shared fixtures
from tests.test_setup_check_reshow import _press_the_button, _running


def test_clicking_dismiss_writes_the_flag_and_hides_the_panel(qapp, tmp_path, monkeypatch):
    with _running(tmp_path) as (check, store, _):
        monkeypatch.setattr(store, "save_soon", lambda: None)
        assert check._panel.isVisible()
        # The scenario the feature exists for: a user stopping the panel
        # forever with the required rows still unmet, not one who already
        # finished and is only clearing a leftover flag.
        assert required_passed(check._facts) is False

        check._panel._dismiss_button.click()

        assert store.config.gui.setup_check_done is True
        assert not check._panel.isVisible()


def test_a_stored_dismiss_stays_hidden_on_the_next_launch_with_rows_unmet(
    qapp, tmp_path, monkeypatch
):
    """A network that blocks mDNS can never pass the vrchat row; a dismiss
    recorded in one session must hold on the next, not just for the
    controller that wrote it."""
    with _running(tmp_path) as (check, store, _):
        monkeypatch.setattr(store, "save_soon", lambda: None)
        check._panel._dismiss_button.click()
        assert store.config.gui.setup_check_done is True

    # A second controller over a store seeded the way a persisted config
    # would be stands in for the next launch.
    with _running(tmp_path, done=True) as (check2, _store2, _):
        assert required_passed(check2._facts) is False
        assert not check2._panel.isVisible()


def test_the_settings_button_still_reopens_after_a_permanent_dismiss(
    qapp, tmp_path, monkeypatch
):
    with _running(tmp_path) as (check, store, _):
        monkeypatch.setattr(store, "save_soon", lambda: None)
        check._panel._dismiss_button.click()
        assert not check._panel.isVisible()

        _press_the_button(store)
        check._recompute()
        assert check._panel.isVisible()


def test_plain_close_panel_leaves_the_flag_false(qapp, tmp_path):
    """Not a regression this change should ever touch: closing the panel
    without the dismiss button must still leave the flag False, so the
    panel returns on the next launch."""
    with _running(tmp_path) as (check, store, _):
        assert check._panel.isVisible()

        check._panel.close_panel()

        assert not check._panel.isVisible()
        assert store.config.gui.setup_check_done is False
