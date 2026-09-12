"""Rebuild the main window in place, carrying the state a fresh one cannot derive.

Lives here rather than in :mod:`vrcc.app` so the composition root stays under the
source cap. Deliberately imports nothing: :mod:`vrcc.app` re-exports this name at
module scope, and a PySide6 import here would pull Qt into the headless
collection of every test that imports from :mod:`vrcc.app` without a display.
"""

from __future__ import annotations


def _swap_main_window(old, make_window, detector, mute):
    """Replace ``old`` with a freshly built MainWindow and carry its runtime
    state across: nothing replays bus events for a late subscriber, so the
    fresh window would otherwise sit on "Starting" and "VRChat: checking"
    until the next transition. The capture label carries verbatim; a red
    failure must stay red whether or not the pipeline ever started, and
    paused-vs-listening re-derives from the captioning toggle, which the
    fresh window reads from the pipeline at construction. ``mute`` is the
    live MuteSync coordinator (``None`` if mute sync was never enabled);
    republishing it alongside the detector means a language change while
    muted doesn't leave the rebuilt window's mute chip hidden."""
    old.disconnect_bridge()
    fresh = make_window()
    fresh.restoreGeometry(old.saveGeometry())
    fresh._engine_states.update(old._engine_states)
    fresh._render_log()
    fresh.set_capture_status(old._capture_ok, old._capture_reason)
    detector.republish()
    if mute is not None:
        mute.republish()
    fresh.show()
    old.hide()
    old.deleteLater()
    return fresh
