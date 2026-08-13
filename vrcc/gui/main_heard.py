"""What the main window does with the speaker-capture stream.

Split from :mod:`vrcc.gui.main_window` for the 500-line cap. Takes the window
as its first argument, the way :mod:`vrcc.gui.main_targets` does.

The meter is the only way to tell the two silent failures apart: a still meter
while VRChat is loud means the capture is on the wrong output device, and a
moving meter while only this user is talking means their own voice is being
played back into that device. Both end in no captions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from vrcc.gui import main_parts
from vrcc.i18n import tr

if TYPE_CHECKING:
    from vrcc.gui.main_window import MainWindow


def set_toggle_state(w: MainWindow, on: bool) -> None:
    """Put every part of the window that shows this state into agreement."""
    w._hear_btn.setText(tr("Hearing others") if on else tr("Hear others"))
    # A property change needs the style re-polished; Qt does not repaint on it.
    w._hear_btn.style().unpolish(w._hear_btn)
    w._hear_btn.style().polish(w._hear_btn)
    w._heard_meter.set_active(on)
    if not on:
        w._heard_meter.set_level(0.0)
    main_parts.set_heard_visible(w, on)


def on_level(w: MainWindow, rms: float) -> None:
    w._heard_meter.set_level(rms)


def on_toggled(w: MainWindow, checked: bool) -> None:
    """Turn the speaker capture on or off live."""
    set_toggle_state(w, checked)
    if w._loading:
        return
    w._store.config.audio.hear_others_enabled = checked
    w._store.save_soon()
    if w._on_hear_others is not None:
        w._on_hear_others(checked)


# Failures that mean the stream is not running, whatever the toggle says.
_FAILURE_CODES = frozenset({"HEARD_NO_LIBRARY", "HEARD_DEVICE_FAILED"})


def on_error(w: MainWindow, code: str) -> None:
    """Put the toggle back down when the capture could not start.

    The capture opens its device on its own thread, so a failure arrives long
    after the click returned.

    Signals are blocked because the publisher has already switched the config
    off; re-entering the toggle handler would only ask a dead stream to stop
    again.
    """
    if code not in _FAILURE_CODES:
        return
    blocked = w._hear_btn.blockSignals(True)
    try:
        w._hear_btn.setChecked(False)
    finally:
        w._hear_btn.blockSignals(blocked)
    # Signals are blocked, so the toggle handler will not run and the rest of
    # the window would otherwise keep showing this as on.
    set_toggle_state(w, False)


def on_phrase(w: MainWindow, event) -> None:
    """Someone else's speech, for reading only. Nothing is suppressed here
    because there is nothing to suppress: HeardStream has no sender."""
    w._caption_model.heard(event.text, list(event.translations))
    w._render_log()
