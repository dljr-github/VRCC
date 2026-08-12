"""What the main window does with the speaker-capture stream.

Split from :mod:`vrcc.gui.main_window` for the 500-line cap, and because these
three answer each other: the toggle switches the stream on, the meter shows
what it is receiving, and the caption rows show what it understood. Takes the
window as its first argument, the way :mod:`vrcc.gui.main_targets` does.

The meter matters more than it looks. It is the only way a user can tell the
two failures apart: a still meter while VRChat is loud means the capture is on
the wrong output device, and a moving meter while only they are talking means
their own voice is being played back into that device. Neither is visible from
the captions alone, because both end in no captions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vrcc.gui.main_window import MainWindow


def on_level(w: MainWindow, rms: float) -> None:
    w._heard_meter.set_level(rms)


def on_toggled(w: MainWindow, checked: bool) -> None:
    """Turn the speaker capture on or off live.

    The setting used to be read only when the app started, so switching it on
    did nothing until a relaunch. The meter dims with it, because a still meter
    beside a live one reads as broken rather than off.
    """
    w._heard_meter.set_active(checked)
    if w._loading:
        return
    w._store.config.audio.hear_others_enabled = checked
    w._store.save_soon()
    if w._on_hear_others is not None:
        w._on_hear_others(checked)


def on_phrase(w: MainWindow, event) -> None:
    """Someone else's speech, for reading only. Nothing is suppressed here
    because there is nothing to suppress: HeardStream has no sender."""
    w._caption_model.heard(event.text, list(event.translations))
    w._render_log()
