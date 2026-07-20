"""Status-strip rendering for the main window: the VRChat presence label,
the mute chip and the capture label. Split out of main_window (500-line cap)
so the capture label's whole truth table lives in one place.

The capture label folds in the pipeline's mute gate: mute sync drops captions
silently inside the pipeline, so a label derived from the toggle alone would
claim "Listening" over a closed gate. Every helper takes the window, mutates
its widgets directly and runs on the GUI thread.
"""

from __future__ import annotations

from vrcc.i18n import tr


def render_vrchat(w, detected) -> None:
    """Paint the VRChat presence label; ``None`` is the initial "checking"
    state before the detector's first publish."""
    tip = tr(
        "Enable OSC in VRChat: Action Menu > Options > OSC > Enabled. "
        "VRChat must be running on this PC."
    )
    if detected is True:
        w._vrchat_label.setText(tr("VRChat: connected"))
        w._vrchat_label.setStyleSheet(f"color: {w._p['good']}; padding: 2px 8px;")
        w._vrchat_label.setToolTip(tr("VRChat's OSC service was found on this network."))
    elif detected is False:
        w._vrchat_label.setText(tr("VRChat: not detected - enable OSC in-game"))
        w._vrchat_label.setStyleSheet(f"color: {w._p['warn']}; padding: 2px 8px;")
        w._vrchat_label.setToolTip(tip)
    else:
        w._vrchat_label.setText(tr("VRChat: checking…"))
        w._vrchat_label.setStyleSheet(f"color: {w._p['muted']}; padding: 2px 8px;")
        w._vrchat_label.setToolTip(tip)


def set_mute_chip(w, muted) -> None:
    # None (state unknown, or mute sync stopped and published MuteChanged(None))
    # hides the chip entirely rather than showing an empty "-" box or keeping
    # a value that no longer tracks VRChat.
    if muted is None:
        w._mute_chip.setVisible(False)
        return
    if muted:
        w._mute_chip.setText(tr("MUTED"))
        color = w._p["bad"]
    else:
        w._mute_chip.setText(tr("LIVE"))
        color = w._p["good"]
    w._mute_chip.setStyleSheet(
        f"color: {w._p['on_badge']}; background: {color}; padding: 2px 8px;"
    )
    w._mute_chip.setVisible(True)


def _mute_gate_closed(pipeline) -> bool:
    # getattr: the minimal pipeline fakes some tests inject carry no
    # mute_gated; treat that as an open gate.
    gate = getattr(pipeline, "mute_gated", None)
    return gate is not None and bool(gate())


def render_capture_status(w) -> bool:
    """Derive the capture label from capture health, the captioning toggle
    and the pipeline's mute gate, in that priority order. The toggle outranks
    the mute gate because the pipeline checks it first: naming the mute while
    the toggle is off would point the user at the wrong control. Returns
    whether the app is actually listening (the green state), so the caller can
    clear stuck live-partial rows when it is not."""
    ok = getattr(w, "_capture_ok", None)
    listening = False
    if ok is None:
        text, color = tr("Starting…"), w._p["muted"]
    elif ok is False:
        reason = getattr(w, "_capture_reason", "")
        if reason:
            text = tr("Not listening - {reason}", reason=reason)
        else:
            text = tr("Not listening")
        color = w._p["bad"]
    elif getattr(w, "_captioning_btn", None) is not None and not w._captioning_btn.isChecked():
        text, color = tr("Paused - not listening"), w._p["warn"]
    elif _mute_gate_closed(w._pipeline):
        text, color = tr("Paused - following your VRChat mute"), w._p["warn"]
    else:
        text, color = tr("Listening"), w._p["good"]
        listening = True
    w._capture_label.setText(text)
    w._capture_label.setStyleSheet(f"color: {color}; padding: 2px 8px;")
    return listening
