"""Tests for the caption log's no-speech empty-state sub-line: SpeechStarted
wired through the bridge, and the two gates in main_window (listening-only,
transition-driven re-render) that keep it from ever contradicting the
capture label beside it.
"""

import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from .test_main_window_ui import _window, qapp  # noqa: F401 -- qapp is a fixture


def test_no_speech_subline_hidden_while_not_listening(qapp, tmp_path):
    # Gate 1: "Paused - not listening" beside a no-speech judgement would
    # contradict it, so the flag alone (meter moved, no speech yet) must not
    # be enough while render_capture_status reports anything but listening.
    w, bridge = _window(tmp_path)
    try:
        w._engine_states["stt"] = "ready"
        w._meter_moved = True
        w._speech_starts = 0
        w._render_log()
        assert "arriving" not in w._log.toPlainText()
    finally:
        w.close(); w.deleteLater(); bridge.detach()


def test_no_speech_subline_shows_after_meter_moves_while_listening(qapp, tmp_path):
    w, bridge = _window(tmp_path)
    try:
        w._engine_states["stt"] = "ready"  # past the "getting ready" copy
        w._captioning_btn.setChecked(True)
        w.set_capture_status(True)
        w._on_mic_level(0.05, 0.0)
        assert "arriving" in w._log.toPlainText()
    finally:
        w.close(); w.deleteLater(); bridge.detach()


def test_no_speech_subline_needs_a_positive_level(qapp, tmp_path):
    # rms == 0 (a dead device, or nothing gated in yet) must never look like
    # movement; only a later positive reading flips the state.
    w, bridge = _window(tmp_path)
    try:
        w._captioning_btn.setChecked(True)
        w.set_capture_status(True)
        w._on_mic_level(0.0, 0.0)
        assert w._meter_moved is False
        assert "arriving" not in w._log.toPlainText()
    finally:
        w.close(); w.deleteLater(); bridge.detach()


def test_no_speech_subline_is_stale_safe_once_speech_starts(qapp, tmp_path):
    w, bridge = _window(tmp_path)
    try:
        w._engine_states["stt"] = "ready"
        w._captioning_btn.setChecked(True)
        w.set_capture_status(True)
        w._on_mic_level(0.05, 0.0)
        assert "arriving" in w._log.toPlainText()
        w._on_speech_started(None)
        w._render_log()  # any later render (a recognized phrase, an engine
        # state change) recomputes from current counts, so this never lingers.
        assert "arriving" not in w._log.toPlainText()
    finally:
        w.close(); w.deleteLater(); bridge.detach()


def test_no_speech_subline_clears_on_pause_with_no_further_caption_event(qapp, tmp_path):
    # Falling edge: pausing gates every later frame, so nothing else will ever
    # repaint the log. If the sub-line survived the pause it would sit under
    # "Paused - not listening" indefinitely (the contradiction gate 1 exists
    # to prevent), so the falling edge itself must clear it.
    w, bridge = _window(tmp_path)
    try:
        w._engine_states["stt"] = "ready"
        w._captioning_btn.setChecked(True)
        w.set_capture_status(True)
        w._on_mic_level(0.05, 0.0)
        assert "arriving" in w._log.toPlainText()

        w._captioning_btn.setChecked(False)  # -> _on_captions_toggled -> pause
        assert "arriving" not in w._log.toPlainText()
    finally:
        w.close(); w.deleteLater(); bridge.detach()


def test_listening_edge_resets_no_speech_state_both_ways(qapp, tmp_path):
    w, bridge = _window(tmp_path)
    try:
        w._captioning_btn.setChecked(True)
        w.set_capture_status(True)
        w._on_speech_started(None)
        assert w._speech_starts == 1

        w.set_capture_status(False)  # falling edge: also a fresh judgement
        assert w._speech_starts == 0
        assert w._meter_moved is False

        w._on_speech_started(None)
        w.set_capture_status(True)  # rising edge: fresh judgement again
        assert w._speech_starts == 0
        assert w._meter_moved is False
    finally:
        w.close(); w.deleteLater(); bridge.detach()


def test_speech_started_reaches_the_window_through_the_bridge(qapp, tmp_path):
    from vrcc.core.events import SpeechStarted

    w, bridge = _window(tmp_path)
    try:
        bridge._bus.publish(SpeechStarted(utterance_id=1))
        qapp.processEvents()
        assert w._speech_starts == 1
    finally:
        w.close(); w.deleteLater(); bridge.detach()


def test_disconnect_bridge_stops_speech_started_too(qapp, tmp_path):
    from vrcc.core.events import SpeechStarted

    w, bridge = _window(tmp_path)
    try:
        w.disconnect_bridge()
        bridge._bus.publish(SpeechStarted(utterance_id=9))
        qapp.processEvents()
        assert w._speech_starts == 0
    finally:
        w.close(); w.deleteLater(); bridge.detach()
