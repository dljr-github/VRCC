"""The window and settings surfaces for captioning what you hear.

The boundary again, from the other side: a heard row must be visibly not-yours
and must never carry a delivery status, because a row that looks like your own
invites the reading that VRChat saw it too.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

from tests.test_app_main_window import _main_window, _store, qapp  # noqa: F401
from vrcc.core.events import HeardPhrase
from vrcc.gui.caption_log import HEARD, CaptionModel, render_row_html


def test_a_heard_phrase_reaches_the_feed(qapp, tmp_path):
    window, bridge = _main_window(_store(tmp_path))
    try:
        window._on_heard_phrase(
            HeardPhrase(text="konnichiwa", language="Japanese",
                        translations=[("English", "hello")])
        )

        text = window._log.toPlainText()
        assert "konnichiwa" in text
        assert "hello" in text
    finally:
        window.close()
        bridge.detach()


def test_a_heard_row_never_claims_to_have_been_sent(qapp, tmp_path):
    """It was never going anywhere. "sent", "queued" and "translating" would
    each be a lie, and the first one is the dangerous lie."""
    window, bridge = _main_window(_store(tmp_path))
    try:
        window._on_heard_phrase(
            HeardPhrase(text="hola", language="Spanish", translations=[])
        )

        text = window._log.toPlainText().lower()
        assert "sent" not in text
        assert "queued" not in text
        assert "translating" not in text
        assert "heard" in text
    finally:
        window.close()
        bridge.detach()


def test_heard_rows_are_drawn_apart_from_your_own():
    model = CaptionModel()
    model.heard("someone else", [])
    model.recognized(1, "me", translate_enabled=False, send_enabled=True)
    heard_row, own_row = model.rows()

    assert heard_row.heard and not own_row.heard
    assert heard_row.status == HEARD
    assert render_row_html(heard_row) != render_row_html(own_row)


def test_a_heard_row_is_complete_on_arrival():
    """No later translated or sent event follows, so it must not sit in a
    pending state waiting for one that never comes."""
    from vrcc.gui.caption_log import _TERMINAL

    model = CaptionModel()
    model.heard("done", [])

    assert model.rows()[0].status in _TERMINAL


def test_heard_rows_cannot_collide_with_a_real_utterance():
    model = CaptionModel()
    model.recognized(1, "mine", translate_enabled=False, send_enabled=True)
    model.heard("theirs", [])

    ids = [r.utterance_id for r in model.rows()]
    assert len(set(ids)) == len(ids)


# -- settings -----------------------------------------------------------------


def _dialog(tmp_path):
    from vrcc.core.config import ConfigStore, default_paths
    from vrcc.gui.settings import SettingsDialog

    store = ConfigStore(default_paths(portable=True, app_dir=tmp_path).config_file)
    return SettingsDialog(store), store


def test_the_speaker_picker_can_be_used_before_the_feature_is_on(qapp, tmp_path):
    """It used to be greyed out until the tick was on, which left the one
    question a user actually has, which speakers, unanswerable first. A
    disabled combo reading "Default speakers" also gives no hint that the tick
    above it is what locked it, so it just looks stuck."""
    dlg, _store_ = _dialog(tmp_path)
    try:
        assert not dlg._hear_check.isChecked()
        assert dlg._hear_device_combo.isEnabled()

        dlg._hear_check.setChecked(True)
        assert dlg._hear_device_combo.isEnabled()
    finally:
        dlg.close()
        dlg.deleteLater()


def test_ticking_it_writes_config(qapp, tmp_path):
    dlg, store = _dialog(tmp_path)
    try:
        dlg._hear_check.setChecked(True)
        assert store.config.audio.hear_others_enabled is True

        dlg._hear_check.setChecked(False)
        assert store.config.audio.hear_others_enabled is False
    finally:
        dlg.close()
        dlg.deleteLater()


def test_the_default_speaker_stores_an_empty_device(qapp, tmp_path):
    """Empty means "whatever the default is at capture time", so a user who
    swaps headsets is followed rather than pinned to a device that is gone."""
    dlg, store = _dialog(tmp_path)
    try:
        dlg._hear_check.setChecked(True)
        dlg._hear_device_combo.setCurrentIndex(0)
        assert store.config.audio.hear_others_device == ""
    finally:
        dlg.close()
        dlg.deleteLater()


def test_the_cpu_warning_shows_only_where_it_applies(qapp, tmp_path, monkeypatch):
    from vrcc.gui import settings_heard

    monkeypatch.setattr(settings_heard.recommend, "detect_tier", lambda index=0: "cpu")
    dlg, _store_ = _dialog(tmp_path)
    try:
        dlg._hear_check.setChecked(True)
        assert dlg._hear_note.isVisibleTo(dlg)
        assert "graphics card" in dlg._hear_note.text()
    finally:
        dlg.close()
        dlg.deleteLater()


def test_no_cpu_warning_on_a_machine_with_a_graphics_card(qapp, tmp_path, monkeypatch):
    from vrcc.gui import settings_heard

    monkeypatch.setattr(settings_heard.recommend, "detect_tier", lambda index=0: "gpu_high")
    dlg, _store_ = _dialog(tmp_path)
    try:
        dlg._hear_check.setChecked(True)
        assert not dlg._hear_note.isVisibleTo(dlg)
    finally:
        dlg.close()
        dlg.deleteLater()


# -- the main-window toggle and meter -----------------------------------------


def test_the_toggle_starts_and_stops_it_without_a_relaunch(qapp, tmp_path):
    """The setting used to be read only at startup, so turning it on did
    nothing until the app was restarted."""
    calls: list[bool] = []
    window, bridge = _main_window(_store(tmp_path))
    window._on_hear_others = calls.append
    try:
        assert not window._hear_btn.isChecked()

        window._hear_btn.setChecked(True)
        assert window._store.config.audio.hear_others_enabled is True
        assert calls == [True]

        window._hear_btn.setChecked(False)
        assert window._store.config.audio.hear_others_enabled is False
        assert calls == [True, False]
    finally:
        window.close()
        bridge.detach()


def test_the_button_follows_the_setting_changed_in_settings(qapp, tmp_path):
    """Two controls for one setting must never disagree."""
    window, bridge = _main_window(_store(tmp_path))
    try:
        window._store.config.audio.hear_others_enabled = True
        window.reload_from_config()

        assert window._hear_btn.isChecked()
    finally:
        window.close()
        bridge.detach()


def test_the_speaker_meter_moves_with_the_loopback_level(qapp, tmp_path):
    """Its own meter, because it is the only way to see what that stream is
    receiving: silence while VRChat is loud means the wrong output device."""
    window, bridge = _main_window(_store(tmp_path))
    try:
        window._on_heard_level(0.3, 0.9)
        moved = window._heard_meter._level

        window._on_heard_level(0.0, 0.0)

        assert moved > 0
        assert window._heard_meter._level == 0
    finally:
        window.close()
        bridge.detach()


def test_the_speaker_meter_is_dimmed_while_the_feature_is_off(qapp, tmp_path):
    """A still meter beside a live one reads as broken rather than off."""
    window, bridge = _main_window(_store(tmp_path))
    try:
        assert not window._heard_meter._active

        window._hear_btn.setChecked(True)
        assert window._heard_meter._active
    finally:
        window.close()
        bridge.detach()


def test_the_mic_and_speaker_meters_are_separate_widgets(qapp, tmp_path):
    window, bridge = _main_window(_store(tmp_path))
    try:
        window._on_mic_level(0.5, 0.9)
        window._on_heard_level(0.0, 0.0)

        assert window._mic_meter._level > 0
        assert window._heard_meter._level == 0
    finally:
        window.close()
        bridge.detach()


def test_a_capture_failure_puts_the_toggle_back_down(qapp, tmp_path):
    """A lit toggle beside a stream that is not running is the exact state that
    made this feature look like it worked while producing nothing. The failure
    arrives from the capture thread long after the click returned, so the
    button cannot un-check itself in the click handler."""
    from vrcc.core.events import AppError

    w, bridge = _main_window(_store(tmp_path))
    try:
        w._hear_btn.setChecked(True)
        assert w._hear_btn.isChecked()

        w._on_app_error(AppError("HEARD_NO_LIBRARY", "soundcard is not installed"))

        assert not w._hear_btn.isChecked()
    finally:
        w.close()
        bridge.detach()


def test_a_capture_failure_names_the_cause_on_screen(qapp, tmp_path):
    """The generic handler sentence would tell a user nothing: the whole point
    of reporting the failure is which of the two causes it was."""
    from vrcc.core.events import AppError

    w, bridge = _main_window(_store(tmp_path))
    try:
        for code, wanted in (
            ("HEARD_NO_LIBRARY", "Reinstall VRCC"),
            ("HEARD_DEVICE_FAILED", "speakers could not be opened"),
        ):
            w._on_app_error(AppError(code, "detail for the log only"))
            shown = w.statusBar().currentMessage()
            assert wanted in shown, (code, shown)
    finally:
        w.close()
        bridge.detach()


def test_an_unrelated_error_leaves_the_toggle_alone(qapp, tmp_path):
    """Only the two speaker-capture codes mean that stream is dead."""
    from vrcc.core.events import AppError

    w, bridge = _main_window(_store(tmp_path))
    try:
        w._hear_btn.setChecked(True)
        w._on_app_error(AppError("CHATBOX_SEND_FAILED", "no VRChat"))
        assert w._hear_btn.isChecked()
    finally:
        w.close()
        bridge.detach()


def test_the_toggle_says_which_state_it_is_in(qapp, tmp_path):
    """Fusion's checked look is a slightly sunken border, invisible on a dark
    surface, so a button with one fixed label read the same on as off."""
    w, bridge = _main_window(_store(tmp_path))
    try:
        off = w._hear_btn.text()
        w._hear_btn.setChecked(True)
        on = w._hear_btn.text()

        assert on != off, "the label must change with the state"
        assert w._hear_btn.property("buttonRole") == "toggle", (
            "the toggle needs the role that paints it when checked"
        )
    finally:
        w.close()
        bridge.detach()


def test_the_speakers_meter_is_labelled_and_only_shown_when_live(qapp, tmp_path):
    """An unlabelled meter placed after the microphone's label read as though
    that one word covered both, and a meter that can never move looks broken."""
    w, bridge = _main_window(_store(tmp_path))
    try:
        # isVisibleTo, not isVisible: every child of an unshown top-level
        # window reports invisible, which would pass the "off" half for free.
        strip = w._heard_meter.parentWidget()
        assert w._heard_label.text(), "the speakers meter needs its own label"
        assert not w._heard_meter.isVisibleTo(strip)
        assert not w._heard_label.isVisibleTo(strip)

        w._hear_btn.setChecked(True)
        assert w._heard_meter.isVisibleTo(strip)
        assert w._heard_label.isVisibleTo(strip)
        assert w._heard_meter._active
    finally:
        w.close()
        bridge.detach()


def test_a_capture_failure_takes_the_whole_on_state_down_with_it(qapp, tmp_path):
    """The failure path un-checks with signals blocked, so nothing else in the
    window would follow unless it is driven directly."""
    from vrcc.core.events import AppError

    w, bridge = _main_window(_store(tmp_path))
    try:
        w._hear_btn.setChecked(True)
        on_label = w._hear_btn.text()

        w._on_app_error(AppError("HEARD_DEVICE_FAILED", "no device"))

        assert not w._hear_btn.isChecked()
        assert w._hear_btn.text() != on_label
        assert not w._heard_meter.isVisibleTo(w._heard_meter.parentWidget())
    finally:
        w.close()
        bridge.detach()
