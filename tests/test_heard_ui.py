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


def test_the_speaker_picker_is_disabled_until_the_feature_is_on(qapp, tmp_path):
    dlg, _store_ = _dialog(tmp_path)
    try:
        assert not dlg._hear_check.isChecked()
        assert not dlg._hear_device_combo.isEnabled()

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
