"""Offscreen Qt tests for the first-run wizard's spoken-language picker:
starts empty rather than pre-ticked from the OS locale, and still pre-ticks
a returning user's stored answer.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path

import pytest

from tests.test_firstrun_ui import qapp, _FakeDownloadManager, _store, _bridge, _tick  # noqa: F401
from vrcc.gui import firstrun_languages


def _wizard(tmp_path):
    from vrcc.gui.firstrun import FirstRunWizard

    store = _store(tmp_path)
    dm = _FakeDownloadManager(tmp_path / "models")
    bridge = _bridge()
    return FirstRunWizard(store, dm, bridge), store, bridge


def test_fresh_first_run_ticks_nothing(qapp, tmp_path):
    # A locale-seeded source language must NOT pre-tick the picker.
    from vrcc.gui.firstrun import FirstRunWizard

    store = _store(tmp_path)
    store.config.stt.source_language = "Japanese"
    dm = _FakeDownloadManager(tmp_path / "models")
    bridge = _bridge()
    wiz = FirstRunWizard(store, dm, bridge)
    try:
        assert firstrun_languages.checked_spoken(wiz) == []
    finally:
        wiz.close(); wiz.deleteLater(); bridge.detach()


def test_stored_spoken_answer_still_preticks(qapp, tmp_path):
    from vrcc.gui.firstrun import FirstRunWizard

    store = _store(tmp_path)
    store.config.stt.spoken_languages = ["Japanese"]
    dm = _FakeDownloadManager(tmp_path / "models")
    bridge = _bridge()
    wiz = FirstRunWizard(store, dm, bridge)
    try:
        assert firstrun_languages.checked_spoken(wiz) == ["Japanese"]
    finally:
        wiz.close(); wiz.deleteLater(); bridge.detach()


def test_proceed_disabled_until_a_language_is_picked(qapp, tmp_path):
    wiz, _store_, bridge = _wizard(tmp_path)
    try:
        assert not wiz._download_btn.isEnabled()
        assert not wiz._manual_btn.isEnabled()
        assert wiz._cancel_btn.isEnabled()
        _tick(wiz, "Japanese", only=True)
        assert wiz._download_btn.isEnabled()
        assert wiz._manual_btn.isEnabled()
    finally:
        wiz.close(); wiz.deleteLater(); bridge.detach()


def test_unticking_the_last_language_disables_proceed(qapp, tmp_path):
    wiz, _store_, bridge = _wizard(tmp_path)
    try:
        _tick(wiz, "Japanese", only=True)
        assert wiz._download_btn.isEnabled()
        _tick(wiz, "Japanese", only=True)  # toggles Japanese back off
        assert not wiz._download_btn.isEnabled()
    finally:
        wiz.close(); wiz.deleteLater(); bridge.detach()


def test_first_run_japanese_end_to_end(qapp, tmp_path, monkeypatch):
    """Fresh first run for a Japanese speaker: the locale seed does not
    pre-tick, proceed is blocked until the answer, answering Japanese
    recommends SenseVoice and enables proceed, and the download marks
    SenseVoice present and active."""
    from vrcc.core import recommend
    from vrcc.gui.firstrun import FirstRunWizard

    monkeypatch.setattr(recommend, "detect_tier", lambda: "cpu")
    monkeypatch.setattr(recommend, "default_device_choice", lambda: "cpu")
    store = _store(tmp_path)
    store.config.stt.source_language = "Japanese"  # a locale guess must not lead
    dm = _FakeDownloadManager(tmp_path / "models")
    bridge = _bridge()
    wiz = FirstRunWizard(store, dm, bridge)
    try:
        assert firstrun_languages.checked_spoken(wiz) == []
        assert not wiz._download_btn.isEnabled()

        _tick(wiz, "Japanese", only=True)
        assert wiz.recommended_whisper == "sense-voice-small"
        assert wiz._download_btn.isEnabled()

        wiz._apply_recommendation()
        wiz._download_body()
        assert store.config.stt.model == "sense-voice-small"
        assert dm.is_whisper_downloaded("sense-voice-small")
    finally:
        wiz.close(); wiz.deleteLater(); bridge.detach()


def test_fresh_run_plan_stays_language_blind_until_a_pick(qapp, tmp_path, monkeypatch):
    """Nothing ticked means no language is assumed for the shown plan either:
    the recommendation is the blind preset, not the one implied by the config's
    default source language."""
    from vrcc.core import recommend
    from vrcc.gui.firstrun import FirstRunWizard

    monkeypatch.setattr(recommend, "detect_tier", lambda: "cpu")
    monkeypatch.setattr(recommend, "default_device_choice", lambda: "cpu")
    store = _store(tmp_path)  # fresh: source_language defaults to English
    dm = _FakeDownloadManager(tmp_path / "models")
    bridge = _bridge()
    wiz = FirstRunWizard(store, dm, bridge)
    try:
        assert wiz._spoken_codes() == ()
        assert wiz.recommended_whisper == "small"  # blind cpu preset, not parakeet
    finally:
        wiz.close(); wiz.deleteLater(); bridge.detach()


def test_failed_download_reset_keeps_proceed_disabled_without_pick(qapp, tmp_path, monkeypatch):
    """A failed download re-enables Cancel but must not re-enable the proceed
    buttons while nothing is ticked."""
    from PySide6.QtWidgets import QMessageBox

    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: None)
    wiz, _store_, bridge = _wizard(tmp_path)
    try:
        wiz._downloading = True
        wiz._on_download_done(False, "boom")
        assert not wiz._download_btn.isEnabled()
        assert not wiz._manual_btn.isEnabled()
        assert wiz._cancel_btn.isEnabled()
    finally:
        wiz.close(); wiz.deleteLater(); bridge.detach()
