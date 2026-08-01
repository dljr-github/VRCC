"""Offscreen Qt tests for the first-run wizard's spoken-language step: the
OS-locale-seeded source language (and a stored answer) pre-tick the picker, and
the proceed buttons stay disabled until at least one language is selected.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path

import pytest

from tests.test_firstrun_ui import qapp, _FakeDownloadManager, _store, _bridge, _tick  # noqa: F401
from vrcc.gui import firstrun_languages


def _wizard(tmp_path, *, source_language=None, spoken_languages=None):
    from vrcc.gui.firstrun import FirstRunWizard

    store = _store(tmp_path)
    if source_language is not None:
        store.config.stt.source_language = source_language
    if spoken_languages is not None:
        store.config.stt.spoken_languages = list(spoken_languages)
    dm = _FakeDownloadManager(tmp_path / "models")
    bridge = _bridge()
    return FirstRunWizard(store, dm, bridge), store, dm, bridge


def test_locale_source_language_preticks_the_picker(qapp, tmp_path):
    # The OS-locale-seeded source language pre-fills the answer the user then
    # confirms or changes; the wizard still shows the question.
    wiz, _store_, _dm, bridge = _wizard(tmp_path, source_language="Japanese")
    try:
        assert firstrun_languages.checked_spoken(wiz) == ["Japanese"]
    finally:
        wiz.close(); wiz.deleteLater(); bridge.detach()


def test_stored_spoken_answer_preticks(qapp, tmp_path):
    wiz, _store_, _dm, bridge = _wizard(tmp_path, spoken_languages=["Japanese"])
    try:
        assert firstrun_languages.checked_spoken(wiz) == ["Japanese"]
    finally:
        wiz.close(); wiz.deleteLater(); bridge.detach()


def test_no_locale_match_leaves_the_picker_empty_and_gated(qapp, tmp_path):
    # "auto" maps to no display language, so nothing pre-ticks and the user
    # must pick before proceeding.
    wiz, _store_, _dm, bridge = _wizard(tmp_path, source_language="auto")
    try:
        assert firstrun_languages.checked_spoken(wiz) == []
        assert not wiz._download_btn.isEnabled()
        assert not wiz._manual_btn.isEnabled()
        assert wiz._cancel_btn.isEnabled()
        _tick(wiz, "Japanese", only=True)
        assert wiz._download_btn.isEnabled()
        assert wiz._manual_btn.isEnabled()
    finally:
        wiz.close(); wiz.deleteLater(); bridge.detach()


def test_unticking_the_last_language_disables_proceed(qapp, tmp_path):
    wiz, _store_, _dm, bridge = _wizard(tmp_path, source_language="auto")
    try:
        _tick(wiz, "Japanese", only=True)
        assert wiz._download_btn.isEnabled()
        _tick(wiz, "Japanese", only=True)  # toggles Japanese back off
        assert not wiz._download_btn.isEnabled()
    finally:
        wiz.close(); wiz.deleteLater(); bridge.detach()


def test_locale_japanese_end_to_end(qapp, tmp_path, monkeypatch):
    """A Japanese-locale user: Japanese pre-ticks, SenseVoice is recommended,
    proceed is enabled, and the download marks SenseVoice present and active."""
    from vrcc.core import recommend

    monkeypatch.setattr(recommend, "detect_tier", lambda: "cpu")
    monkeypatch.setattr(recommend, "default_device_choice", lambda: "cpu")
    wiz, store, dm, bridge = _wizard(tmp_path, source_language="Japanese")
    try:
        assert firstrun_languages.checked_spoken(wiz) == ["Japanese"]
        assert wiz.recommended_whisper == "sense-voice-small"
        assert wiz._download_btn.isEnabled()

        wiz._apply_recommendation()
        wiz._download_body()
        assert store.config.stt.model == "sense-voice-small"
        assert dm.is_whisper_downloaded("sense-voice-small")
    finally:
        wiz.close(); wiz.deleteLater(); bridge.detach()


def test_failed_download_reset_keeps_proceed_disabled_without_pick(qapp, tmp_path, monkeypatch):
    """A failed download re-enables Cancel but must not re-enable the proceed
    buttons while nothing is ticked."""
    from PySide6.QtWidgets import QMessageBox

    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: None)
    wiz, _store_, _dm, bridge = _wizard(tmp_path, source_language="auto")
    try:
        wiz._downloading = True
        wiz._on_download_done(False, "boom")
        assert not wiz._download_btn.isEnabled()
        assert not wiz._manual_btn.isEnabled()
        assert wiz._cancel_btn.isEnabled()
    finally:
        wiz.close(); wiz.deleteLater(); bridge.detach()
