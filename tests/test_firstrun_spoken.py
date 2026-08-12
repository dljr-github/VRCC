"""Offscreen Qt tests for the first-run wizard's spoken-language step: the
OS-locale-seeded source language (and a stored answer) pre-tick the picker, and
the proceed buttons stay disabled until at least one language is selected.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path

import pytest
from PySide6.QtCore import Qt

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


def test_unticking_everything_makes_the_plan_language_blind(
    qapp, tmp_path, monkeypatch
):
    """resolve_source_language promises the recommendation "falls back to being
    language blind" when nobody answers. It kept ranking for the language just
    removed, because spoken_whisper_codes reads the stored source when the
    multi-select is empty."""
    from vrcc.core import calibrate, recommend

    monkeypatch.setattr(recommend, "detect_tier", lambda index=0: "cpu")
    monkeypatch.setattr(recommend, "default_device_choice", lambda index=0: "cpu")
    # Pin the CPU calibration factor. It is measured from the machine running
    # the tests, and it scales the latency gate that decides between "small"
    # and "base": a slower runner (CI) legitimately ranks differently from a
    # fast desktop. These tests are about language coverage driving the
    # ranking, not about how quick the runner is.
    monkeypatch.setattr(calibrate, "cached_factor", lambda cfg, remeasure=False: 1.0)
    wiz, store, _dm, bridge = _wizard(tmp_path, source_language="Japanese")
    try:
        assert wiz.recommended_whisper == "small"

        _tick(wiz, "Japanese", only=True)  # toggles the only pick back off

        assert firstrun_languages.checked_spoken(wiz) == []
        assert store.config.stt.source_language == "auto"
        assert wiz.recommended_whisper == recommend.WHISPER_PREFERENCE["cpu"][0]
        assert "SenseVoice" not in wiz._summary_label.text()
    finally:
        wiz.close(); wiz.deleteLater(); bridge.detach()


def test_declining_to_answer_keeps_a_source_a_silent_model_cannot_replace(
    qapp, tmp_path, monkeypatch
):
    """Parakeet detects the language but tags every result "en", so "auto" is
    not a source it can honour while translating. A stale answer beats one the
    translator would be handed wrong."""
    from vrcc.core import calibrate, recommend

    monkeypatch.setattr(recommend, "detect_tier", lambda index=0: "cpu")
    monkeypatch.setattr(recommend, "default_device_choice", lambda index=0: "cpu")
    # Pin the CPU calibration factor. It is measured from the machine running
    # the tests, and it scales the latency gate that decides between "small"
    # and "base": a slower runner (CI) legitimately ranks differently from a
    # fast desktop. These tests are about language coverage driving the
    # ranking, not about how quick the runner is.
    monkeypatch.setattr(calibrate, "cached_factor", lambda cfg, remeasure=False: 1.0)
    wiz, store, _dm, bridge = _wizard(tmp_path, source_language="German")
    try:
        assert wiz.recommended_whisper == "parakeet-tdt-0.6b-v3"
        assert store.config.translate.enabled is True

        _tick(wiz, "German", only=True)  # toggles the only pick back off

        assert firstrun_languages.checked_spoken(wiz) == []
        assert store.config.stt.source_language == "German"
    finally:
        wiz.close(); wiz.deleteLater(); bridge.detach()


def test_locale_japanese_end_to_end(qapp, tmp_path, monkeypatch):
    """A Japanese-locale user: Japanese pre-ticks, SenseVoice is recommended,
    proceed is enabled, and the download marks SenseVoice present and active."""
    from vrcc.core import calibrate, recommend

    monkeypatch.setattr(recommend, "detect_tier", lambda index=0: "cpu")
    monkeypatch.setattr(recommend, "default_device_choice", lambda index=0: "cpu")
    # Pin the CPU calibration factor. It is measured from the machine running
    # the tests, and it scales the latency gate that decides between "small"
    # and "base": a slower runner (CI) legitimately ranks differently from a
    # fast desktop. These tests are about language coverage driving the
    # ranking, not about how quick the runner is.
    monkeypatch.setattr(calibrate, "cached_factor", lambda cfg, remeasure=False: 1.0)
    wiz, store, dm, bridge = _wizard(tmp_path, source_language="Japanese")
    try:
        assert firstrun_languages.checked_spoken(wiz) == ["Japanese"]
        assert wiz.recommended_whisper == "small"
        assert wiz._download_btn.isEnabled()

        wiz._apply_recommendation()
        wiz._download_body()
        assert store.config.stt.model == "small"
        assert dm.is_whisper_downloaded("small")
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


def test_the_models_window_answer_survives_back_in_the_wizard(qapp, tmp_path):
    # "Choose manually" opens the Models window over this same store, and that
    # window offers the same picker. On the path where nothing usable was
    # downloaded the wizard stays open, and _apply_recommendation writes its own
    # ticks: without a re-read those stale ticks overwrite the newer answer.
    wiz, store, _dm, bridge = _wizard(tmp_path, source_language="English")
    try:
        assert firstrun_languages.checked_spoken(wiz) == ["English"]

        # What the Models window does to the shared config.
        store.config.stt.spoken_languages = ["Japanese"]
        store.config.stt.source_language = "Japanese"
        firstrun_languages.resync_spoken(wiz)

        assert firstrun_languages.checked_spoken(wiz) == ["Japanese"]

        wiz._apply_recommendation()
        assert store.config.stt.spoken_languages == ["Japanese"]
        assert store.config.stt.source_language == "Japanese"
    finally:
        wiz.close(); wiz.deleteLater(); bridge.detach()


def test_resync_does_not_refire_the_picker_handler(qapp, tmp_path):
    # setCheckState during a programmatic re-tick would otherwise run
    # on_spoken_changed once per row and write the half-applied state back.
    wiz, store, _dm, bridge = _wizard(tmp_path, source_language="English")
    try:
        calls = []
        original = firstrun_languages.on_spoken_changed
        firstrun_languages.on_spoken_changed = lambda w: calls.append(w)
        try:
            store.config.stt.spoken_languages = ["Japanese", "Korean"]
            firstrun_languages.resync_spoken(wiz)
        finally:
            firstrun_languages.on_spoken_changed = original

        assert calls == []
        assert firstrun_languages.checked_spoken(wiz) == ["Japanese", "Korean"]
    finally:
        wiz.close(); wiz.deleteLater(); bridge.detach()


def test_wizard_never_finishes_with_the_target_equal_to_the_source(tmp_path):
    """The shipped default target is Japanese and so is the pre-tick on a
    Japanese system, so the likeliest first run of all (open VRCC, press the
    primary button) finished with source == target. The pipeline drops a
    source-equal target, so the wizard promised translation and delivered none.
    """
    wiz, store, dm, bridge = _wizard(tmp_path, source_language="Japanese")
    try:
        assert store.config.stt.source_language == "Japanese"
        assert wiz._target_combo.currentText() != "Japanese"

        wiz._apply_recommendation()
        cfg = store.config
        live = [t for t in cfg.translate.targets if t != cfg.stt.source_language]
        assert live, "the wizard must leave at least one target it can translate"
    finally:
        wiz.close()
        bridge.detach()


def test_wizard_retargets_when_the_user_ticks_the_target_language(tmp_path):
    wiz, store, dm, bridge = _wizard(tmp_path, source_language="English")
    try:
        wiz._set_combo_text(wiz._target_combo, "Korean")
        for i in range(wiz._spoken_list.count()):
            item = wiz._spoken_list.item(i)
            item.setCheckState(
                Qt.CheckState.Checked if item.text() == "Korean"
                else Qt.CheckState.Unchecked
            )

        assert store.config.stt.source_language == "Korean"
        assert wiz._target_combo.currentText() != "Korean"
    finally:
        wiz.close()
        bridge.detach()


def test_choosing_models_manually_refuses_a_voice_model_that_cannot_serve(tmp_path):
    """resolve_source_language leaves the stored value alone when NOTHING the
    user ticked is servable. Starting there captions in silence, and the main
    window's rescue nudge has nothing better on disk to offer."""
    from PySide6.QtWidgets import QMessageBox

    from vrcc.gui import models_dialog

    wiz, store, dm, bridge = _wizard(tmp_path, source_language="German")
    shown: list[str] = []
    accepted: list[bool] = []
    real_info, real_exec = QMessageBox.information, models_dialog.ModelsDialog.exec
    try:
        QMessageBox.information = lambda *a, **k: shown.append(a[2])
        models_dialog.ModelsDialog.exec = lambda self: dm.downloaded.update(
            {"sense-voice-small", "m2m100-418M-int8"}
        )
        wiz.accept = lambda: accepted.append(True)

        wiz._on_choose_manually()

        assert accepted == [], "must not start on a model that cannot serve German"
        assert shown and "cannot transcribe" in shown[0]
    finally:
        QMessageBox.information = real_info
        models_dialog.ModelsDialog.exec = real_exec
        wiz.close()
        bridge.detach()
