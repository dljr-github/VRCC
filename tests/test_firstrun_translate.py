"""Declining translation in the first-run wizard.

app.run shows the wizard modally before any window exists, so Settings cannot
be reached until it closes: without a tick here the 483 MB translation model is
the price of finishing first run.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

from tests.test_firstrun_device_ui import _teardown, _wizard
from vrcc.gui import firstrun_plan
from vrcc.stt.registry import WHISPER_MODELS
from vrcc.translate.registry import MT_MODELS


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_the_wizard_offers_a_translation_tick_and_starts_it_on(
    qapp, tmp_path, monkeypatch
):
    wiz, store, _dm, bridge = _wizard(tmp_path, monkeypatch, tier="cpu")
    try:
        assert wiz._translate_check.isChecked()
        assert store.config.translate.enabled is True
    finally:
        _teardown(wiz, bridge)


def test_unticking_translation_drops_the_model_from_the_plan(
    qapp, tmp_path, monkeypatch
):
    wiz, store, _dm, bridge = _wizard(tmp_path, monkeypatch, tier="cpu")
    try:
        whisper_mb = WHISPER_MODELS[wiz.recommended_whisper].size_mb
        mt_name = "M2M100"

        wiz._translate_check.setChecked(False)

        assert store.config.translate.enabled is False
        assert firstrun_plan.pending_mb(wiz) == whisper_mb
        text = wiz._summary_label.text()
        assert mt_name not in text
        assert "Translation:" not in text
        assert not wiz._license_note.isVisibleTo(wiz)
        # Nothing to send a translation to while it is off.
        assert not wiz._target_combo.isEnabled()
    finally:
        _teardown(wiz, bridge)


def test_reticking_translation_brings_the_model_back(qapp, tmp_path, monkeypatch):
    wiz, store, _dm, bridge = _wizard(tmp_path, monkeypatch, tier="cpu")
    try:
        wiz._translate_check.setChecked(False)
        wiz._translate_check.setChecked(True)

        assert store.config.translate.enabled is True
        assert "Translation:" in wiz._summary_label.text()
        assert wiz._target_combo.isEnabled()
        assert firstrun_plan.pending_mb(wiz) == (
            WHISPER_MODELS[wiz.recommended_whisper].size_mb
            + MT_MODELS[wiz.recommended_mt].size_mb
        )
    finally:
        _teardown(wiz, bridge)


def test_declined_translation_downloads_the_voice_model_only(
    qapp, tmp_path, monkeypatch
):
    wiz, store, dm, bridge = _wizard(tmp_path, monkeypatch, tier="cpu")
    try:
        wiz._translate_check.setChecked(False)
        wiz._apply_recommendation()
        wiz._download_body()

        assert dm.is_whisper_downloaded(store.config.stt.model)
        assert not dm.is_mt_downloaded(MT_MODELS[wiz.recommended_mt])
        assert wiz._configured_models_present()
    finally:
        _teardown(wiz, bridge)


def test_the_missing_translation_model_hint_points_at_the_tick_not_settings(
    qapp, tmp_path, monkeypatch
):
    """The old wording sent the user to Settings, which does not open until
    this wizard closes."""
    from PySide6.QtWidgets import QMessageBox

    wiz, store, dm, bridge = _wizard(tmp_path, monkeypatch, tier="cpu")
    store.config.stt.model = "large-v3"  # configured, but NOT downloaded
    store.config.translate.model = "nllb-3.3B-int8"

    def fake_exec(self):
        dm.downloaded.add("small")  # a voice model, but no translation model
        return 0

    monkeypatch.setattr("vrcc.gui.models_dialog.ModelsDialog.exec", fake_exec)
    shown: list[str] = []
    monkeypatch.setattr(
        QMessageBox, "information", lambda *a, **k: shown.append(a[2])
    )
    try:
        wiz._on_choose_manually()

        assert shown, "the wizard must say why it cannot start"
        assert "Translate my speech" in shown[0]
        assert "Settings" not in shown[0]
    finally:
        _teardown(wiz, bridge)
