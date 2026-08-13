"""The first-run download step: what freezes while it runs, and what comes back.

The worker thread re-reads ``recommended_whisper`` / ``recommended_mt`` at call
time, but ``_apply_recommendation`` has already frozen that pair into config, so
any control that can move the plan has to be frozen for the duration.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt

from tests.test_firstrun_device_ui import qapp, _teardown, _wizard  # noqa: F401


def _tick_english(wiz) -> None:
    """Satisfy the proceed gate, which needs a spoken language picked."""
    for i in range(wiz._spoken_list.count()):
        item = wiz._spoken_list.item(i)
        if item.text() == "English":
            item.setCheckState(Qt.CheckState.Checked)
            return
    raise AssertionError("English missing from the picker")


def test_plan_inputs_freeze_while_the_download_runs(qapp, tmp_path, monkeypatch):
    """The worker re-reads recommended_whisper/recommended_mt between its two
    fetches, but _apply_recommendation already froze that pair into config.
    A control left live would fetch one model and leave config naming the other,
    and startup would then point the engine at a directory with no model.bin."""
    wiz, _store_, _dm, bridge = _wizard(tmp_path, monkeypatch, tier="gpu_high")
    try:
        _tick_english(wiz)
        monkeypatch.setattr(wiz, "_download_body", lambda: None)

        wiz._on_download_and_start()

        assert not wiz._device_choice.isEnabled()
        assert not wiz._spoken_list.isEnabled()
        assert not wiz._target_combo.isEnabled()
    finally:
        _teardown(wiz, bridge)


def test_a_failed_download_hands_the_controls_back(qapp, tmp_path, monkeypatch):
    # Frozen for the download is right; frozen for good after it fails is not.
    from PySide6.QtWidgets import QMessageBox

    wiz, _store_, _dm, bridge = _wizard(tmp_path, monkeypatch, tier="gpu_high")
    try:
        _tick_english(wiz)
        monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: None)
        wiz._set_buttons_enabled(False)

        wiz._on_download_done(False, "network went away")

        assert wiz._device_choice.isEnabled()
        assert wiz._spoken_list.isEnabled()
        assert wiz._target_combo.isEnabled()
        assert wiz._cancel_btn.isEnabled()
    finally:
        _teardown(wiz, bridge)
