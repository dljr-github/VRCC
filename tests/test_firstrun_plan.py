"""The first-run wizard's plan summary: where the voice model will actually
run, what is still left to download, and the disk check in front of the fetch.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

from tests.test_firstrun_device_ui import _teardown, _wizard
from vrcc.core import hardware
from vrcc.gui import firstrun_plan, model_fit
from vrcc.stt.registry import WHISPER_MODELS
from vrcc.translate.registry import MT_MODELS


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _gpu_wizard(tmp_path, monkeypatch, language="Japanese"):
    """A GPU-choice wizard on a machine whose CUDA is usable, speaking
    ``language``. Japanese lands the plan on SenseVoice, which is the case the
    Run-on control cannot honour."""
    monkeypatch.setattr(hardware, "can_run_cuda", lambda: True)
    wiz, store, dm, bridge = _wizard(
        tmp_path, monkeypatch, tier="gpu_high", default_choice="gpu"
    )
    store.config.stt.source_language = language
    from tests.test_firstrun_ui import _tick

    _tick(wiz, language, only=True)
    return wiz, store, dm, bridge


def test_the_gpu_choice_never_plans_a_processor_only_voice_model(
    qapp, tmp_path, monkeypatch
):
    """The contradiction voice_device_note exists to explain: stt.device="auto"
    resolves the onnxruntime models to the processor even on a usable card, so
    picking GPU and being handed one would leave Settings reading "Auto: using
    your processor" right after.

    Since the CJK presets moved to faster-whisper no reachable plan does that,
    which is a stronger guarantee than explaining it after the fact. Pinned
    across every spoken language rather than asserted for one, because it is
    the preset tables that make it true and they change.
    """
    from vrcc.core import recommend
    from vrcc.core.languages import LANGUAGES
    from vrcc.stt.registry import WHISPER_MODELS

    for tier in ("gpu_high", "gpu_low"):
        for code in sorted({lang.whisper for lang in LANGUAGES.values()}):
            whisper, _mt = recommend.preset_for_choice(
                "gpu", tier, (code,), 1.0, 24 * 1024
            )
            assert not WHISPER_MODELS[whisper].runs_on_onnxruntime, (
                tier, code, whisper
            )


def test_no_processor_note_under_the_cpu_choice(qapp, tmp_path, monkeypatch):
    # Nothing is being contradicted there: the user asked for the processor,
    # and the cpu tier does still plan Parakeet for the languages it covers.
    wiz, _store_, _dm, bridge = _gpu_wizard(tmp_path, monkeypatch)
    try:
        wiz._device_choice.set_value("CPU")
        assert firstrun_plan.voice_device_note(wiz) is None
    finally:
        _teardown(wiz, bridge)

def test_total_download_counts_only_what_is_missing(qapp, tmp_path, monkeypatch):
    """"Choose existing models" can fetch half the plan and come back with the
    wizard still open; quoting the whole plan there names a download that is
    mostly done."""
    wiz, _store_, dm, bridge = _wizard(tmp_path, monkeypatch, tier="cpu")
    try:
        whisper_mb = WHISPER_MODELS[wiz.recommended_whisper].size_mb
        mt_mb = MT_MODELS[wiz.recommended_mt].size_mb
        assert firstrun_plan.pending_mb(wiz) == whisper_mb + mt_mb

        dm.downloaded.add(wiz.recommended_mt)
        wiz._refresh_plan()

        assert firstrun_plan.pending_mb(wiz) == whisper_mb
        from vrcc.gui.model_labels import fmt_size

        assert f"Total download: {fmt_size(whisper_mb)}" in wiz._summary_label.text()
    finally:
        _teardown(wiz, bridge)


# -- disk space ---------------------------------------------------------------


def _capture_disk_check(monkeypatch, message):
    """Stand in for model_fit.disk_warning, recording the size it was asked
    about and returning ``message``."""
    asked: list[int] = []

    def fake(models_dir, size_mb):
        asked.append(size_mb)
        return message

    monkeypatch.setattr(model_fit, "disk_warning", fake)
    return asked


def test_download_is_refused_when_the_disk_cannot_hold_the_pair(
    qapp, tmp_path, monkeypatch
):
    """Without this the fetch starts on a full disk and ends in a raw
    downloader traceback."""
    from PySide6.QtWidgets import QMessageBox

    wiz, _store_, _dm, bridge = _wizard(tmp_path, monkeypatch, tier="cpu")
    try:
        asked = _capture_disk_check(monkeypatch, "no room")
        monkeypatch.setattr(
            QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.No
        )
        started: list[bool] = []
        monkeypatch.setattr(wiz, "_download_body", lambda: started.append(True))

        wiz._on_download_and_start()

        assert started == []
        assert not wiz._downloading
        assert not wiz._whisper_bar.isVisibleTo(wiz)
        # Both halves of the plan, not one model at a time like the Models window.
        assert asked == [firstrun_plan.pending_mb(wiz)]
        assert asked[0] > MT_MODELS[wiz.recommended_mt].size_mb
    finally:
        _teardown(wiz, bridge)


def test_download_proceeds_when_the_low_disk_warning_is_accepted(
    qapp, tmp_path, monkeypatch
):
    from PySide6.QtWidgets import QMessageBox

    wiz, store, _dm, bridge = _wizard(tmp_path, monkeypatch, tier="cpu")
    try:
        _capture_disk_check(monkeypatch, "no room")
        monkeypatch.setattr(
            QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Yes
        )
        monkeypatch.setattr(wiz, "_download_body", lambda: None)

        wiz._on_download_and_start()

        assert wiz._downloading
        assert store.config.stt.model == wiz.recommended_whisper
    finally:
        _teardown(wiz, bridge)


def test_a_roomy_disk_asks_nothing(qapp, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QMessageBox

    wiz, _store_, _dm, bridge = _wizard(tmp_path, monkeypatch, tier="cpu")
    try:
        _capture_disk_check(monkeypatch, None)
        asked: list[tuple] = []
        monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: asked.append(a))
        monkeypatch.setattr(wiz, "_download_body", lambda: None)

        wiz._on_download_and_start()

        assert asked == []
        assert wiz._downloading
    finally:
        _teardown(wiz, bridge)


# -- greyed target languages --------------------------------------------------


def test_the_default_plan_can_write_chinese_traditional(qapp, tmp_path, monkeypatch):
    """The wizard has no model picker, so a target its plan cannot write is a
    dead end. NLLB renders both Chinese scripts, so on the shipped preset the
    entry is simply available."""
    wiz, store, _dm, bridge = _wizard(tmp_path, monkeypatch, tier="cpu")
    try:
        assert store.config.translate.model.startswith("nllb")
        combo = wiz._target_combo
        item = combo.model().item(combo.findText("Chinese Traditional"))

        assert item.isEnabled()
        assert item.toolTip() == ""
    finally:
        _teardown(wiz, bridge)
