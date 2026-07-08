"""Offscreen Qt tests for the first-run wizard's CPU/GPU device choice:
CPU default, live preset refresh, device writes on both accept paths, and
the disabled GPU segment on CPU-only machines.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

from tests.test_firstrun_ui import _FakeDownloadManager, _bridge, _store
from vrcc.core import recommend
from vrcc.stt.registry import WHISPER_MODELS


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _wizard(tmp_path, monkeypatch, tier="gpu_high", default_choice="cpu"):
    from vrcc.gui.firstrun import FirstRunWizard

    monkeypatch.setattr(recommend, "detect_tier", lambda: tier)
    # Pin the VRAM-driven default so tests are deterministic on any machine.
    monkeypatch.setattr(recommend, "default_device_choice", lambda: default_choice)
    store = _store(tmp_path)
    dm = _FakeDownloadManager(tmp_path / "models")
    bridge = _bridge()
    return FirstRunWizard(store, dm, bridge), store, dm, bridge


def _teardown(wiz, bridge) -> None:
    wiz.close()
    wiz.deleteLater()
    bridge.detach()


def test_firstrun_defaults_to_cpu_and_download_path_writes_cpu_device(
    qapp, tmp_path, monkeypatch
):
    wiz, store, dm, bridge = _wizard(tmp_path, monkeypatch, tier="gpu_high")
    try:
        assert wiz._device_choice.value() == "CPU"
        wiz._apply_recommendation()
        wiz._download_body()  # synchronous download of the recommended models
        assert store.config.stt.device == "cpu"
        assert store.config.translate.device == "cpu"
        assert dm.is_whisper_downloaded(store.config.stt.model)
    finally:
        _teardown(wiz, bridge)


def test_firstrun_cpu_choice_shows_cpu_preset_despite_gpu_tier(
    qapp, tmp_path, monkeypatch
):
    wiz, _store_, _dm, bridge = _wizard(tmp_path, monkeypatch, tier="gpu_high")
    try:
        cpu_label = WHISPER_MODELS[recommend.PRESETS["cpu"][0]].label
        gpu_label = WHISPER_MODELS[recommend.PRESETS["gpu_high"][0]].label
        text = wiz._summary_label.text()
        assert f"Speech: {cpu_label}" in text
        assert gpu_label not in text
    finally:
        _teardown(wiz, bridge)


def test_firstrun_gpu_choice_updates_lines_and_writes_auto(qapp, tmp_path, monkeypatch):
    wiz, store, _dm, bridge = _wizard(tmp_path, monkeypatch, tier="gpu_high")
    try:
        wiz._device_choice.set_value("GPU")
        gpu_label = WHISPER_MODELS[recommend.PRESETS["gpu_high"][0]].label
        assert f"Speech: {gpu_label}" in wiz._summary_label.text()
        assert "Total download:" in wiz._summary_label.text()
        wiz._apply_recommendation()
        assert store.config.stt.device == "auto"
        assert store.config.translate.device == "auto"
    finally:
        _teardown(wiz, bridge)


def test_firstrun_gpu_segment_disabled_when_no_gpu(qapp, tmp_path, monkeypatch):
    wiz, _store_, _dm, bridge = _wizard(tmp_path, monkeypatch, tier="cpu")
    try:
        gpu_btn = wiz._device_choice._buttons["GPU"]
        assert not gpu_btn.isEnabled()
        assert gpu_btn.toolTip() == "No graphics card detected."
        assert wiz._device_choice._buttons["CPU"].isEnabled()
        assert wiz._device_choice.value() == "CPU"
    finally:
        _teardown(wiz, bridge)


def test_firstrun_explainer_mentions_memory_tradeoff(qapp, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QLabel

    wiz, _store_, _dm, bridge = _wizard(tmp_path, monkeypatch)
    try:
        texts = [lbl.text() for lbl in wiz.findChildren(QLabel)]
        assert any("use more memory" in t for t in texts)
    finally:
        _teardown(wiz, bridge)


def test_firstrun_run_on_tooltip_explains_vram_tradeoff(qapp, tmp_path, monkeypatch):
    wiz, _store_, _dm, bridge = _wizard(tmp_path, monkeypatch, tier="gpu_high")
    try:
        tip = wiz._device_choice.toolTip()
        assert "VRAM" in tip
        assert "leaves your graphics card alone" in tip
    finally:
        _teardown(wiz, bridge)


def test_choose_manually_auto_select_uses_cpu_tier_and_writes_device(
    qapp, tmp_path, monkeypatch
):
    """CPU choice on a GPU machine: the auto-select path must pick from the
    cpu preference order (not the detected gpu tier) and write cpu devices."""
    wiz, store, dm, bridge = _wizard(tmp_path, monkeypatch, tier="gpu_high")
    store.config.stt.model = "large-v3"  # configured, but NOT downloaded
    store.config.translate.model = "nllb-3.3B-int8"

    def fake_exec(self):
        # Both the cpu preset pair and the (higher gpu-preference) gpu_high
        # pair are on disk -- the cpu order must win under the CPU choice.
        dm.downloaded.update(
            {"small", "nllb-600M-int8", "large-v3-turbo", "nllb-1.3B-int8"}
        )
        return 0

    monkeypatch.setattr("vrcc.gui.models_dialog.ModelsDialog.exec", fake_exec)
    accepted: list[bool] = []
    monkeypatch.setattr(wiz, "accept", lambda: accepted.append(True))
    try:
        wiz._on_choose_manually()
        assert accepted == [True]
        assert store.config.stt.model == "small"
        assert store.config.translate.model == "nllb-600M-int8"
        assert store.config.stt.device == "cpu"
        assert store.config.translate.device == "cpu"
    finally:
        _teardown(wiz, bridge)


def test_choose_manually_existing_models_early_return_writes_device(
    qapp, tmp_path, monkeypatch
):
    """Fresh-install defaults ARE the cpu preset: a user who downloads exactly
    those via the Models dialog hits the configured-models-present early
    return -- the visible CPU choice must still be written, never ignored."""
    wiz, store, dm, bridge = _wizard(tmp_path, monkeypatch, tier="gpu_high")
    dm.downloaded.update({store.config.stt.model, store.config.translate.model})
    monkeypatch.setattr("vrcc.gui.models_dialog.ModelsDialog.exec", lambda self: 0)
    accepted: list[bool] = []
    monkeypatch.setattr(wiz, "accept", lambda: accepted.append(True))
    try:
        wiz._on_choose_manually()
        assert accepted == [True]
        assert store.config.stt.device == "cpu"
        assert store.config.translate.device == "cpu"
    finally:
        _teardown(wiz, bridge)


def test_choose_manually_existing_models_gpu_choice_writes_auto(
    qapp, tmp_path, monkeypatch
):
    wiz, store, dm, bridge = _wizard(tmp_path, monkeypatch, tier="gpu_high")
    # A stale forced-CPU device must be overwritten by the GPU choice.
    store.config.stt.device = store.config.translate.device = "cpu"
    dm.downloaded.update({store.config.stt.model, store.config.translate.model})
    wiz._device_choice.set_value("GPU")
    monkeypatch.setattr("vrcc.gui.models_dialog.ModelsDialog.exec", lambda self: 0)
    accepted: list[bool] = []
    monkeypatch.setattr(wiz, "accept", lambda: accepted.append(True))
    try:
        wiz._on_choose_manually()
        assert accepted == [True]
        assert store.config.stt.device == "auto"
        assert store.config.translate.device == "auto"
    finally:
        _teardown(wiz, bridge)


def test_choose_manually_auto_select_gpu_choice_writes_auto(
    qapp, tmp_path, monkeypatch
):
    wiz, store, dm, bridge = _wizard(tmp_path, monkeypatch, tier="gpu_high")
    store.config.stt.model = "large-v3"  # configured, but NOT downloaded
    store.config.translate.model = "nllb-3.3B-int8"
    wiz._device_choice.set_value("GPU")

    def fake_exec(self):
        dm.downloaded.update({"large-v3-turbo", "nllb-1.3B-int8"})
        return 0

    monkeypatch.setattr("vrcc.gui.models_dialog.ModelsDialog.exec", fake_exec)
    accepted: list[bool] = []
    monkeypatch.setattr(wiz, "accept", lambda: accepted.append(True))
    try:
        wiz._on_choose_manually()
        assert accepted == [True]
        assert store.config.stt.model == "large-v3-turbo"
        assert store.config.translate.model == "nllb-1.3B-int8"
        assert store.config.stt.device == "auto"
        assert store.config.translate.device == "auto"
    finally:
        _teardown(wiz, bridge)


def test_firstrun_defaults_to_gpu_on_16gb_card(qapp, tmp_path, monkeypatch):
    wiz, _store_, _dm, bridge = _wizard(
        tmp_path, monkeypatch, tier="gpu_high", default_choice="gpu"
    )
    try:
        assert wiz._device_choice.value() == "GPU"
        # And the recommended preset follows the GPU tier, not the CPU one.
        assert (wiz.recommended_whisper, wiz.recommended_mt) == recommend.PRESETS[
            "gpu_high"
        ]
    finally:
        _teardown(wiz, bridge)


def test_device_choice_mirrors_into_config_immediately(qapp, tmp_path, monkeypatch):
    """The Models dialog's tier badge reads config -- the wizard's visible
    choice must be written at construction and on every change, not only on
    accept, so the badge can never recommend the gpu tier after a CPU pick."""
    wiz, store, _dm, bridge = _wizard(tmp_path, monkeypatch, tier="gpu_high")
    try:
        assert store.config.stt.device == "cpu"  # default mirrored at construction
        assert recommend.tier_for_config(store.config) == "cpu"
        wiz._device_choice.set_value("GPU")
        assert store.config.stt.device == "auto"
        assert store.config.translate.device == "auto"
        wiz._device_choice.set_value("CPU")
        assert recommend.tier_for_config(store.config) == "cpu"
    finally:
        _teardown(wiz, bridge)
