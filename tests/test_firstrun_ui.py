"""Offscreen Qt smoke tests: Settings/Models dialog construction and the
FirstRunWizard flow (recommendation, download, "choose manually").
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path

import pytest

from vrcc.core.bus import EventBus
from vrcc.core.config import ConfigStore, default_paths
from vrcc.translate.registry import MT_MODELS


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


class _FakeDownloadManager:
    """Minimal stand-in for DownloadManager: tracks a set of downloaded ids."""

    def __init__(self, models_dir: Path) -> None:
        self._dir = Path(models_dir)
        self.models_dir = self._dir  # exposed like the real DownloadManager
        self.downloaded: set[str] = set()

    def is_whisper_downloaded(self, model_id: str) -> bool:
        return model_id in self.downloaded

    def is_mt_downloaded(self, spec) -> bool:
        return spec.id in self.downloaded

    def ensure_whisper(self, model_id: str) -> Path:
        self.downloaded.add(model_id)
        return self.whisper_model_dir(model_id)

    def ensure_mt(self, spec) -> Path:
        self.downloaded.add(spec.id)
        return self.mt_model_dir(spec)

    def delete(self, kind: str, model_id: str) -> None:
        self.downloaded.discard(model_id)

    def whisper_model_dir(self, model_id: str) -> Path:
        return self._dir / "whisper" / model_id

    def mt_model_dir(self, spec) -> Path:
        return self._dir / "mt" / spec.id


def _store(tmp_path: Path) -> ConfigStore:
    return ConfigStore(default_paths(portable=True, app_dir=tmp_path).config_file)


def _bridge():
    from vrcc.gui.bridge import BusBridge

    return BusBridge(EventBus())


def test_settings_dialog_constructs(qapp, tmp_path):
    from vrcc.gui.settings import SettingsDialog

    store = _store(tmp_path)
    dlg = SettingsDialog(store)
    try:
        assert dlg.windowTitle()
    finally:
        dlg.close()
        dlg.deleteLater()


def test_models_dialog_constructs(qapp, tmp_path):
    from vrcc.gui.models_dialog import ModelsDialog

    dm = _FakeDownloadManager(tmp_path / "models")
    bridge = _bridge()
    dlg = ModelsDialog(dm, bridge)
    try:
        # One row per whisper + MT model.
        from vrcc.stt.registry import WHISPER_MODELS

        assert len(dlg._rows) == len(WHISPER_MODELS) + len(MT_MODELS)
    finally:
        dlg.close()
        dlg.deleteLater()
        bridge.detach()


def test_firstrun_wizard_constructs(qapp, tmp_path):
    from vrcc.gui.firstrun import FirstRunWizard

    store = _store(tmp_path)
    dm = _FakeDownloadManager(tmp_path / "models")
    bridge = _bridge()
    wiz = FirstRunWizard(store, dm, bridge)
    try:
        # A recommendation was computed (whisper id + optional mt id).
        assert wiz.recommended_whisper
        assert wiz.recommended_mt
    finally:
        wiz.close()
        wiz.deleteLater()
        bridge.detach()


def test_firstrun_headline_scales_with_font_scale(qapp, tmp_path):
    # The 20px headline must follow the text-size preset like caption_log
    # does (at Large it used to render smaller than the scaled body).
    from vrcc.gui.firstrun import FirstRunWizard

    store = _store(tmp_path)
    store.config.gui.font_scale = 1.2
    dm = _FakeDownloadManager(tmp_path / "models")
    bridge = _bridge()
    wiz = FirstRunWizard(store, dm, bridge)
    try:
        assert f"font-size: {round(20 * 1.2)}px" in wiz._headline.styleSheet()
    finally:
        wiz.close(); wiz.deleteLater(); bridge.detach()


def test_firstrun_wizard_has_language_pickers(qapp, tmp_path):
    from vrcc.gui.firstrun import FirstRunWizard

    store = _store(tmp_path)
    dm = _FakeDownloadManager(tmp_path / "models")
    bridge = _bridge()
    wiz = FirstRunWizard(store, dm, bridge)
    try:
        wiz._source_combo.setCurrentText("Japanese")
        assert store.config.stt.source_language == "Japanese"
    finally:
        wiz.close(); wiz.deleteLater(); bridge.detach()


def test_firstrun_bars_hidden_before_download(qapp, tmp_path):
    """Design review: the wizard used to show two empty progress bars before
    any download started -- both must stay hidden until their download runs."""
    from vrcc.gui.firstrun import FirstRunWizard

    store = _store(tmp_path)
    dm = _FakeDownloadManager(tmp_path / "models")
    bridge = _bridge()
    wiz = FirstRunWizard(store, dm, bridge)
    try:
        assert not wiz._whisper_bar.isVisibleTo(wiz)
        assert not wiz._mt_bar.isVisibleTo(wiz)
    finally:
        wiz.close(); wiz.deleteLater(); bridge.detach()


def test_firstrun_summary_names_translation_model_in_gb_no_license(qapp, tmp_path):
    """Design review: the translation line had no model name and repeated the
    license (already stated once in the Note line below)."""
    from vrcc.gui.firstrun import FirstRunWizard
    from vrcc.gui.model_labels import mt_display_name

    store = _store(tmp_path)
    store.config.translate.enabled = True
    dm = _FakeDownloadManager(tmp_path / "models")
    bridge = _bridge()
    wiz = FirstRunWizard(store, dm, bridge)
    try:
        text = wiz._summary_label.text()
        assert mt_display_name(wiz.recommended_mt) in text
        assert "GB" in text or "MB" in text
        assert "Total download:" in text
        assert "license" not in text.lower()
    finally:
        wiz.close(); wiz.deleteLater(); bridge.detach()


def test_firstrun_license_mentioned_exactly_once(qapp, tmp_path):
    from PySide6.QtWidgets import QLabel

    from vrcc.gui.firstrun import FirstRunWizard

    store = _store(tmp_path)
    store.config.translate.enabled = True
    dm = _FakeDownloadManager(tmp_path / "models")
    bridge = _bridge()
    wiz = FirstRunWizard(store, dm, bridge)
    try:
        blob = " ".join(lbl.text().lower() for lbl in wiz.findChildren(QLabel))
        assert blob.count("license") == 1
    finally:
        wiz.close(); wiz.deleteLater(); bridge.detach()


def test_firstrun_download_button_is_primary(qapp, tmp_path):
    from vrcc.gui.firstrun import FirstRunWizard

    store = _store(tmp_path)
    dm = _FakeDownloadManager(tmp_path / "models")
    bridge = _bridge()
    wiz = FirstRunWizard(store, dm, bridge)
    try:
        assert wiz._download_btn.property("buttonRole") == "primary"
    finally:
        wiz.close(); wiz.deleteLater(); bridge.detach()


def test_firstrun_wizard_download_and_start_marks_models(qapp, tmp_path):
    """Driving the download path with a fake manager marks the recommended
    models present and applies them to config (no threads needed: we call the
    worker body synchronously)."""
    from vrcc.gui.firstrun import FirstRunWizard

    store = _store(tmp_path)
    dm = _FakeDownloadManager(tmp_path / "models")
    bridge = _bridge()
    wiz = FirstRunWizard(store, dm, bridge)
    try:
        wiz._apply_recommendation()
        wiz._download_body()  # synchronous download of the recommended models
        assert dm.is_whisper_downloaded(store.config.stt.model)
        spec = MT_MODELS[store.config.translate.model]
        assert dm.is_mt_downloaded(spec)
    finally:
        wiz.close()
        wiz.deleteLater()
        bridge.detach()


def test_choose_manually_does_not_overwrite_config(qapp, tmp_path, monkeypatch):
    """Keep-invariant: when the user's *configured* models are already present,
    "Choose manually" must start with them and never rewrite config to the
    recommended preset (a user must be able to run a model they already had)."""
    from vrcc.core import recommend
    from vrcc.gui.firstrun import FirstRunWizard

    monkeypatch.setattr(recommend, "detect_tier", lambda: "gpu_high")
    store = _store(tmp_path)
    store.config.stt.model = "tiny"  # user's own pick, not the recommendation
    store.config.translate.model = "nllb-600M-int8"
    dm = _FakeDownloadManager(tmp_path / "models")
    dm.downloaded.update({"tiny", "nllb-600M-int8"})  # configured models present
    # Also downloaded: the gpu_high preset pair, which outranks the picks above
    # in WHISPER_PREFERENCE/MT_PREFERENCE -- if the early-return invariant ever
    # regressed, best_downloaded would pick these instead and the assertions
    # below would catch it.
    dm.downloaded.update({"large-v3-turbo", "nllb-1.3B-int8"})
    bridge = _bridge()
    wiz = FirstRunWizard(store, dm, bridge)
    # GPU choice keeps the gpu_high preference order live, so the pair above
    # still ranks top and the early-return invariant stays load-bearing.
    wiz._device_choice.set_value("GPU")
    # the recommendation must differ from the user's pick for the test to bite
    assert wiz.recommended_whisper != "tiny"
    monkeypatch.setattr("vrcc.gui.models_dialog.ModelsDialog.exec", lambda self: 0)
    try:
        wiz._on_choose_manually()
        assert store.config.stt.model == "tiny"
        assert store.config.translate.model == "nllb-600M-int8"
    finally:
        wiz.close()
        wiz.deleteLater()
        bridge.detach()


def test_choose_manually_auto_selects_best_downloaded(qapp, tmp_path, monkeypatch):
    """Configured models absent but the user downloaded a different, usable set
    in the models dialog: config is pointed at the best downloaded pair and the
    wizard accepts (the same start path the download flow uses)."""
    from vrcc.gui.firstrun import FirstRunWizard

    store = _store(tmp_path)
    store.config.stt.model = "large-v3"  # configured, but NOT downloaded
    store.config.translate.model = "nllb-3.3B-int8"
    dm = _FakeDownloadManager(tmp_path / "models")
    bridge = _bridge()
    wiz = FirstRunWizard(store, dm, bridge)
    wiz.tier = "cpu"  # pin the tier so best_downloaded is deterministic

    def fake_exec(self):
        dm.downloaded.update({"small", "nllb-600M-int8"})  # user downloads these
        return 0

    monkeypatch.setattr("vrcc.gui.models_dialog.ModelsDialog.exec", fake_exec)
    accepted: list[bool] = []
    monkeypatch.setattr(wiz, "accept", lambda: accepted.append(True))
    try:
        wiz._on_choose_manually()
        assert accepted == [True]  # wizard started
        assert store.config.stt.model == "small"
        assert store.config.translate.model == "nllb-600M-int8"
    finally:
        wiz.close()
        wiz.deleteLater()
        bridge.detach()


def test_choose_manually_nothing_downloaded_stays_open(qapp, tmp_path, monkeypatch):
    """Configured models absent and nothing usable downloaded: the wizard must
    stay open (no accept), leave config untouched, and show a plain hint."""
    from PySide6.QtWidgets import QMessageBox

    from vrcc.gui.firstrun import FirstRunWizard

    store = _store(tmp_path)
    store.config.stt.model = "large-v3"  # configured, but NOT downloaded
    store.config.translate.model = "nllb-3.3B-int8"
    dm = _FakeDownloadManager(tmp_path / "models")  # nothing downloaded
    bridge = _bridge()
    wiz = FirstRunWizard(store, dm, bridge)
    monkeypatch.setattr("vrcc.gui.models_dialog.ModelsDialog.exec", lambda self: 0)
    accepted: list[bool] = []
    monkeypatch.setattr(wiz, "accept", lambda: accepted.append(True))
    hints: list[tuple] = []
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: hints.append(a))
    try:
        wiz._on_choose_manually()
        assert accepted == []  # wizard stays open
        assert hints  # a plain-language hint was shown
        assert store.config.stt.model == "large-v3"  # config untouched
        assert store.config.translate.model == "nllb-3.3B-int8"
    finally:
        wiz.close()
        wiz.deleteLater()
        bridge.detach()


def test_choose_manually_half_downloaded_stays_open(qapp, tmp_path, monkeypatch):
    """Configured models absent, translation enabled, and only a whisper model
    downloaded (no MT model): a voice model alone can't satisfy translation,
    so the wizard must stay open and leave config untouched."""
    from PySide6.QtWidgets import QMessageBox

    from vrcc.gui.firstrun import FirstRunWizard

    store = _store(tmp_path)
    store.config.stt.model = "large-v3"  # configured, but NOT downloaded
    store.config.translate.model = "nllb-3.3B-int8"
    dm = _FakeDownloadManager(tmp_path / "models")
    dm.downloaded.add("small")  # a whisper model downloaded, but no MT model
    bridge = _bridge()
    wiz = FirstRunWizard(store, dm, bridge)
    wiz.tier = "cpu"  # pin the tier so best_downloaded is deterministic
    monkeypatch.setattr("vrcc.gui.models_dialog.ModelsDialog.exec", lambda self: 0)
    accepted: list[bool] = []
    monkeypatch.setattr(wiz, "accept", lambda: accepted.append(True))
    hints: list[tuple] = []
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: hints.append(a))
    try:
        wiz._on_choose_manually()
        assert accepted == []  # wizard stays open
        assert hints  # a plain-language hint was shown
        assert store.config.stt.model == "large-v3"  # config untouched
        assert store.config.translate.model == "nllb-3.3B-int8"
    finally:
        wiz.close()
        wiz.deleteLater()
        bridge.detach()


def test_models_dialog_shows_active_downloaded_model_as_in_use(qapp, tmp_path):
    """A downloaded model that config already points at renders read-only
    ("In use"); the dialog itself never writes to the config store -- that
    only happens via Settings now."""
    from vrcc.gui.models_dialog import ModelsDialog

    store = _store(tmp_path)
    store.config.stt.model = "tiny"
    dm = _FakeDownloadManager(tmp_path / "models")
    dm.downloaded.add("tiny")  # a downloaded whisper model
    bridge = _bridge()
    dlg = ModelsDialog(dm, bridge, config_store=store)
    try:
        tiny_row = next(r for r in dlg._rows if r.model_id == "tiny")
        assert dlg._is_active(tiny_row)
    finally:
        dlg.close()
        dlg.deleteLater()
        bridge.detach()


def test_models_dialog_without_store_never_shows_active(qapp, tmp_path):
    from vrcc.gui.models_dialog import ModelsDialog

    dm = _FakeDownloadManager(tmp_path / "models")
    bridge = _bridge()
    dlg = ModelsDialog(dm, bridge)  # no config_store
    try:
        assert all(not dlg._is_active(r) for r in dlg._rows)
    finally:
        dlg.close()
        dlg.deleteLater()
        bridge.detach()
