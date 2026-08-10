"""What the Models window says before and during a download.

Three states used to be silent: a download in flight refused to close the
window and greyed every button with nothing on screen to say why, a model too
large for the card was offered with only its size, and a translation model that
cannot write one of the configured target languages looked like every other one.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import threading

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from vrcc.core.bus import EventBus
from vrcc.core.config import ConfigStore, default_paths


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


class _FakeDM:
    def __init__(self, models_dir=None):
        self.downloaded = set()
        self.models_dir = models_dir

    def is_whisper_downloaded(self, mid):
        return mid in self.downloaded

    def is_mt_downloaded(self, spec):
        return spec.id in self.downloaded

    def ensure_whisper(self, mid):
        self.downloaded.add(mid)

    def ensure_mt(self, spec):
        self.downloaded.add(spec.id)

    def delete(self, kind, mid):
        self.downloaded.discard(mid)


class _NoThread:
    """Runs nothing: the tests want the in-flight UI state, not the download."""

    def __init__(self, target=None, name=None, daemon=None):
        self._target = target

    def start(self):
        pass


def _dlg(tmp_path, targets=None, spoken=None):
    from vrcc.gui.bridge import BusBridge
    from vrcc.gui.models_dialog import ModelsDialog

    store = ConfigStore(default_paths(portable=True, app_dir=tmp_path).config_file)
    store.config.stt.source_language = "auto"
    store.config.stt.spoken_languages = list(spoken or [])
    if targets is not None:
        store.config.translate.targets = list(targets)
    bridge = BusBridge(EventBus())
    dlg = ModelsDialog(_FakeDM(tmp_path / "models"), bridge, config_store=store)
    return dlg, store, bridge


def _row(dlg, model_id):
    return next(r for r in dlg._rows if r.model_id == model_id)


# -- download in flight ------------------------------------------------------


def test_download_in_flight_says_why_close_is_off(qapp, tmp_path, monkeypatch):
    monkeypatch.setattr(threading, "Thread", _NoThread)
    dlg, _store, bridge = _dlg(tmp_path)
    try:
        assert dlg._close_btn.isEnabled()
        assert not dlg._status.isVisibleTo(dlg)

        row = _row(dlg, "small")
        dlg._start_download(row)

        assert not dlg._close_btn.isEnabled()
        assert dlg._status.isVisibleTo(dlg)
        text = dlg._status.text()
        assert row.display_name in text
        assert "cannot be stopped" in text
        assert "Close" in text
    finally:
        dlg._downloading_id = None  # release the close guard for teardown
        dlg.close()
        dlg.deleteLater()
        bridge.detach()


def test_finished_download_restores_close_and_clears_the_status(
    qapp, tmp_path, monkeypatch
):
    monkeypatch.setattr(threading, "Thread", _NoThread)
    dlg, _store, bridge = _dlg(tmp_path)
    try:
        dlg._start_download(_row(dlg, "small"))
        dlg._on_op_finished("small", True, "")

        assert dlg._close_btn.isEnabled()
        assert not dlg._status.isVisibleTo(dlg)
    finally:
        dlg.close()
        dlg.deleteLater()
        bridge.detach()


# -- graphics-card fit -------------------------------------------------------


def test_row_marks_a_model_too_big_for_the_card(qapp, tmp_path, monkeypatch):
    from vrcc.gui import model_fit

    monkeypatch.setattr(
        model_fit,
        "vram_warning",
        lambda size_mb, device="auto", model_id=None, device_index=0,
               compute_type="auto": (
            "This model may be too large for your graphics card."
            if size_mb >= 1000
            else None
        ),
    )
    dlg, _store, bridge = _dlg(tmp_path)
    try:
        big = _row(dlg, "large-v3")
        small = _row(dlg, "tiny")
        assert big._note.isVisibleTo(dlg)
        assert "too large for your graphics card" in big._note.text()
        assert not small._note.isVisibleTo(dlg)
    finally:
        dlg.close()
        dlg.deleteLater()
        bridge.detach()


# -- collapsed translation target --------------------------------------------


def test_row_warns_when_the_model_cannot_write_a_configured_target(qapp, tmp_path):
    dlg, _store, bridge = _dlg(tmp_path, targets=["Chinese Traditional"])
    try:
        m2m = _row(dlg, "m2m100-418M-int8")
        nllb = _row(dlg, "nllb-600M-int8")
        assert m2m._note.isVisibleTo(dlg)
        assert "Chinese Traditional" in m2m._note.text()
        assert "Chinese Simplified" in m2m._note.text()
        assert not nllb._note.isVisibleTo(dlg)
    finally:
        dlg.close()
        dlg.deleteLater()
        bridge.detach()


def test_no_collapse_warning_when_no_target_is_affected(qapp, tmp_path):
    dlg, _store, bridge = _dlg(tmp_path, targets=["Japanese"])
    try:
        assert not _row(dlg, "m2m100-418M-int8")._note.isVisibleTo(dlg)
    finally:
        dlg.close()
        dlg.deleteLater()
        bridge.detach()


# -- empty language picker ---------------------------------------------------


def test_hint_shows_while_no_language_is_ticked(qapp, tmp_path):
    dlg, _store, bridge = _dlg(tmp_path)
    try:
        assert dlg._spoken_hint.isVisibleTo(dlg)
        assert "detect" in dlg._spoken_hint.text()
    finally:
        dlg.close()
        dlg.deleteLater()
        bridge.detach()


def test_hint_agrees_with_the_source_actually_in_force(qapp, tmp_path):
    # The hint reads the config after the tick is resolved, so whichever way
    # an empty picker is treated (detect, or keep the stored language) the
    # sentence on screen matches what the engines will run.
    dlg, store, bridge = _dlg(tmp_path, spoken=["German"])
    try:
        store.config.stt.source_language = "German"
        picker = dlg._spoken_list
        item = next(
            picker.item(i)
            for i in range(picker.count())
            if picker.item(i).text() == "German"
        )
        item.setCheckState(Qt.CheckState.Unchecked)

        text = dlg._spoken_hint.text()
        assert dlg._spoken_hint.isVisibleTo(dlg)
        source = store.config.stt.source_language
        if source == "auto":
            assert "detect" in text
        else:
            assert source in text
    finally:
        dlg.close()
        dlg.deleteLater()
        bridge.detach()


def test_hint_hides_once_a_language_is_ticked_and_returns_on_untick(qapp, tmp_path):
    dlg, _store, bridge = _dlg(tmp_path)
    try:
        picker = dlg._spoken_list
        item = next(
            picker.item(i)
            for i in range(picker.count())
            if picker.item(i).text() == "Japanese"
        )
        item.setCheckState(Qt.CheckState.Checked)
        assert not dlg._spoken_hint.isVisibleTo(dlg)

        item.setCheckState(Qt.CheckState.Unchecked)
        assert dlg._spoken_hint.isVisibleTo(dlg)
    finally:
        dlg.close()
        dlg.deleteLater()
        bridge.detach()
