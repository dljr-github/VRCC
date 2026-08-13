"""What the window does when an engine fails to load.

Untested until now, which is how a real bug shipped alongside the fix: capture
correctly stopped depending on the translator, but the dead translator stayed
attached to the pipeline, so every utterance raised inside translate().

Four promises ride on this handler, and each is a way the app can lie about its
own state:
- a dead TRANSLATOR must not claim capture stopped. Transcription is still
  running, and no later event repaints that label, so the claim would stand for
  the rest of the session.
- a dead VOICE MODEL must claim exactly that, because nothing is being captured.
- a kind that republishes "failed" must not stack one modal per attempt.
- going "ready" must re-arm the report, or a model that recovers and fails
  again dies silently.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QMessageBox

from tests.test_app_main_window import _main_window, _store, qapp  # noqa: F401
from vrcc.core.events import EngineStateChanged


@pytest.fixture
def silent_modal(monkeypatch):
    """Swallow the failure modal and record its bodies. Without this the
    offscreen suite HANGS on the first failure rather than failing."""
    shown: list[str] = []
    monkeypatch.setattr(
        QMessageBox, "warning",
        lambda *a, **k: shown.append(a[2]) or QMessageBox.StandardButton.Ok,
    )
    return shown


def _fail(window, engine: str, detail: str = "CUDA out of memory") -> None:
    window._on_engine_state(EngineStateChanged(engine, "failed", detail))


def test_a_dead_translator_leaves_the_capture_label_alone(qapp, tmp_path, silent_modal):
    window, bridge = _main_window(_store(tmp_path))
    try:
        before = window._capture_label.text()

        _fail(window, "mt")

        assert window._capture_label.text() == before, (
            "a failed translator must not claim VRCC stopped listening"
        )
        assert silent_modal, "but it must still say something"
    finally:
        window.close()
        bridge.detach()


def test_a_dead_voice_model_does_say_capture_stopped(qapp, tmp_path, silent_modal):
    window, bridge = _main_window(_store(tmp_path))
    try:
        _fail(window, "stt")

        assert "not listening" in window._capture_label.text().lower()
    finally:
        window.close()
        bridge.detach()


def test_the_failure_modal_names_a_next_step(qapp, tmp_path, silent_modal):
    window, bridge = _main_window(_store(tmp_path))
    try:
        _fail(window, "stt")

        assert silent_modal
        body = silent_modal[0]
        assert "Models" in body, "a failure with no way out is not actionable"
        assert "CUDA out of memory" in body, "the cause is the useful half"
    finally:
        window.close()
        bridge.detach()


def test_a_repeated_failure_does_not_stack_one_dialog_per_attempt(
    qapp, tmp_path, silent_modal
):
    window, bridge = _main_window(_store(tmp_path))
    try:
        for _ in range(4):
            _fail(window, "stt")

        assert len(silent_modal) == 1
    finally:
        window.close()
        bridge.detach()


def test_recovering_re_arms_the_report(qapp, tmp_path, silent_modal):
    """A model that fails, is fixed, then fails again must be heard the second
    time. Suppressing forever would be worse than the stacking it prevents."""
    window, bridge = _main_window(_store(tmp_path))
    try:
        _fail(window, "stt")
        window._on_engine_state(EngineStateChanged("stt", "ready", "cpu:int8"))
        _fail(window, "stt")

        assert len(silent_modal) == 2
    finally:
        window.close()
        bridge.detach()


def test_the_empty_feed_stops_inviting_speech_after_a_voice_failure(
    qapp, tmp_path, silent_modal
):
    window, bridge = _main_window(_store(tmp_path))
    try:
        _fail(window, "stt")

        text = window._log.toPlainText().lower()
        assert "say something" not in text, (
            "asking someone to talk to a model that never loaded wastes their time"
        )
        assert "models" in text
    finally:
        window.close()
        bridge.detach()


# -- _retire_failed_engines -------------------------------------------------


class _MtPipe:
    """Only what _retire_failed_engines touches, with the real Pipeline's
    meaning of mt_active (``self._mt is not None``)."""

    def __init__(self):
        self._mt = object()

    @property
    def mt_active(self):
        return self._mt is not None

    def set_mt(self, engine):
        self._mt = engine


def test_a_translator_that_never_loaded_comes_off_the_pipeline():
    """Captions now start without it, so the engine whose load() raised must be
    unhooked. Left attached, mt_active reads True and every utterance raises
    inside translate() for the rest of the session."""
    from vrcc.app import _FAILED, _retire_failed_engines

    pipe = _MtPipe()
    loaded = {"stt": "small", "mt": "nllb-600M-int8"}

    _retire_failed_engines({"mt"}, loaded, dict(loaded), pipe)

    assert not pipe.mt_active
    assert loaded["mt"] is _FAILED
    assert loaded["stt"] == "small", "the healthy kind is untouched"


def test_a_swap_won_during_the_load_window_keeps_its_live_engine():
    """The reason both actions sit behind the startup-id guard: if the user
    swapped models while the startup load was still running, the engine now
    installed is theirs and is alive."""
    from vrcc.app import _retire_failed_engines

    pipe = _MtPipe()
    live = pipe._mt
    loaded = {"stt": "small", "mt": "m2m100-418M-int8"}  # swapped since startup

    _retire_failed_engines({"mt"}, loaded, {"stt": "small", "mt": "nllb-600M-int8"}, pipe)

    assert pipe._mt is live
    assert loaded["mt"] == "m2m100-418M-int8"


def test_a_dead_voice_model_does_not_touch_the_translator():
    from vrcc.app import _FAILED, _retire_failed_engines

    pipe = _MtPipe()
    loaded = {"stt": "small", "mt": "nllb-600M-int8"}

    _retire_failed_engines({"stt"}, loaded, dict(loaded), pipe)

    assert pipe.mt_active
    assert loaded["stt"] is _FAILED
