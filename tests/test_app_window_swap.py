"""Offscreen Qt tests for ``app._swap_main_window``: the UI-language-change
rebuild path that must carry runtime state (VRChat detection, capture status,
mute state) across to the freshly built window.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path

import pytest

from vrcc.core.bus import EventBus
from vrcc.core.config import ConfigStore, default_paths


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _paths(tmp_path: Path):
    return default_paths(portable=True, app_dir=tmp_path)


def _store(tmp_path: Path) -> ConfigStore:
    return ConfigStore(_paths(tmp_path).config_file)


class _FakePipeline:
    """Minimal pipeline surface MainWindow needs."""

    def __init__(self, accept_typed: bool = True, mt_active: bool = True) -> None:
        self.captioning_enabled = True
        self.mute_gate = False
        self.typed: list[str] = []
        self._accept_typed = accept_typed
        self.mt_active = mt_active

    def submit_typed(self, text: str) -> bool:
        self.typed.append(text)
        return self._accept_typed

    def set_captioning(self, enabled: bool) -> None:
        self.captioning_enabled = bool(enabled)

    def mute_gated(self) -> bool:
        return self.mute_gate


def _rebuild_env(tmp_path):
    """Bridge, detector, pipeline and window factory for driving
    app._swap_main_window (the UI-language change path) for real."""
    from vrcc.gui.bridge import BusBridge
    from vrcc.gui.main_window import MainWindow
    from vrcc.osc.vrchat_detect import VrchatDetector

    class _Zc:
        def close(self) -> None:
            pass

    class _Browser:
        def cancel(self) -> None:
            pass

    bus = EventBus()
    bridge = BusBridge(bus)
    detector = VrchatDetector(
        bus, zeroconf_factory=_Zc, browser_factory=lambda zc, st, listener: _Browser()
    )
    store = _store(tmp_path)
    pipeline = _FakePipeline()

    def make_window():
        return MainWindow(
            bridge, store, pipeline,
            on_open_settings=lambda: None, on_open_models=lambda: None,
        )

    return bridge, detector, pipeline, make_window


def test_rebuilt_window_is_repushed_vrchat_state_and_capture_ok(qapp, tmp_path):
    # The detector publishes VrchatDetected only on transitions, so the fresh
    # window is told the current state via republish(); the capture label
    # carries over from the old window, and paused-vs-listening re-derives
    # from the captioning toggle, so a merely paused run renders amber
    # Paused instead of red failure or gray Starting.
    from vrcc.app import _swap_main_window

    bridge, detector, pipeline, make_window = _rebuild_env(tmp_path)
    pipeline.captioning_enabled = False  # the user paused before the rebuild

    old = make_window()
    fresh = None
    try:
        detector.start()
        detector.add_service(
            None, "_oscjson._tcp.local.", "VRChat-Client-7._oscjson._tcp.local."
        )
        assert "connected" in old._vrchat_label.text()
        old.set_capture_status(True)  # the pipeline started and is healthy

        fresh = _swap_main_window(old, make_window, detector, None)

        assert "connected" in fresh._vrchat_label.text()
        assert fresh._capture_label.text() == "Paused - not listening"
    finally:
        for w in (old, fresh):
            if w is not None:
                w.close()
                w.deleteLater()
        bridge.detach()


def test_rebuilt_window_keeps_a_capture_failure_red(qapp, tmp_path):
    # A failed engine paints red with a reason, whether from a mid-session
    # swap (pipeline started) or before capture ever started; the rebuild
    # must carry that verbatim rather than repaint green "Listening" over a
    # pipeline that is not captioning, or reset to gray "Starting".
    from vrcc.app import _swap_main_window

    bridge, detector, _pipeline, make_window = _rebuild_env(tmp_path)

    old = make_window()
    fresh = None
    try:
        old.set_capture_status(False, "a model failed to load")

        fresh = _swap_main_window(old, make_window, detector, None)

        assert fresh._capture_label.text() == (
            "Not listening - a model failed to load"
        )
    finally:
        for w in (old, fresh):
            if w is not None:
                w.close()
                w.deleteLater()
        bridge.detach()


class _FakeMuteSync:
    """Stand-in for MuteSync in _swap_main_window tests: only republish()
    matters here, MuteSync's own behavior is covered in test_mutesync.py."""

    def __init__(self) -> None:
        self.republish_calls = 0

    def republish(self) -> None:
        self.republish_calls += 1


def test_swap_main_window_republishes_mute_state(qapp, tmp_path):
    # Regression: a language change while muted rebuilt the window but never
    # told it the current mute state (MuteChanged only fires on a real
    # transition), so the rebuilt chip stayed hidden until the next toggle.
    from vrcc.app import _swap_main_window

    bridge, detector, _pipeline, make_window = _rebuild_env(tmp_path)
    mute = _FakeMuteSync()

    old = make_window()
    fresh = None
    try:
        fresh = _swap_main_window(old, make_window, detector, mute)
        assert mute.republish_calls == 1
    finally:
        for w in (old, fresh):
            if w is not None:
                w.close()
                w.deleteLater()
        bridge.detach()


def test_swap_main_window_guards_a_none_mute(qapp, tmp_path):
    # mute sync disabled (never built) must not crash the rebuild.
    from vrcc.app import _swap_main_window

    bridge, detector, _pipeline, make_window = _rebuild_env(tmp_path)
    old = make_window()
    fresh = None
    try:
        fresh = _swap_main_window(old, make_window, detector, None)  # must not raise
        assert fresh is not None
    finally:
        for w in (old, fresh):
            if w is not None:
                w.close()
                w.deleteLater()
        bridge.detach()
