"""The boot seam: what it builds, what it hands to run(), and what it never does twice."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication

import vrcc.boot as boot_mod
import vrcc.gui.style as style_mod


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture(autouse=True)
def _restore_app_chrome(qapp):
    """boot() themes the one QApplication instance for real, and every test
    module in this process shares it (only one can ever exist per process).
    Leaving its stylesheet/font/palette set here would carry the boot theme
    into whichever test module happens to run next and change what its
    widgets measure. ``_BASE_POINT_SIZE`` is the same story one level down:
    apply_font_scale caches it on first use, so a stale cache from here would
    outlive the font this fixture just restored."""
    font_before = qapp.font()
    style_before = qapp.styleSheet()
    palette_before = qapp.palette()
    style_mod._BASE_POINT_SIZE = None
    yield
    qapp.setFont(font_before)
    qapp.setStyleSheet(style_before)
    qapp.setPalette(palette_before)
    style_mod._BASE_POINT_SIZE = None


def _use_tmp_paths(monkeypatch, tmp_path):
    """boot() calls default_paths(portable) itself, so patching what it
    returns is the seam that keeps setup_logging and ConfigStore off the
    user's actual per-machine log/config directory just because a test calls
    the real boot(). Built through the real default_paths (portable mode,
    app_dir=tmp_path) rather than a hand-rolled Paths, so every field is
    exactly what production code would compute."""
    from vrcc.core.config import default_paths as real_default_paths

    paths = real_default_paths(True, app_dir=tmp_path)
    monkeypatch.setattr(boot_mod, "default_paths", lambda portable: paths)
    return paths


def test_boot_signature_matches_the_cli_contract():
    """cli.main() calls boot with these three keywords by name, and two tests in
    test_cli.py stub it with exactly this signature."""
    import inspect

    params = inspect.signature(boot_mod.boot).parameters
    assert list(params) == ["portable", "verbose", "guard"]


def test_boot_passes_the_guard_straight_through(qapp, monkeypatch, tmp_path):
    """PR 1's raise watch is installed inside run(), not here. If boot ever stops
    forwarding the guard, a second launch would silently stop raising the window
    and every existing test would still pass."""
    _use_tmp_paths(monkeypatch, tmp_path)
    seen = {}

    def _fake_run(**kwargs):
        seen.update(kwargs)
        return 0

    monkeypatch.setattr(boot_mod, "_run_app", _fake_run)
    monkeypatch.setattr(boot_mod, "_walk_imports", lambda progress: None)
    sentinel = object()
    assert boot_mod.boot(portable=True, verbose=False, guard=sentinel) == 0
    assert seen["guard"] is sentinel
    assert seen["portable"] is True


def test_boot_builds_logging_and_config_once(qapp, monkeypatch, tmp_path):
    """setup_logging is not idempotent: it adds a fresh file handler every call,
    so a second call writes a second log file with every record duplicated."""
    _use_tmp_paths(monkeypatch, tmp_path)
    calls = []
    monkeypatch.setattr(boot_mod, "setup_logging", lambda *a, **k: calls.append("log"))
    monkeypatch.setattr(boot_mod, "_walk_imports", lambda progress: None)
    monkeypatch.setattr(boot_mod, "_run_app", lambda **k: 0)
    boot_mod.boot()
    assert calls == ["log"]


def test_boot_hands_run_the_store_it_built(qapp, monkeypatch, tmp_path):
    """Must fail if boot ever built a second ConfigStore and forwarded that one
    instead: run()'s teardown closes over whatever store it was handed, so a
    mismatch here would silently discard anything the first store set."""
    _use_tmp_paths(monkeypatch, tmp_path)
    built_stores = []
    real_config_store = boot_mod.ConfigStore

    def _tracking_store(*a, **k):
        store = real_config_store(*a, **k)
        built_stores.append(store)
        return store

    monkeypatch.setattr(boot_mod, "ConfigStore", _tracking_store)
    seen = {}
    monkeypatch.setattr(boot_mod, "_walk_imports", lambda progress: None)
    monkeypatch.setattr(boot_mod, "_run_app", lambda **k: seen.update(k) or 0)
    boot_mod.boot()
    assert len(built_stores) == 1
    assert seen["store"] is built_stores[0]
    assert seen["paths"] is not None
    assert seen["progress"] is not None


def test_boot_returns_run_exit_code(qapp, monkeypatch, tmp_path):
    _use_tmp_paths(monkeypatch, tmp_path)
    monkeypatch.setattr(boot_mod, "_walk_imports", lambda progress: None)
    monkeypatch.setattr(boot_mod, "_run_app", lambda **k: 7)
    assert boot_mod.boot() == 7


def test_walk_reports_every_phase_in_order(qapp):
    """The walk must name each phase before importing it, so a launch that dies
    inside an import leaves the failing step as the log's last line."""
    from vrcc.core.progress import PHASES, LogProgress

    progress = LogProgress()
    boot_mod._walk_imports(progress)
    assert progress.steps == [key for key, _ in PHASES]


def test_walk_survives_a_failing_import(qapp, monkeypatch):
    """One broken subsystem must not stop the app from starting: run() reports
    engine failures through the UI, which cannot happen if boot died first.
    Pinned to the full phase list, not just a nonempty one, so a regression
    that widens the try to wrap the whole loop (aborting the remaining groups
    after the first failure) would be caught rather than pass by accident."""
    from vrcc.core.progress import PHASES, LogProgress

    def _boom(name):
        raise ImportError(name)

    monkeypatch.setattr(boot_mod, "_import_module", _boom)
    progress = LogProgress()
    boot_mod._walk_imports(progress)
    assert progress.steps == [key for key, _ in PHASES]
