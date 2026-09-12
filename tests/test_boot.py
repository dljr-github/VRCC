"""The boot seam: what it builds, what it hands to run(), and what it never does twice."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication

import vrcc.boot as boot_mod


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
    widgets measure."""
    font_before = qapp.font()
    style_before = qapp.styleSheet()
    palette_before = qapp.palette()
    yield
    qapp.setFont(font_before)
    qapp.setStyleSheet(style_before)
    qapp.setPalette(palette_before)


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


def test_boot_builds_logging_and_config_once(qapp, monkeypatch):
    """setup_logging is not idempotent: it adds a fresh file handler every call,
    so a second call writes a second log file with every record duplicated."""
    calls = []
    monkeypatch.setattr(boot_mod, "setup_logging", lambda *a, **k: calls.append("log"))
    monkeypatch.setattr(boot_mod, "_walk_imports", lambda progress: None)
    monkeypatch.setattr(boot_mod, "_run_app", lambda **k: 0)
    boot_mod.boot()
    assert calls == ["log"]


def test_boot_hands_run_the_store_it_built(qapp, monkeypatch):
    seen = {}
    monkeypatch.setattr(boot_mod, "_walk_imports", lambda progress: None)
    monkeypatch.setattr(boot_mod, "_run_app", lambda **k: seen.update(k) or 0)
    boot_mod.boot()
    assert seen["store"] is not None
    assert seen["paths"] is not None
    assert seen["progress"] is not None


def test_boot_returns_run_exit_code(qapp, monkeypatch):
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
    engine failures through the UI, which cannot happen if boot died first."""
    from vrcc.core.progress import LogProgress

    def _boom(name):
        raise ImportError(name)

    monkeypatch.setattr(boot_mod, "_import_module", _boom)
    progress = LogProgress()
    boot_mod._walk_imports(progress)
    assert len(progress.steps) > 0
