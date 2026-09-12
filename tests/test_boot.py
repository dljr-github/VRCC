"""The boot seam: what it builds, what it hands to run(), and what it never does twice."""

from __future__ import annotations

import logging
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
    """phase_labels() and the panel both read PHASES in this order, so a walk
    that visited them out of order would show progress that does not match
    what it is about to import."""
    from vrcc.core.progress import PHASES, LogProgress

    progress = LogProgress()
    boot_mod._walk_imports(progress)
    assert progress.steps == [key for key, _ in PHASES]


def test_walk_reaches_every_phase_and_logs_a_failing_group(qapp, monkeypatch, caplog):
    """A failing group must not stop the walk from naming the remaining
    phases -- it is the walk that needs to survive here, not the launch:
    vrcc.app re-imports these same modules at its own module scope, so a
    truly broken import still kills the process once _run_app hands off to
    run(). Pinned to the full phase list, not just a nonempty one, so a
    regression that widens the try to wrap the whole loop (aborting the
    remaining groups after the first failure) would be caught rather than
    pass by accident. The caplog check ties this to what the catch is
    actually for: the traceback lands in the log."""
    from vrcc.core.progress import PHASES, LogProgress

    def _boom(name):
        raise ImportError(name)

    monkeypatch.setattr(boot_mod, "_import_module", _boom)
    progress = LogProgress()
    with caplog.at_level(logging.WARNING, logger="vrcc.boot"):
        boot_mod._walk_imports(progress)
    assert progress.steps == [key for key, _ in PHASES]
    failures = [r for r in caplog.records if "failed to import" in r.getMessage()]
    assert len(failures) == len(PHASES)
    assert all(r.exc_info for r in failures)


def test_boot_hands_run_a_panel_backed_reporter(qapp, monkeypatch, tmp_path):
    """The normal path is real code with no coverage before this test: handing
    run() a bare LogProgress even when the panel built fine, or never calling
    panel.show(), would pass every other test in this file."""
    _use_tmp_paths(monkeypatch, tmp_path)
    monkeypatch.setattr(boot_mod, "_walk_imports", lambda progress: None)
    seen = {}
    monkeypatch.setattr(boot_mod, "_run_app", lambda **k: seen.update(k) or 0)
    boot_mod.boot()
    progress = seen["progress"]
    try:
        assert isinstance(progress, boot_mod._Both)
        assert progress._panel.isVisible()
    finally:
        progress.close()
        progress._panel.deleteLater()


def test_walk_pumps_events_through_the_panel_reporter(qapp, monkeypatch):
    """boot.py's processEvents() call is the entire reason _Both exists
    instead of just handing the panel to _walk_imports directly: without it
    the import walk runs synchronously with nothing pumping the event loop,
    and the panel never paints a single step. See the report for the probe
    that confirms deleting that call makes this test fail."""
    from vrcc.core.progress import PHASES, LogProgress
    from vrcc.gui.boot_panel import BootPanel

    monkeypatch.setattr(boot_mod, "_import_module", lambda name: None)
    calls = []
    monkeypatch.setattr(QApplication, "processEvents", lambda *a, **k: calls.append(1))

    panel = BootPanel()
    progress = boot_mod._Both(panel, LogProgress(), qapp)
    try:
        boot_mod._walk_imports(progress)
    finally:
        panel.close_panel()
        panel.deleteLater()
    assert len(calls) >= len(PHASES)


def test_boot_falls_back_to_log_progress_when_the_panel_fails_to_build(qapp, monkeypatch, tmp_path):
    """BootPanel is imported inside boot()'s own try block precisely so a
    raising constructor still leaves boot() with a reporter run() can use --
    the walk must still complete and still be logged, not just not-crash."""
    from vrcc.core.progress import PHASES

    _use_tmp_paths(monkeypatch, tmp_path)
    monkeypatch.setattr(boot_mod, "_import_module", lambda name: None)

    class _Boom:
        def __init__(self, *a, **k):
            raise RuntimeError("no panel for you")

    monkeypatch.setattr("vrcc.gui.boot_panel.BootPanel", _Boom)
    seen = {}
    monkeypatch.setattr(boot_mod, "_run_app", lambda **k: seen.update(k) or 0)
    boot_mod.boot()
    progress = seen["progress"]
    assert isinstance(progress, boot_mod.LogProgress)
    assert progress.steps == [key for key, _ in PHASES]


def test_both_close_is_idempotent(qapp, caplog):
    """run() calls progress.close() once if the first-run wizard opens and
    again once the main window is ready (app.py:183 and :442); a log someone
    is troubleshooting must not show "boot steps complete" twice for one
    launch."""
    from vrcc.core.progress import LogProgress
    from vrcc.gui.boot_panel import BootPanel

    panel = BootPanel()
    progress = boot_mod._Both(panel, LogProgress(), qapp)
    try:
        with caplog.at_level(logging.INFO, logger="vrcc.core.progress"):
            progress.close()
            progress.close()
    finally:
        panel.deleteLater()
    completions = [r for r in caplog.records if "boot steps complete" in r.getMessage()]
    assert len(completions) == 1
