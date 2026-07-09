"""Tests for :mod:`vrcc.core.logs`: one DEBUG file per run, only the newest
few kept, console only when verbose.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import pytest

from vrcc.core.logs import LOG_KEEP, prune_logs, setup_logging


@pytest.fixture()
def clean_root():
    """Restore the root logger: _setup_logging adds handlers to it."""
    root = logging.getLogger()
    handlers = list(root.handlers)
    level = root.level
    root.handlers.clear()
    yield root
    for handler in root.handlers:
        handler.close()
    root.handlers[:] = handlers
    root.setLevel(level)


def _log_names(logs_dir: Path) -> set[str]:
    return {p.name for p in logs_dir.glob("vrcc*.log*")}


def _console_handlers(root: logging.Logger) -> list[logging.Handler]:
    # Exact type: pytest's own capture handler subclasses StreamHandler and is
    # (re)attached to the root logger around every test phase.
    return [h for h in root.handlers if type(h) is logging.StreamHandler]


def _aged(path: Path, order: int) -> None:
    """Give ``path`` a distinct mtime so pruning has a stable ordering."""
    stamp = 1_600_000_000 + order * 60
    os.utime(path, (stamp, stamp))


def test_setup_logging_writes_a_debug_file_for_this_run(tmp_path, clean_root):
    logs_dir = tmp_path / "logs"

    setup_logging(logs_dir, verbose=False)
    logging.getLogger("vrcc.test").debug("a debug line")
    for handler in clean_root.handlers:
        handler.flush()

    names = _log_names(logs_dir)
    assert len(names) == 1
    name = names.pop()
    assert name.startswith("vrcc-") and name.endswith(".log")
    # DEBUG reaches the file even without --verbose: that is what makes an
    # attached log useful in a bug report.
    assert "a debug line" in (logs_dir / name).read_text(encoding="utf-8")


def test_setup_logging_adds_console_only_when_verbose(tmp_path, clean_root):
    setup_logging(tmp_path / "quiet", verbose=False)
    assert _console_handlers(clean_root) == []

    for handler in list(clean_root.handlers):
        handler.close()
    clean_root.handlers.clear()

    setup_logging(tmp_path / "loud", verbose=True)
    assert len(_console_handlers(clean_root)) == 1


def test_setup_logging_keeps_only_the_newest_runs(tmp_path, clean_root):
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    old = []
    for i in range(8):
        path = logs_dir / f"vrcc-2020010{i}-000000.log"
        path.write_text(f"old {i}", encoding="utf-8")
        _aged(path, i)
        old.append(path.name)
    # A leftover file from the previous rotating-handler layout ages out too.
    legacy = logs_dir / "vrcc.log"
    legacy.write_text("legacy", encoding="utf-8")
    _aged(legacy, -1)

    setup_logging(logs_dir, verbose=False)

    names = _log_names(logs_dir)
    assert len(names) == LOG_KEEP
    fresh = names - set(old)
    assert len(fresh) == 1 and fresh.pop().startswith("vrcc-")
    # The four newest previous runs survive; older ones and the legacy file go.
    assert names & set(old) == set(old[-(LOG_KEEP - 1):])
    assert "vrcc.log" not in names


def test_prune_logs_never_raises_on_a_missing_directory(tmp_path):
    prune_logs(tmp_path / "nope")
