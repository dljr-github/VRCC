"""Tests for the CLI entry point's process-level guards."""

from __future__ import annotations

import sys

from vrcc.cli import _ensure_std_streams


def test_ensure_std_streams_replaces_none():
    """Regression: a windowed exe / pythonw run has sys.stdout/stderr == None,
    which crashed anything that wrote to them. The guard must install real
    writable streams so those writes are harmless."""
    saved_out, saved_err = sys.stdout, sys.stderr
    sys.stdout = None
    sys.stderr = None
    try:
        _ensure_std_streams()
        assert sys.stdout is not None
        assert sys.stderr is not None
        sys.stdout.write("out")  # must not raise
        sys.stderr.write("err")  # must not raise
        sys.stdout.flush()
        sys.stderr.flush()
    finally:
        for stream in (sys.stdout, sys.stderr):
            if stream not in (saved_out, saved_err):
                try:
                    stream.close()
                except Exception:
                    pass
        sys.stdout, sys.stderr = saved_out, saved_err


def test_ensure_std_streams_leaves_real_streams_untouched():
    """When the streams already exist, the guard must not replace them."""
    before_out, before_err = sys.stdout, sys.stderr
    _ensure_std_streams()
    assert sys.stdout is before_out
    assert sys.stderr is before_err
