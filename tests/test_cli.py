"""Tests for the CLI entry point's process-level guards."""

from __future__ import annotations

import sys

import pytest

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


def test_second_launch_returns_zero_without_importing_the_app(monkeypatch):
    """The whole point of the guard's position: a refused launch must die
    before paying for the expensive vrcc.app import.

    Proven by absence, not by poisoning sys.modules. A poisoned entry would sit
    unused in the refused branch and the test would pass even if the ordering
    were wrong. An earlier test module in the same session may already have
    imported vrcc.app, so drop it first and assert it did not come back.
    """
    import inspect

    import vrcc.cli as cli

    monkeypatch.delitem(sys.modules, "vrcc.app", raising=False)
    monkeypatch.delitem(sys.modules, "vrcc.boot", raising=False)
    monkeypatch.setattr(sys, "argv", ["vrcc"])

    rung = []

    class _Refused:
        def acquire(self):
            return False

        def ring(self):
            rung.append(True)
            return True

        def release(self):
            pass

    monkeypatch.setattr(cli, "InstanceGuard", lambda *a, **k: _Refused())
    assert cli.main() == 0
    assert rung == [True]
    assert "vrcc.app" not in sys.modules
    assert "vrcc.boot" not in sys.modules
    # sys.modules alone cannot see a hoist to module scope: this test module
    # already imported vrcc.cli at collection time, so cli.py would not
    # re-execute and vrcc.boot would never reappear even if the import moved
    # above main(). Pin the import to main()'s own source as well.
    assert "from vrcc.boot import boot" in inspect.getsource(cli.main)


def test_first_launch_runs_the_app(monkeypatch):
    import types

    import vrcc.cli as cli

    fake = types.ModuleType("vrcc.boot")
    calls = []

    def _run(portable=False, verbose=False, guard=None):
        calls.append((portable, verbose, guard))
        return 7

    fake.boot = _run
    monkeypatch.setitem(sys.modules, "vrcc.boot", fake)
    monkeypatch.setattr(sys, "argv", ["vrcc", "--portable"])

    class _Allowed:
        def acquire(self):
            return True

        def release(self):
            pass

    allowed = _Allowed()
    monkeypatch.setattr(cli, "InstanceGuard", lambda *a, **k: allowed)
    assert cli.main() == 7
    assert calls == [(True, False, allowed)]


def test_guard_is_created_after_argparse(monkeypatch):
    """A bad argument must exit on argparse's terms, not leave a kernel object
    behind."""
    import vrcc.cli as cli

    created = []
    monkeypatch.setattr(cli, "InstanceGuard", lambda *a, **k: created.append(1))
    monkeypatch.setattr(sys, "argv", ["vrcc", "--nonsense"])
    with pytest.raises(SystemExit):
        cli.main()
    assert created == []


def test_release_runs_even_when_the_app_raises(monkeypatch):
    """release() sits in a finally: a crash inside run() must not leave the
    mutex handle open for the rest of the process."""
    import types

    import vrcc.cli as cli

    fake = types.ModuleType("vrcc.boot")

    def _run(portable=False, verbose=False, guard=None):
        raise RuntimeError("boom")

    fake.boot = _run
    monkeypatch.setitem(sys.modules, "vrcc.boot", fake)
    monkeypatch.setattr(sys, "argv", ["vrcc"])

    released = []

    class _Allowed:
        def acquire(self):
            return True

        def release(self):
            released.append(True)

    monkeypatch.setattr(cli, "InstanceGuard", lambda *a, **k: _Allowed())
    with pytest.raises(RuntimeError):
        cli.main()
    assert released == [True]


def test_run_signature_accepts_guard():
    """Every other test in this module installs a fake vrcc.app, so none of
    them would notice guard= being dropped from the real run(). This is the
    one check that looks at the actual function."""
    import inspect

    import vrcc.app

    assert "guard" in inspect.signature(vrcc.app.run).parameters


def test_boot_signature_accepts_the_cli_keywords():
    """Test 7 pins the real run(); this pins the real boot(). Every other test
    in this module installs a fake, so without these two a signature change
    would break every launch while the suite stayed green."""
    import inspect

    import vrcc.boot

    params = inspect.signature(vrcc.boot.boot).parameters
    assert {"portable", "verbose", "guard"} <= set(params)
