"""Tests for the single-instance guard.

The happy paths use real kernel objects, not mocks: CreateMutexW twice in one
process with the same name is exactly the two-copy case, and CI is
windows-latest. Only the two branches that cannot be provoked for real -- a
failed CreateMutexW and the already-exists race, where the fake handle
_close_handle receives is never a real one -- patch _create_mutex and
_close_handle. Every test uses a uuid-suffixed name so a crashed run cannot
poison the next one.
"""

from __future__ import annotations

import sys
import uuid

import pytest

from vrcc.core.instance import InstanceGuard, allow_multiple

pytestmark = pytest.mark.skipif(
    sys.platform != "win32", reason="the guard is a Windows kernel object"
)


def _name() -> str:
    return f"VRCC.Test.{uuid.uuid4().hex}"


def test_first_acquire_wins():
    guard = InstanceGuard(_name())
    try:
        assert guard.acquire() is True
    finally:
        guard.release()


def test_second_acquire_is_refused():
    name = _name()
    first, second = InstanceGuard(name), InstanceGuard(name)
    try:
        assert first.acquire() is True
        assert second.acquire() is False
    finally:
        second.release()
        first.release()


def test_release_frees_the_name_for_a_later_copy():
    """The kernel destroys the object with its last handle, which is why there
    is no stale-lock recovery path anywhere in this module."""
    name = _name()
    first = InstanceGuard(name)
    assert first.acquire() is True
    first.release()
    second = InstanceGuard(name)
    try:
        assert second.acquire() is True
    finally:
        second.release()


def test_doorbell_is_not_rung_by_default():
    guard = InstanceGuard(_name())
    try:
        assert guard.acquire() is True
        assert guard.doorbell_rang() is False
    finally:
        guard.release()


def test_ring_is_seen_once_then_consumed():
    """Auto-reset: one ring is one raise, so a burst of clicks cannot queue a
    stack of raises the user then has to dismiss."""
    name = _name()
    winner, loser = InstanceGuard(name), InstanceGuard(name)
    try:
        assert winner.acquire() is True
        assert loser.acquire() is False
        assert loser.ring() is True
        assert winner.doorbell_rang() is True
        assert winner.doorbell_rang() is False
    finally:
        loser.release()
        winner.release()


def test_ring_before_acquire_is_still_seen():
    """The doorbell is created before the mutex is claimed, so a process that
    loses a dead heat always has something to ring, even one that never gets
    as far as calling acquire() on its own guard."""
    import vrcc.core.instance as instance

    name = _name()
    winner, loser = InstanceGuard(name), InstanceGuard(name)
    try:
        assert winner.acquire() is True
        loser._event, _ = instance._create_event(loser.event_name)
        assert loser.ring() is True
        assert winner.doorbell_rang() is True
    finally:
        loser.release()
        winner.release()


def test_release_is_idempotent_and_safe_before_acquire():
    guard = InstanceGuard(_name())
    guard.release()
    guard.release()


def test_doorbell_rang_is_false_when_never_acquired():
    guard = InstanceGuard(_name())
    assert guard.doorbell_rang() is False


def test_allow_multiple_reads_the_env_var(monkeypatch):
    monkeypatch.delenv("VRCC_ALLOW_MULTIPLE", raising=False)
    assert allow_multiple() is False
    monkeypatch.setenv("VRCC_ALLOW_MULTIPLE", "1")
    assert allow_multiple() is True
    monkeypatch.setenv("VRCC_ALLOW_MULTIPLE", "0")
    assert allow_multiple() is False
    monkeypatch.setenv("VRCC_ALLOW_MULTIPLE", "")
    assert allow_multiple() is False


def test_allow_multiple_lets_a_second_guard_acquire(monkeypatch):
    monkeypatch.setenv("VRCC_ALLOW_MULTIPLE", "1")
    name = _name()
    first, second = InstanceGuard(name), InstanceGuard(name)
    try:
        assert first.acquire() is True
        assert second.acquire() is True
    finally:
        second.release()
        first.release()


def test_acquire_fails_open_when_the_kernel_call_fails(monkeypatch):
    """A guard that cannot be read must never stop the app from starting.
    Only ERROR_ALREADY_EXISTS refuses; everything else runs."""
    import vrcc.core.instance as instance

    guard = InstanceGuard(_name())
    monkeypatch.setattr(instance, "_create_mutex", lambda name: (None, 5))
    try:
        assert guard.acquire() is True
    finally:
        guard.release()


def test_acquire_refuses_only_on_already_exists(monkeypatch):
    import vrcc.core.instance as instance

    guard = InstanceGuard(_name())
    closed = []
    monkeypatch.setattr(
        instance, "_create_mutex", lambda name: (1234, instance.ERROR_ALREADY_EXISTS)
    )
    monkeypatch.setattr(instance, "_close_handle", lambda handle: closed.append(handle))
    assert guard.acquire() is False
    assert 1234 in closed
    # Undo before release so the real doorbell handle this guard opened gets
    # a real CloseHandle instead of another append to the fake.
    monkeypatch.undo()
    guard.release()


def test_names_are_session_local():
    """Local\\ keeps the guard inside the logon session, so two users on one
    machine each get their own copy rather than one blocking the other."""
    guard = InstanceGuard("VRCC.Test.Naming")
    assert guard.mutex_name.startswith("Local\\")
    assert guard.event_name.startswith("Local\\")
    assert guard.mutex_name != guard.event_name
