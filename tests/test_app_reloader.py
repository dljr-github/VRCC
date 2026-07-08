"""Tests for ``_Reloader`` (live model hot-swap on a background thread,
marshaled onto the GUI thread) and ``_status_after_swap``.

Tests inject BOTH a synchronous ``spawn`` (run the swap body inline instead
of on a real thread) and a synchronous ``marshal`` (``lambda fn: fn()``) so
the whole swap completes within ``request()`` and assertions are
deterministic without any joins.
"""

from __future__ import annotations


def test_reloader_swaps_stt_unload_first():
    from vrcc.core.reloading import _Reloader

    class FakeEngine:
        def __init__(self, tag):
            self.tag = tag
            self.loaded = False
            self.unloaded = False

        def load(self):
            self.loaded = True

        def warm_up(self):
            pass

        def unload(self):
            self.unloaded = True

    old = FakeEngine("old")
    events = []

    class FakePipe:
        def __init__(self):
            self.stt = old

        def detach_stt(self):
            events.append("detach")
            e, self.stt = self.stt, None
            return e

        def set_stt(self, e):
            events.append("install")
            self.stt = e

        def detach_mt(self):
            return None

        def set_mt(self, e):
            pass

    pipe = FakePipe()
    new = FakeEngine("new")
    swaps = []
    status = []
    r = _Reloader(
        pipeline=pipe,
        build=lambda kind, target: (new, "new-id"),
        load=lambda e: e.load() if e else None,
        set_swapping=lambda v: swaps.append(v),
        set_status=lambda ok, reason="": status.append((ok, reason)),
        marshal=lambda fn: fn(),  # synchronous
        spawn=lambda fn: fn(),  # run the swap body inline
        bus=type("B", (), {"publish": lambda self, e: None})(),
        loaded={"stt": "old-id", "mt": None},
    )
    r._on_pending = lambda kind: None
    r.request("stt", "new-id")
    assert old.unloaded and new.loaded  # unload-first, then load
    assert events == ["detach", "install"]
    assert pipe.stt is new
    assert swaps == [True, False]  # paused then resumed
    assert status[0] == (False, "Switching voice model…")
    assert status[-1] == (True, "")


def test_reloader_noop_when_target_already_loaded():
    from vrcc.core.reloading import _Reloader

    called = []
    r = _Reloader(
        pipeline=None,
        build=lambda k, t: called.append(k),
        load=lambda e: None,
        set_swapping=lambda v: None,
        set_status=lambda *a, **k: None,
        marshal=lambda fn: fn(),
        spawn=lambda fn: fn(),
        bus=None,
        loaded={"stt": "same", "mt": None},
    )
    r.request("stt", "same")
    assert called == []  # build never invoked


def test_reloader_failed_load_pauses_and_frees_old():
    from vrcc.core.reloading import _Reloader

    class Boom:
        def load(self):
            raise RuntimeError("no fit")

        def warm_up(self):
            pass

        def unload(self):
            pass

    class FakePipe:
        def detach_stt(self):
            return Boom()

        def set_stt(self, e):
            self.installed = e

        def detach_mt(self):
            return None

        def set_mt(self, e):
            pass

    published = []
    status = []
    r = _Reloader(
        pipeline=FakePipe(),
        build=lambda kind, target: (Boom(), "x"),
        load=lambda e: e.load(),
        set_swapping=lambda v: None,
        set_status=lambda ok, reason="": status.append((ok, reason)),
        marshal=lambda fn: fn(),
        spawn=lambda fn: fn(),
        bus=type("B", (), {"publish": lambda self, e: published.append(e)})(),
        loaded={"stt": "old", "mt": None},
    )
    r._on_pending = lambda kind: None
    r.request("stt", "x")
    assert published and published[0].code == "MODEL_SWITCH_FAILED"
    assert status[-1][0] is False  # capture left paused/not-capturing


def test_reloader_coalesces_request_during_swap():
    """A request that arrives while a swap is running is deferred, then
    re-processed via ``_on_pending`` after the running swap finishes."""
    from vrcc.core.reloading import _Reloader

    class FakeEngine:
        def load(self):
            pass

        def warm_up(self):
            pass

        def unload(self):
            pass

    class FakePipe:
        def __init__(self):
            self.stt = FakeEngine()

        def detach_stt(self):
            e, self.stt = self.stt, None
            return e

        def set_stt(self, e):
            self.stt = e

        def detach_mt(self):
            return None

        def set_mt(self, e):
            pass

    reprocessed = []
    r = _Reloader(
        pipeline=FakePipe(),
        build=lambda kind, target: (FakeEngine(), f"{kind}-id"),
        load=lambda e: e.load() if e else None,
        set_swapping=lambda v: None,
        set_status=lambda ok, reason="": None,
        marshal=lambda fn: fn(),
        spawn=lambda fn: fn(),
        bus=type("B", (), {"publish": lambda self, e: None})(),
        loaded={"stt": "old-id", "mt": None},
    )
    r._on_pending = lambda kind: reprocessed.append(kind)

    # Make the STT build fire a second request mid-swap (as if the user changed
    # the MT model while the voice-model swap was still running). Because the
    # reloader is busy, that request must be coalesced into _pending, not run.
    fired = {"once": False}

    def build_then_request(kind, target):
        if not fired["once"]:
            fired["once"] = True
            r.request("mt", "mt-id")  # arrives mid-swap -> deferred
        return (FakeEngine(), f"{kind}-id")

    r._build = build_then_request
    r.request("stt", "new-id")
    assert reprocessed == ["mt"]  # deferred kind re-processed exactly once


# -- _Reloader failure-path fixes (C1 / I2 / I3 / I5) ------------------------


class _SwapEngine:
    """Loadable/unloadable fake; ``fail=True`` raises from ``load()``."""

    def __init__(self, fail: bool = False):
        self.fail = fail
        self.loaded = False
        self.unloaded = False

    def load(self):
        if self.fail:
            raise RuntimeError("no fit")
        self.loaded = True

    def warm_up(self):
        pass

    def unload(self):
        self.unloaded = True


class _SwapPipe:
    """Records every install per kind; detach hands out the live engine."""

    def __init__(self):
        self.installed = {"stt": [], "mt": []}
        self.engines = {"stt": _SwapEngine(), "mt": _SwapEngine()}

    def detach_stt(self):
        e, self.engines["stt"] = self.engines["stt"], None
        return e

    def set_stt(self, e):
        self.installed["stt"].append(e)
        self.engines["stt"] = e

    def detach_mt(self):
        e, self.engines["mt"] = self.engines["mt"], None
        return e

    def set_mt(self, e):
        self.installed["mt"].append(e)
        self.engines["mt"] = e


def _make_reloader(build, loaded):
    """Reloader with synchronous spawn/marshal and recording fakes.
    Returns ``(reloader, pipe, log)``; ``log`` records swaps/status/errors."""
    from types import SimpleNamespace

    from vrcc.core.reloading import _Reloader

    pipe = _SwapPipe()
    log = SimpleNamespace(swaps=[], status=[], errors=[])
    r = _Reloader(
        pipeline=pipe,
        build=build,
        load=lambda e: e.load() if e is not None else None,
        set_swapping=log.swaps.append,
        set_status=lambda ok, reason="": log.status.append((ok, reason)),
        marshal=lambda fn: fn(),
        spawn=lambda fn: fn(),
        bus=type("B", (), {"publish": lambda self, e: log.errors.append(e)})(),
        loaded=loaded,
    )
    r._on_pending = lambda kind: None
    return r, pipe, log


def test_reloader_failed_swap_allows_reselecting_previous_model():
    """C1: after a failed swap the old engine is already freed; re-selecting
    the old id must run a fresh swap, not no-op on the stale loaded entry."""
    builds = []

    def build(kind, target):
        builds.append(target)
        return (_SwapEngine(fail=(target == "large-v3")), target)

    r, pipe, log = _make_reloader(build, loaded={"stt": "small", "mt": None})
    r.request("stt", "large-v3")  # load fails; "small" is long gone
    assert log.errors and log.errors[0].code == "MODEL_SWITCH_FAILED"
    r.request("stt", "small")  # back to the old id -> must actually swap
    assert builds == ["large-v3", "small"]
    assert pipe.engines["stt"] is not None and pipe.engines["stt"].loaded
    assert log.swaps[-1] is False  # gate released after the recovery swap
    assert log.status[-1] == (True, "")


def test_reloader_failed_swap_installs_none_not_broken_engine():
    """I2a: an engine whose load() raised must not be installed -- the
    pipeline's None paths drop/pass-through jobs instead of raising."""
    r, pipe, log = _make_reloader(
        lambda kind, target: (_SwapEngine(fail=True), target),
        loaded={"stt": "old", "mt": None},
    )
    r.request("stt", "new")
    assert pipe.installed["stt"] == [None]


def test_reloader_other_kind_success_keeps_gate_while_one_failed():
    """I2b: a successful MT swap must not clear the swap gate or paint the
    status green while the STT engine is dead from its own failed swap."""
    r, pipe, log = _make_reloader(
        lambda kind, target: (_SwapEngine(fail=(kind == "stt")), target),
        loaded={"stt": "old", "mt": "mt-old"},
    )
    r.request("stt", "new")  # fails -> stt dead
    r.request("mt", "mt-new")  # succeeds
    assert log.swaps[-1] is True  # gate still on
    assert log.status[-1][0] is False  # not painted green


def test_reloader_recovery_swap_releases_gate():
    """I2c: after a failed STT swap, a later successful STT swap clears the
    failure and restores the gate + green status."""
    r, pipe, log = _make_reloader(
        lambda kind, target: (_SwapEngine(fail=(target == "bad")), target),
        loaded={"stt": "old", "mt": None},
    )
    r.request("stt", "bad")
    r.request("stt", "good")
    assert log.swaps[-1] is False
    assert log.status[-1] == (True, "")
    assert pipe.engines["stt"].loaded


class _PopsNoopFirst(set):
    """Deterministic pop order: plain-set pop is hash-randomized, which would
    let a one-kind drain sometimes pass by luck (popping the kind whose
    reprocess starts a swap chains into draining the other)."""

    def pop(self):
        item = max(self)  # "stt" > "mt": pop the no-op kind first
        self.remove(item)
        return item


def test_reloader_drains_all_pending_kinds():
    """I3: two kinds coalesced during one swap must BOTH be re-evaluated,
    even when the first re-evaluation is a no-op (change-then-change-back)."""
    builds = []
    fired = {"done": False}

    def build(kind, target):
        if not fired["done"]:
            fired["done"] = True
            r.request("stt", "other")  # mid-swap -> pending
            r.request("mt", "mt-new")  # mid-swap -> pending
        builds.append((kind, target))
        return (_SwapEngine(), target)

    r, pipe, log = _make_reloader(build, loaded={"stt": "old", "mt": "mt-old"})
    r._pending = _PopsNoopFirst()
    reprocessed = []
    targets = {"stt": "new-id", "mt": "mt-new"}  # stt reprocess -> no-op

    def on_pending(kind):
        reprocessed.append(kind)
        r.request(kind, targets[kind])

    r._on_pending = on_pending
    r.request("stt", "new-id")
    assert sorted(reprocessed) == ["mt", "stt"]  # nothing stranded
    assert ("mt", "mt-new") in builds  # the sibling kind's swap really ran


def test_reloader_build_receives_requested_target():
    """I5: the swap builds exactly the requested target, not whatever the
    config says at build time (a coalesced replay must not resurrect a
    deleted model)."""
    seen = []

    def build(kind, target):
        seen.append((kind, target))
        return (None, None) if target is None else (_SwapEngine(), target)

    r, pipe, log = _make_reloader(build, loaded={"stt": "old", "mt": "mt-old"})
    r.request("mt", None)  # e.g. model deleted -> target recomputed to None
    assert seen == [("mt", None)]
    assert pipe.installed["mt"] == [None]
    assert log.status[-1] == (True, "")  # a (None, None) swap is a success


# -- _status_after_swap (I4: no false "Capturing" after a swap) --------------


def test_status_after_swap_does_not_paint_green_when_pipeline_down():
    """I4: swap succeeded but the pipeline never started; the start attempt
    fails -> status stays red, never a false 'Capturing'."""
    from vrcc.core.reloading import _status_after_swap

    calls = []
    started = [False]
    _status_after_swap(
        True,
        "",
        started=started,
        start=lambda: False,
        set_status=lambda ok, reason="": calls.append((ok, reason)),
    )
    assert calls == [(False, "microphone unavailable")]
    assert started == [False]


def test_status_after_swap_recovers_pipeline_when_start_succeeds():
    """I4: swap succeeded, pipeline down, start attempt succeeds -> genuine
    recovery: flag flips and the status is green."""
    from vrcc.core.reloading import _status_after_swap

    calls = []
    started = [False]
    _status_after_swap(
        True,
        "",
        started=started,
        start=lambda: True,
        set_status=lambda ok, reason="": calls.append((ok, reason)),
    )
    assert calls == [(True, "")]
    assert started == [True]


def test_status_after_swap_passes_through_otherwise():
    """Already-started success and any failure pass through; no start attempt
    (start would raise if called)."""
    from vrcc.core.reloading import _status_after_swap

    calls = []
    boom = lambda: 1 / 0  # noqa: E731
    _status_after_swap(
        True, "", started=[True], start=boom,
        set_status=lambda ok, reason="": calls.append((ok, reason)),
    )
    _status_after_swap(
        False, "Switching voice model…", started=[False], start=boom,
        set_status=lambda ok, reason="": calls.append((ok, reason)),
    )
    assert calls == [(True, ""), (False, "Switching voice model…")]
