"""Forced-rebuild tests for ``_Reloader``: a device/compute/thread change
keeps the model id, so a plain request would no-op -- ``force=True`` must run
one real swap, and that intent has to survive coalescing during an in-flight
swap.

Split out of ``test_app_reloader.py`` to keep both files under the source cap.
The small synchronous ``spawn``/``marshal`` swap fakes are duplicated here so
each file runs standalone.
"""

from __future__ import annotations


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


def test_reloader_force_rebuilds_when_model_id_unchanged():
    """A device/compute/thread change keeps the model id, so a plain request
    would no-op. force=True must run one real swap, gated during the rebuild."""
    builds = []

    def build(kind, target):
        builds.append((kind, target))
        return (_SwapEngine(), target)

    r, pipe, log = _make_reloader(build, loaded={"stt": "small", "mt": None})
    r.request("stt", "small", force=True)  # same id, but rebuild anyway
    assert builds == [("stt", "small")]  # exactly once, through the reloader
    assert log.swaps == [True, False]  # capture gated during, released after
    assert pipe.engines["stt"] is not None and pipe.engines["stt"].loaded
    assert log.status[-1] == (True, "")


def test_reloader_unforced_same_id_still_noops():
    """Guard the default: without force, re-requesting the loaded id no-ops."""
    builds = []

    def build(kind, target):
        builds.append((kind, target))
        return (_SwapEngine(), target)

    r, pipe, log = _make_reloader(build, loaded={"stt": "small", "mt": None})
    r.request("stt", "small")  # force defaults to False
    assert builds == []


def test_reloader_forced_rebuild_survives_coalesce_during_swap():
    """A forced rebuild requested while another kind is mid-swap must still run
    when replayed: the invalidated id keeps the no-force replay from no-opping."""
    builds = []
    fired = {"done": False}

    def build(kind, target):
        if not fired["done"]:
            fired["done"] = True
            r.request("stt", "small", force=True)  # arrives mid-swap -> pending
        builds.append((kind, target))
        return (_SwapEngine(), target)

    r, pipe, log = _make_reloader(build, loaded={"stt": "small", "mt": "mt-old"})
    targets = {"stt": "small", "mt": "mt-new"}

    def on_pending(kind):
        r.request(kind, targets[kind])  # replay carries no force flag

    r._on_pending = on_pending
    r.request("mt", "mt-new")
    assert ("stt", "small") in builds  # the forced stt rebuild really ran


def test_reloader_forced_rebuild_survives_a_same_kind_swap_in_flight():
    """Changing the STT device while an STT model swap runs: the swap records
    its own id on finishing, so the forced intent must outlive it or the engine
    keeps running on the old device with the config saying otherwise."""
    builds = []
    fired = {"done": False}

    def build(kind, target):
        if not fired["done"]:
            fired["done"] = True
            # The device change lands mid-swap and keeps the same model id.
            r.request("stt", "base", force=True)
        builds.append((kind, target))
        return (_SwapEngine(), target)

    r, pipe, log = _make_reloader(build, loaded={"stt": "small", "mt": "mt-old"})

    def on_pending(kind):
        r.request(kind, "base")  # the replay carries no force flag

    r._on_pending = on_pending
    r.request("stt", "base")

    assert builds == [("stt", "base"), ("stt", "base")]
    assert log.status[-1] == (True, "")
