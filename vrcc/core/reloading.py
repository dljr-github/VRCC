"""Engine loading and live model hot-swap (Qt-free, unit-testable).

Two independent jobs share this module: :class:`EngineLoader` runs the
startup load()+warm_up() of STT/MT on a background thread; :class:`_Reloader`
hot-swaps a running engine for a different model while the app is up. Both
report through the injected ``bus``/callables so :mod:`vrcc.app` (the only
Qt-aware caller) can marshal results onto the GUI thread.
"""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING, Callable

from vrcc.core.bus import EventBus
from vrcc.core.events import AppError
from vrcc.i18n import tr, tr_noop

if TYPE_CHECKING:
    from vrcc.stt.engine import SttEngine
    from vrcc.translate.engine import TranslateEngine

logger = logging.getLogger("vrcc.app")


class EngineLoader:
    """Loads + warms the STT (then MT) engine on one background thread.

    :meth:`start` runs load()+warm_up() for stt then mt (if present) on a daemon
    thread, then calls ``on_complete(success)`` (on the loader thread; a failure
    also publishes an :class:`AppError`). Engines publish their own state events.
    Each engine loads in its own try block, so one failure never skips the
    other; ``failed_kinds`` records which kind(s) failed for the caller.
    """

    def __init__(
        self,
        stt: SttEngine,
        mt: TranslateEngine | None,
        bus: EventBus,
        on_complete: Callable[[bool], None],
    ) -> None:
        self._stt = stt
        self._mt = mt
        self._bus = bus
        self._on_complete = on_complete
        self._thread: threading.Thread | None = None
        # Which engine kinds failed to load ("stt"/"mt"). Read by run() after
        # on_complete(False) to seed the reloader's per-kind failure state.
        self.failed_kinds: set[str] = set()

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._run, name="EngineLoader", daemon=True
        )
        self._thread.start()

    def join(self, timeout: float | None = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout)

    def is_alive(self) -> bool:
        """Whether the loader thread is still running (e.g. inside a native
        model load). Checked at shutdown: finalization must not race a live
        native load."""
        return self._thread is not None and self._thread.is_alive()

    def _run(self) -> None:
        for kind, engine in (("stt", self._stt), ("mt", self._mt)):
            if engine is None:
                continue
            try:
                engine.load()
                engine.warm_up()
            except Exception as exc:  # noqa: BLE001 -- report, never crash the thread
                self.failed_kinds.add(kind)
                logger.exception("%s engine load failed", kind)
                self._bus.publish(AppError("ENGINE_LOAD_FAILED", str(exc)))
        try:
            self._on_complete(not self.failed_kinds)
        except Exception:  # noqa: BLE001 -- a bad callback must not crash the thread
            logger.exception("engine-load completion callback raised")


# Status shown while a live model swap is in flight (capture paused).
_SWITCH_STATUS = {
    "stt": tr_noop("Switching voice model…"),
    "mt": tr_noop("Switching translation model…"),
}

# _Reloader._loaded value after a failed swap: equals no target id (not even a
# legitimate None MT target), so ANY re-selection triggers a fresh swap.
_FAILED = object()


def _spawn_daemon(fn: Callable[[], None]) -> None:
    """Default :class:`_Reloader` swap runner: run ``fn`` on a fresh daemon
    thread. Tests inject a synchronous replacement (``lambda fn: fn()``)."""
    threading.Thread(target=fn, name="ModelReloader", daemon=True).start()


class _Reloader:
    """Unload-first live model hot-swap orchestrator (Qt-free, unit-testable).

    :meth:`request(kind, target_id)` (GUI thread) no-ops if already installed,
    else pauses capture and runs ONE swap at a time on a daemon thread (detach
    -> unload -> build -> load -> marshal install back); mid-swap requests
    coalesce and re-evaluate against current config when it ends. On failure the
    kind is marked ``_FAILED`` and capture stays gated. All Qt/model contact is
    via injected callables so tests can drive it with fakes.
    """

    def __init__(
        self,
        *,
        pipeline,
        build,
        load,
        set_swapping,
        set_status,
        marshal,
        bus,
        loaded,
        spawn=None,
    ):
        self._pipeline = pipeline
        self._build = build            # build(kind, target_id) -> (engine|None, id|None)
        self._load = load              # load(engine) -> None (load+warm_up)
        self._set_swapping = set_swapping
        self._set_status = set_status
        self._marshal = marshal        # marshal(fn): run fn on the GUI thread
        self._bus = bus
        self._loaded = dict(loaded)    # {"stt": id|None, "mt": id|None}
        self._spawn = spawn or _spawn_daemon
        # Set by run() to the on_model_change closure so coalesced kinds are
        # re-evaluated against the current config.
        self._on_pending: Callable[[str], None] | None = None
        self._lock = threading.Lock()
        self._busy = False
        self._pending: set[str] = set()
        self._failed: set[str] = set()  # kinds whose last swap failed

    def request(self, kind: str, target_id) -> None:
        """(GUI thread) Ensure ``kind`` ends up installed as ``target_id``."""
        with self._lock:
            # No-op if the desired model is already installed and idle (covers
            # same-value re-selection and change-then-change-back).
            if self._loaded.get(kind) == target_id and not self._busy:
                return
            if self._busy:
                # A swap is already running; remember this kind and re-check it
                # against the current config when that swap finishes.
                self._pending.add(kind)
                return
            self._busy = True
        self._start(kind, target_id)

    def _start(self, kind: str, target_id) -> None:
        self._set_swapping(True)
        self._set_status(False, tr(_SWITCH_STATUS[kind]))
        self._spawn(lambda: self._run(kind, target_id))

    def _run(self, kind: str, target_id) -> None:
        ok, new, new_id = True, None, None
        try:
            detach = (
                self._pipeline.detach_stt
                if kind == "stt"
                else self._pipeline.detach_mt
            )
            old = detach()
            if old is not None:
                old.unload()
            # Build exactly the requested target -- never a (possibly drifted)
            # config value re-read at build time.
            new, new_id = self._build(kind, target_id)
            self._load(new)                   # load()+warm_up(); no-op if None
        except Exception as exc:  # noqa: BLE001 -- report, never crash the swap thread
            ok = False
            logger.exception("model hot-swap failed")
            self._bus.publish(AppError("MODEL_SWITCH_FAILED", str(exc)))
        self._marshal(lambda: self._finish(kind, new, new_id, ok))

    def _finish(self, kind: str, new, new_id, ok: bool) -> None:
        install = (
            self._pipeline.set_stt if kind == "stt" else self._pipeline.set_mt
        )
        # Never install an engine whose load failed: the pipeline's None paths
        # drop/pass-through jobs cleanly instead of raising per utterance.
        install(new if ok else None)
        if ok:
            self._loaded[kind] = new_id
            self._failed.discard(kind)
        else:
            # Old engine already freed. _FAILED equals no target id, so
            # re-selecting ANY model (even the one that just failed) triggers a
            # fresh swap, not a no-op.
            self._loaded[kind] = _FAILED
            self._failed.add(kind)
        if not self._failed:
            self._set_swapping(False)
            self._set_status(True)
        else:
            # Some kind is dead: keep captions gated and the status red until
            # a successful swap of that kind (or translate-off) clears it.
            self._set_swapping(True)
            self._set_status(False, tr("a model failed to load"))
        with self._lock:
            self._busy = False
            pending = sorted(self._pending)
            self._pending.clear()
        for k in pending:
            # Re-evaluate every coalesced kind against current config on the GUI
            # thread; one that starts a swap makes later ones coalesce again,
            # no-ops drain.
            self._marshal(lambda k=k: self._reprocess(k))

    def _reprocess(self, kind: str) -> None:
        if self._on_pending is not None:
            self._on_pending(kind)  # recomputes target_id and calls request()


def _status_after_swap(ok: bool, reason: str, *, started, start, set_status) -> None:
    """Reloader status sink for :func:`run`: a successful swap must not paint
    "Capturing" while the pipeline never started (startup engine/mic failure).
    ``started`` is a one-element mutable flag; on first success ``start()``
    attempts the guarded pipeline start, reporting the mic on failure. Qt-free."""
    if not ok:
        set_status(False, reason)
        return
    if not started[0]:
        if not start():
            set_status(False, tr("microphone unavailable"))
            return
        started[0] = True
    set_status(True, reason)
