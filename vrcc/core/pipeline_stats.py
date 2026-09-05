"""Per-call STT timing accumulator, the session-cumulative totals it feeds,
and the end-of-run summary line.

Owned by :class:`~vrcc.core.pipeline.Pipeline`: ``self._stats`` is the
current run's accumulator (created once in ``__init__``, reset by
``begin_run`` on every :meth:`Pipeline.start`); ``self._session`` is the
whole-session accumulator (created once in ``__init__``, never reset) that
``log_summary`` folds each run into before deciding whether to emit a line.
``restart_source``/``reinit_audio_and_resume`` stop the pipeline and start it
again to swap a live device; they pass ``restarting=True`` into
``Pipeline.stop()`` so the fold happens but nothing is logged, and the
eventual real stop reports the whole session in one line instead of a
fragment per restart. Import direction: pipeline imports this module (never
the reverse at runtime).
"""

from __future__ import annotations

import logging
import threading
import time
from typing import TYPE_CHECKING

from vrcc.audio.frames import FRAME_LEN, SAMPLE_RATE

if TYPE_CHECKING:
    from vrcc.core.pipeline import Pipeline

# Same logger as the orchestrator: one operational stream for the pipeline.
logger = logging.getLogger("vrcc.core.pipeline")


class SttCallStats:
    """Thread-safe accumulator for one run's per-call STT engine timings.

    One lock acquisition per method, mirroring pipeline_state.py's
    convention: a caller never needs two calls where one lock block did
    the job.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._reset_locked()

    def reset(self) -> None:
        """Fresh run: never carry a prior run's timings into this one."""
        with self._lock:
            self._reset_locked()

    def _reset_locked(self) -> None:
        self.speculative_calls = 0
        self.final_calls = 0
        self.reuse_count = 0
        self.total_wall_s = 0.0
        self.max_wall_s = 0.0
        self.total_audio_s = 0.0
        # Read back at summary time as time.monotonic() - run_start: this
        # run's wall-clock duration, for the keep-up ratio (engine seconds
        # against time that actually elapsed, not audio processed).
        self.run_start = time.monotonic()

    def record_call(self, speculative: bool, audio_s: float, wall_s: float) -> None:
        """Fold in one completed engine call. The caller times the call
        itself outside any lock; this only takes the lock afterward, to add
        the finished numbers in."""
        with self._lock:
            if speculative:
                self.speculative_calls += 1
            else:
                self.final_calls += 1
            self.total_wall_s += wall_s
            self.max_wall_s = max(self.max_wall_s, wall_s)
            self.total_audio_s += audio_s

    def record_reuse(self) -> None:
        """A final reused its speculative's cached result: no engine call
        was made for it."""
        with self._lock:
            self.reuse_count += 1

    def snapshot(self) -> dict:
        """A plain-dict copy of the current counters, safe to read after the
        lock is released."""
        with self._lock:
            return {
                "speculative_calls": self.speculative_calls,
                "final_calls": self.final_calls,
                "reuse_count": self.reuse_count,
                "total_wall_s": self.total_wall_s,
                "max_wall_s": self.max_wall_s,
                "total_audio_s": self.total_audio_s,
                "run_start": self.run_start,
            }


class SessionStats:
    """Cumulative totals across every run of one Pipeline lifetime: from the
    first :meth:`Pipeline.start` through the real, non-restarting
    :meth:`Pipeline.stop`. A mid-session device swap folds its run into this
    instead of resetting it, so the log always shows the whole session
    rather than whatever slice happened since the last restart.

    Mutated only by :func:`log_summary`, always called with
    ``Pipeline._lifecycle_lock`` held: no lock of its own.
    """

    def __init__(self) -> None:
        self.speculative_calls = 0
        self.final_calls = 0
        self.reuse_count = 0
        self.total_wall_s = 0.0
        self.max_wall_s = 0.0
        self.total_audio_s = 0.0
        self.elapsed_s = 0.0
        self.dropped_frames = 0
        self.skipped_speculatives = 0
        self.stale_speculatives = 0

    def fold_in(
        self,
        call_snap: dict,
        elapsed_s: float,
        dropped_frames: int,
        skipped_speculatives: int,
        stale_speculatives: int,
    ) -> None:
        """Add one finished run's numbers into the running session totals."""
        self.speculative_calls += call_snap["speculative_calls"]
        self.final_calls += call_snap["final_calls"]
        self.reuse_count += call_snap["reuse_count"]
        self.total_wall_s += call_snap["total_wall_s"]
        self.max_wall_s = max(self.max_wall_s, call_snap["max_wall_s"])
        self.total_audio_s += call_snap["total_audio_s"]
        self.elapsed_s += elapsed_s
        self.dropped_frames += dropped_frames
        self.skipped_speculatives += skipped_speculatives
        self.stale_speculatives += stale_speculatives


def begin_run(p: "Pipeline") -> None:
    """Reset this run's counters at start(). ``p._session`` is untouched, so
    a restart keeps accumulating toward the one summary the eventual real
    stop emits."""
    p._dropped_frames = 0
    p._skipped_speculatives = 0
    p._stale_speculatives = 0
    p._stats.reset()


def log_summary(p: "Pipeline", *, restarting: bool = False) -> None:
    """Fold the run just stopped into ``p._session``, then, unless another
    start() is about to continue this same session (``restarting``), emit
    ONE INFO line covering the whole session so far: total calls (with the
    speculative/final split); engine time against audio actually fed to the
    engine, a per-call speed rather than a keep-up signal, since a
    speculative re-transcribes the growing prefix of the same utterance and
    this total counts overlapping audio more than once; engine time against
    the session's own wall-clock duration, the number that answers whether
    the pipeline kept up (at or above 1.0 means the engine was saturated);
    the average and slowest call; how many finals reused a speculative
    instead of an engine call; dropped frames; and speculatives shed under
    real backpressure (a full queue) versus ones dropped for the ordinary,
    costless reason that the speaker kept talking past them.

    Errors here are logged and swallowed, never raised: a stats failure
    must not keep the pipeline from stopping cleanly.
    """
    try:
        snap = p._stats.snapshot()
        elapsed_s = max(0.0, time.monotonic() - snap["run_start"])
        p._session.fold_in(
            snap, elapsed_s,
            p._dropped_frames, p._skipped_speculatives, p._stale_speculatives,
        )
        if restarting:
            return
        _emit(p._session)
    except Exception:  # noqa: BLE001 -- a stats failure must not break stop()
        logger.debug("STT run summary failed", exc_info=True)


def _emit(s: SessionStats) -> None:
    total_calls = s.speculative_calls + s.final_calls
    avg_wall_s = s.total_wall_s / total_calls if total_calls else 0.0
    per_call_text = (
        f"{s.total_wall_s / s.total_audio_s:.2f}x" if s.total_audio_s > 0 else "n/a"
    )
    load_text = f"{s.total_wall_s / s.elapsed_s:.2f}x" if s.elapsed_s > 0 else "n/a"
    dropped_s = s.dropped_frames * FRAME_LEN / SAMPLE_RATE

    logger.info(
        "STT run: %d calls (%d speculative, %d final). %.1fs engine time "
        "on %.1fs of audio fed to it (%s per call). Engine busy %.1fs of "
        "%.1fs run time (%s). Average %.2fs per call, slowest %.2fs. %d "
        "finals reused a speculative. Dropped %d frames, about %.1fs of "
        "audio. Skipped %d speculatives on a full queue (backpressure) and "
        "%d more because the speaker kept talking (normal, costs nothing).",
        total_calls, s.speculative_calls, s.final_calls,
        s.total_wall_s, s.total_audio_s, per_call_text,
        s.total_wall_s, s.elapsed_s, load_text,
        avg_wall_s, s.max_wall_s,
        s.reuse_count,
        s.dropped_frames, dropped_s,
        s.skipped_speculatives,
        s.stale_speculatives,
    )
