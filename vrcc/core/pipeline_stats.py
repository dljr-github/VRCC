"""Per-call STT timing accumulator, the input-character and latency
accumulators that ride alongside it, the session-cumulative totals they all
feed, and the end-of-run summary line.

Owned by :class:`~vrcc.core.pipeline.Pipeline`: ``self._stats`` is the
current run's STT-call accumulator, ``self._input`` the current run's
mic-input-character accumulator, ``self._latency`` the current run's
finalize-to-submit tracker (each created once in ``__init__``, reset by
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

import numpy as np

from vrcc.audio.frames import FRAME_LEN, SAMPLE_RATE

if TYPE_CHECKING:
    from vrcc.core.pipeline import Pipeline

# Same logger as the orchestrator: one operational stream for the pipeline.
logger = logging.getLogger("vrcc.core.pipeline")

# Frames are float32 in [-1, 1] (see vrcc.core.energy_gate's rms()); a sample
# at or beyond this magnitude is counted as clipped.
_FULL_SCALE = 1.0


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
        self.total_wait_s = 0.0
        self.max_wait_s = 0.0
        self.total_audio_s = 0.0
        self._latency_samples: list[float] = []
        # Read back at summary time as time.monotonic() - run_start: this
        # run's wall-clock duration, for the keep-up ratio (engine seconds
        # against time that actually elapsed, not audio processed).
        self.run_start = time.monotonic()

    def record_call(
        self, speculative: bool, audio_s: float, wait_s: float, call_s: float
    ) -> None:
        """Fold in one completed engine call, timed as two spans: ``wait_s``
        to acquire the STT slot's lock (EngineSlot.borrow() holds it across
        the whole call, so a concurrent model swap or the 'heard' stream
        blocks here, not the engine) and ``call_s`` for transcribe() itself.
        wall_s is their sum; the keep-up ratio and the average/slowest
        figures below both read wall_s. The caller times the call outside
        any lock; this only takes the lock afterward, to add the finished
        numbers in."""
        with self._lock:
            if speculative:
                self.speculative_calls += 1
            else:
                self.final_calls += 1
            wall_s = wait_s + call_s
            self.total_wall_s += wall_s
            self.max_wall_s = max(self.max_wall_s, wall_s)
            self.total_wait_s += wait_s
            self.max_wait_s = max(self.max_wait_s, wait_s)
            self.total_audio_s += audio_s

    def record_reuse(self) -> None:
        """A final reused its speculative's cached result: no engine call
        was made for it."""
        with self._lock:
            self.reuse_count += 1

    def record_latency(self, elapsed_s: float) -> None:
        """One utterance's finalize-to-chatbox-submission time (see
        LatencyTracker). Not a call: never touches speculative_calls/
        final_calls/reuse_count."""
        with self._lock:
            self._latency_samples.append(elapsed_s)

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
                "total_wait_s": self.total_wait_s,
                "max_wait_s": self.max_wait_s,
                "total_audio_s": self.total_audio_s,
                "latency_samples": list(self._latency_samples),
                "run_start": self.run_start,
            }


class InputStats:
    """What the microphone actually delivered this run, so a "captions
    stopped working" report has something to check besides the transcript:
    an RMS distribution, the clipped-frame fraction, and how many finals the
    STT engine's quality gate suppressed before anything downstream saw them.

    ``record_level`` is fed by the same MicLevel readings the meter already
    gets (one per frame in production, published from pipeline_frames's
    pre-gate or the segmenter's SegLevel; see Pipeline.__init__'s
    subscription), so the RMS is never computed twice. ``record_frame`` is
    fed the raw frame from
    Pipeline._seg_loop, the one place already holding it before it fans out,
    for the one check nothing upstream already makes: whether it clipped.
    Grows with run length, one float per frame kept in ``_rms_samples``; a
    caption session is not expected to run long enough for that to matter.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._reset_locked()

    def reset(self) -> None:
        with self._lock:
            self._reset_locked()

    def _reset_locked(self) -> None:
        self._rms_samples: list[float] = []
        self.frame_count = 0
        self.clipped_frames = 0
        self.gated_utterances = 0

    def record_level(self, rms: float) -> None:
        with self._lock:
            self._rms_samples.append(rms)

    def record_frame(self, frame: "np.ndarray") -> None:
        clipped = bool(np.any(np.abs(frame) >= _FULL_SCALE))
        with self._lock:
            self.frame_count += 1
            if clipped:
                self.clipped_frames += 1

    def record_gate_suppressed(self) -> None:
        """One final the STT engine's quality gate returned None for (no
        text, avg_logprob/no_speech_prob/compression_ratio outside its
        configured gates): nothing downstream ever sees it or its samples."""
        with self._lock:
            self.gated_utterances += 1

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "rms_samples": list(self._rms_samples),
                "frame_count": self.frame_count,
                "clipped_frames": self.clipped_frames,
                "gated_utterances": self.gated_utterances,
            }


class LatencyTracker:
    """Per-utterance wall clock from finalization to chatbox submission,
    keyed by utterance id so a submission on the MT worker thread can still
    find the start the STT worker recorded for it.

    This is the end-to-end figure a user reporting slow captions means:
    ``note_finalized`` is called from handle_final, when the segmenter hands
    off a completed utterance; ``pop_elapsed`` is called right after the
    ChatboxSender.submit_message call each send path makes (OSC's own
    coalesce/split delay past that point is not included). It is a
    different, wider window than CaptionModel.latency_ms in the GUI
    (vrcc/gui/caption_log.py, outside this module): that clock starts on
    PhraseRecognized, which _send_caption only publishes after the STT call
    has already returned, so it excludes STT queue wait, the engine lock
    wait and the inference itself.

    An utterance dropped between finalize and submit (regated,
    quality-gated, or abandoned mid-call by a stop) leaks its entry until
    ``reset()`` clears it at the next run's begin_run -- never across a
    restart, and never explicitly closed on those paths since they are the
    exception, not the common case.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._starts: dict[int, float] = {}

    def reset(self) -> None:
        with self._lock:
            self._starts.clear()

    def note_finalized(self, utterance_id: int) -> None:
        with self._lock:
            self._starts[utterance_id] = time.monotonic()

    def pop_elapsed(self, utterance_id: int) -> float | None:
        """Seconds since note_finalized(utterance_id), or None when it was
        never recorded (typed text has no finalize event) or already popped
        (nothing here submits the same utterance id twice today, but a
        second pop must not report a bogus near-zero span if that changes)."""
        with self._lock:
            start = self._starts.pop(utterance_id, None)
        if start is None:
            return None
        return max(0.0, time.monotonic() - start)


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
        self.total_wait_s = 0.0
        self.max_wait_s = 0.0
        self.total_audio_s = 0.0
        self.elapsed_s = 0.0
        self.dropped_frames = 0
        self.skipped_speculatives = 0
        self.stale_speculatives = 0
        self.latency_samples: list[float] = []
        self.rms_samples: list[float] = []
        self.frame_count = 0
        self.clipped_frames = 0
        self.gated_utterances = 0

    def fold_in(
        self,
        call_snap: dict,
        input_snap: dict,
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
        self.total_wait_s += call_snap["total_wait_s"]
        self.max_wait_s = max(self.max_wait_s, call_snap["max_wait_s"])
        self.total_audio_s += call_snap["total_audio_s"]
        self.latency_samples.extend(call_snap["latency_samples"])
        self.elapsed_s += elapsed_s
        self.dropped_frames += dropped_frames
        self.skipped_speculatives += skipped_speculatives
        self.stale_speculatives += stale_speculatives
        self.rms_samples.extend(input_snap["rms_samples"])
        self.frame_count += input_snap["frame_count"]
        self.clipped_frames += input_snap["clipped_frames"]
        self.gated_utterances += input_snap["gated_utterances"]


def begin_run(p: "Pipeline") -> None:
    """Reset this run's counters at start(). ``p._session`` is untouched, so
    a restart keeps accumulating toward the one summary the eventual real
    stop emits."""
    p._dropped_frames = 0
    p._skipped_speculatives = 0
    p._stale_speculatives = 0
    p._stats.reset()
    p._input.reset()
    p._latency.reset()


def note_dropped_frame(p: "Pipeline") -> None:
    """Count one oldest-frame drop from Pipeline._on_frame's queue-full
    path; warn once so a run that falls behind real time doesn't spam the
    log on every subsequent frame."""
    p._dropped_frames += 1
    if p._dropped_frames == 1:
        logger.warning(
            "frame queue full (>%d); dropping oldest frames -- the "
            "pipeline is falling behind real time (further drops counted)",
            p._frame_queue.maxsize,
        )


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
    the average and slowest call, and how much of that was spent waiting for
    the STT slot's lock rather than transcribing; how many finals reused a
    speculative instead of an engine call; dropped frames; speculatives shed
    under real backpressure (a full queue) versus ones dropped for the
    ordinary, costless reason that the speaker kept talking past them; the
    input level distribution and clipped-frame fraction; how many finals the
    quality gate suppressed; and the finalize-to-chatbox-submission latency.

    Errors here are logged and swallowed, never raised: a stats failure
    must not keep the pipeline from stopping cleanly.
    """
    try:
        snap = p._stats.snapshot()
        input_snap = p._input.snapshot()
        elapsed_s = max(0.0, time.monotonic() - snap["run_start"])
        p._session.fold_in(
            snap, input_snap, elapsed_s,
            p._dropped_frames, p._skipped_speculatives, p._stale_speculatives,
        )
        if restarting:
            return
        _emit(p._session)
    except Exception:  # noqa: BLE001 -- a stats failure must not break stop()
        logger.debug("STT run summary failed", exc_info=True)


def _percentile(sorted_values: list, pct: float) -> float:
    """Linear-interpolated percentile (0-100) of an already-sorted,
    non-empty list; callers only call this after checking for at least one
    sample."""
    n = len(sorted_values)
    if n == 1:
        return sorted_values[0]
    rank = (pct / 100.0) * (n - 1)
    lo = int(rank)
    hi = min(lo + 1, n - 1)
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (rank - lo)


def _emit(s: SessionStats) -> None:
    total_calls = s.speculative_calls + s.final_calls
    avg_wall_s = s.total_wall_s / total_calls if total_calls else 0.0
    per_call_text = (
        f"{s.total_wall_s / s.total_audio_s:.2f}x" if s.total_audio_s > 0 else "n/a"
    )
    load_text = f"{s.total_wall_s / s.elapsed_s:.2f}x" if s.elapsed_s > 0 else "n/a"
    dropped_s = s.dropped_frames * FRAME_LEN / SAMPLE_RATE
    wait_pct = (
        f"{100 * s.total_wait_s / s.total_wall_s:.0f}%" if s.total_wall_s > 0 else "n/a"
    )

    if s.rms_samples:
        levels = sorted(s.rms_samples)
        rms_text = (
            f"p10 {_percentile(levels, 10):.3f}, median {_percentile(levels, 50):.3f}, "
            f"p90 {_percentile(levels, 90):.3f} of 1.0 full scale"
        )
    else:
        rms_text = "n/a"
    clip_text = (
        f"{100 * s.clipped_frames / s.frame_count:.2f}%" if s.frame_count else "n/a"
    )
    if s.latency_samples:
        lat = sorted(s.latency_samples)
        latency_text = (
            f"median {_percentile(lat, 50):.2f}s, p90 {_percentile(lat, 90):.2f}s"
        )
    else:
        latency_text = "n/a"

    logger.info(
        "STT run: %d calls (%d speculative, %d final). %.1fs engine time "
        "on %.1fs of audio fed to it (%s per call). Engine busy %.1fs of "
        "%.1fs run time (%s). Average %.2fs per call, slowest %.2fs; %.2fs "
        "of that was spent waiting for the engine lock (%s), slowest wait "
        "%.2fs. %d finals reused a speculative. Dropped %d frames, about "
        "%.1fs of audio. Skipped %d speculatives on a full queue "
        "(backpressure) and %d more because the speaker kept talking "
        "(normal, costs nothing). Input level (frame RMS): %s. %s of frames "
        "clipped. %d finals suppressed by the quality gate. "
        "Finalize-to-chatbox latency: %s.",
        total_calls, s.speculative_calls, s.final_calls,
        s.total_wall_s, s.total_audio_s, per_call_text,
        s.total_wall_s, s.elapsed_s, load_text,
        avg_wall_s, s.max_wall_s, s.total_wait_s, wait_pct, s.max_wait_s,
        s.reuse_count,
        s.dropped_frames, dropped_s,
        s.skipped_speculatives,
        s.stale_speculatives,
        rms_text, clip_text, s.gated_utterances, latency_text,
    )
