"""Pipeline orchestrator: mic -> segmenter -> STT -> translation -> chatbox.

Owns four run-bound worker threads (audio callback, segmenter, STT, MT);
composes the speculative-reuse/typing state (pipeline_state) and delegates
per-frame work to pipeline_frames and per-job work to pipeline_jobs. Zero Qt
so any layer can drive it; the load-bearing thread/lock contracts are noted
inline at each site.
"""

from __future__ import annotations

import logging
import queue
import threading
from typing import TYPE_CHECKING

import numpy as np

from vrcc.audio.segmenter import (
    SegDiscard, SegFinal, SegLevel, SegPartial, SegSpeculative, SegSpeechStart,
)
from vrcc.core import pipeline_frames, pipeline_jobs
from vrcc.core.events import AppError, MicLevel, SpeechStarted
from vrcc.core.pipeline_jobs import _NO_ENGINE
from vrcc.core.pipeline_state import SpecCache, TypingTracker

if TYPE_CHECKING:
    from vrcc.audio.source import AudioSource
    from vrcc.audio.segmenter import Segmenter
    from vrcc.core.bus import EventBus
    from vrcc.core.config import AppConfig
    from vrcc.osc.chatbox import ChatboxSender
    from vrcc.osc.mutesync import MuteSync
    from vrcc.stt.engine import SttEngine, SttResult
    from vrcc.translate.engine import TranslateEngine

logger = logging.getLogger("vrcc.core.pipeline")

# Frame queue cap: overflow drops the OLDEST frame (newer audio is worth more).
FRAME_QUEUE_MAX = 100
# STT/MT job queues: small so a slow engine backpressures up the chain.
JOB_QUEUE_MAX = 4
JOIN_TIMEOUT_S = 2.0
# Blocked-enqueue poll: re-check the stop flag so stop() can't deadlock.
_PUT_POLL_S = 0.1


class Pipeline:
    """Owns the running capture->caption->translate->send engine.

    ``mt`` may be ``None`` (translation off) and ``mute`` may be ``None`` (no
    mute sync). :meth:`start`/:meth:`stop` spin up / tear down the worker
    threads (2 s join timeouts); both are idempotent.
    """

    def __init__(
        self,
        config: "AppConfig",
        bus: "EventBus",
        source: "AudioSource",
        segmenter: "Segmenter",
        stt: "SttEngine",
        mt: "TranslateEngine | None",
        chatbox: "ChatboxSender",
        mute: "MuteSync | None" = None,
    ) -> None:
        self._config = config
        self._bus = bus
        self._source = source
        self._segmenter = segmenter
        self._stt = stt
        self._mt = mt
        self._chatbox = chatbox
        self._mute = mute

        # Master captioning toggle; joins the segmenter worker's listen gate
        # (frames are dropped, nothing buffered) and gates STT enqueue like a
        # muted mute-sync. Starts off: the user opts in via the main-window
        # toggle each launch.
        self._captioning = False
        # Transition memory for the segmenter worker's listen gate, owned by
        # that thread.
        self._frame_gated = False

        # _stt_lock/_mt_lock guard engine calls AND swaps, so an engine is
        # never unloaded mid-call; _swapping pauses new-caption creation during
        # a swap. Lock order: never held under the SpecCache/TypingTracker
        # locks (no cycle).
        self._stt_lock = threading.Lock()
        self._mt_lock = threading.Lock()
        self._swapping = False

        # Queues + threads (created fresh in start()).
        self._frame_queue: queue.Queue = queue.Queue(maxsize=FRAME_QUEUE_MAX)
        self._stt_queue: queue.Queue = queue.Queue(maxsize=JOB_QUEUE_MAX)
        self._mt_queue: queue.Queue = queue.Queue(maxsize=JOB_QUEUE_MAX)
        self._seg_thread: threading.Thread | None = None
        self._stt_thread: threading.Thread | None = None
        self._mt_thread: threading.Thread | None = None

        # Speculative-reuse and typing bookkeeping (each guards its own lock).
        self._spec = SpecCache()
        self._typing = TypingTracker()

        # Caps in-flight partial-transcription jobs at one: set in
        # handle_partial (segmenter thread), cleared in process_stt_job (STT
        # worker thread), both under this lock, so a burst of SegPartial
        # events can never flood the shared STT queue and starve final jobs.
        self._partial_lock = threading.Lock()
        self._partial_pending = False

        self._dropped_frames = 0

        self._lifecycle_lock = threading.Lock()
        # Current run's stop event: replaced each start() so a worker abandoned
        # by a timed-out stop() join keeps its own (set) event and can't be
        # un-stopped by a restart. Passed to workers as a thread arg.
        self._stop_flag = threading.Event()
        self._join_timeout_s = JOIN_TIMEOUT_S  # tests shrink this
        self._started = False
        # Capture intent across a failed restart_source (see that method).
        self._resume_pending = False

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        """Start workers and begin capture; no-op if running. If the source
        fails to open, workers are unwound, ``_started`` stays False and the
        error propagates (left as if start() was never called).
        """
        with self._lifecycle_lock:
            if self._started:
                return
            # Fresh stop event per run: a zombie worker from a prior run holds
            # the OLD (set) event, so this restart is invisible to it.
            stop = threading.Event()
            self._stop_flag = stop

            # Fresh queues + state so a restart never inherits a prior run's
            # sentinels, backlog, half-resolved utterances, or audio the
            # segmenter buffered from a previous device.
            self._frame_queue = queue.Queue(maxsize=FRAME_QUEUE_MAX)
            self._stt_queue = queue.Queue(maxsize=JOB_QUEUE_MAX)
            self._mt_queue = queue.Queue(maxsize=JOB_QUEUE_MAX)
            self._spec.reset()
            self._typing.reset()
            self._segmenter.reset()
            self._frame_gated = False
            self._dropped_frames = 0

            # Each worker is bound to THIS run's queue + stop event via thread
            # args (never re-read from self), so a worker abandoned by a
            # timed-out stop() can never consume a later run's queue.
            self._seg_thread = self._spawn(
                self._seg_loop, "PipelineSegmenter", self._frame_queue, stop
            )
            self._stt_thread = self._spawn(
                self._stt_loop, "PipelineSTT", self._stt_queue, stop
            )
            if self._mt is not None:
                self._mt_thread = self._spawn(
                    self._mt_loop, "PipelineMT", self._mt_queue, stop
                )

            # Source last (consumers ready before frames arrive). On a mic-open
            # failure, unwind the just-spawned workers so _started stays False
            # (submit_typed then reports PIPELINE_NOT_RUNNING, not a dead queue).
            try:
                self._source.start(self._on_frame)
            except Exception:
                stop.set()
                self._terminate(self._frame_queue, self._seg_thread)
                self._terminate(self._stt_queue, self._stt_thread)
                self._terminate(self._mt_queue, self._mt_thread)
                self._seg_thread = self._stt_thread = self._mt_thread = None
                raise
            self._started = True

    def stop(self) -> None:
        """Stop capture and join every worker (2 s each). Idempotent, safe
        before :meth:`start`. A pending speculative is abandoned (stop flag set
        -> in-flight jobs return without publishing). A join that times out
        abandons the worker as a daemon zombie bound to this run's queue/stop
        event, so it can't interfere with a subsequent :meth:`start`.
        """
        with self._lifecycle_lock:
            if not self._started:
                return
            self._started = False
            self._stop_flag.set()

            # Source first: no new frames after this.
            try:
                self._source.stop()
            except Exception:
                logger.warning("source.stop() raised during pipeline stop", exc_info=True)

            # Terminate each worker with a sentinel, in dependency order
            # (segmenter feeds STT feeds MT), joining before moving on.
            self._terminate(self._frame_queue, self._seg_thread)
            self._terminate(self._stt_queue, self._stt_thread)
            self._terminate(self._mt_queue, self._mt_thread)
            self._seg_thread = self._stt_thread = self._mt_thread = None

        # Best-effort: drop the typing indicator we may have left on.
        self._set_typing(False)

    @property
    def captioning_enabled(self) -> bool:
        return self._captioning

    @property
    def segmenter(self) -> "Segmenter":
        """The run's segmenter. Exposed so pipeline_jobs can request a commit
        (early sentence injection) without reaching for the private attribute."""
        return self._segmenter

    def set_captioning(self, enabled: bool) -> None:
        """Master captioning toggle; when off, the segmenter worker drops
        frames (nothing is buffered) and STT jobs aren't created."""
        self._captioning = bool(enabled)

    # -- live model swap ---------------------------------------------------

    def set_swapping(self, value: bool) -> None:
        """Pause/resume new-caption creation during a model swap."""
        self._swapping = bool(value)

    def detach_stt(self) -> "SttEngine | None":
        """Remove and return the current STT engine. Taking ``_stt_lock`` waits
        for any in-flight ``transcribe`` first, so it's never unloaded mid-call."""
        with self._stt_lock:
            old, self._stt = self._stt, None
            return old

    def set_stt(self, engine: "SttEngine | None") -> None:
        """Install a new STT engine (picked up by the next STT job)."""
        with self._stt_lock:
            self._stt = engine

    def detach_mt(self) -> "TranslateEngine | None":
        """Remove and return the current MT engine. Taking ``_mt_lock`` waits
        for any in-flight ``translate`` first, so it's never unloaded mid-call."""
        with self._mt_lock:
            old, self._mt = self._mt, None
            return old

    def set_mt(self, engine: "TranslateEngine | None") -> None:
        """Install a new MT engine (``None`` disables translation)."""
        with self._mt_lock:
            self._mt = engine

    def set_mute(self, mute: "MuteSync | None") -> None:
        """Install a mute-sync coordinator (``None`` removes it). The
        segmenter worker reads it per frame through the listen gate; the
        attribute write is atomic and a flip mid-frame takes effect on the
        next frame."""
        self._mute = mute

    def restart_source(self, new_source: "AudioSource") -> bool:
        """Swap the audio source live via the proven stop()/start() path;
        engines untouched. Returns whether capture runs after. A failed device
        open re-raises after start() unwinds itself, but the capture intent
        survives (``_resume_pending``), so the next swap to a good device
        resumes capture instead of inheriting the stopped state."""
        want_running = self._started or self._resume_pending
        if self._started:
            self.stop()
        self._source = new_source
        if want_running:
            self._resume_pending = True
            self.start()
            self._resume_pending = False
        return self._started

    def reinit_audio_and_resume(self, reinit, make_source) -> bool:
        """Stop capture, run ``reinit`` while no source stream is open, then
        resume on a freshly built source (built AFTER reinit so it resolves
        against the refreshed device list). Mirrors restart_source's
        stop()/start() intent-preservation. Returns whether capture runs after."""
        want_running = self._started or self._resume_pending
        if self._started:
            self.stop()
        reinit()
        self._source = make_source()
        if want_running:
            self._resume_pending = True
            self.start()
            self._resume_pending = False
        return self._started

    def set_source_gain(self, gain_db: float, auto: bool) -> None:
        """Push a live gain change to the current source (no restart). No-op if
        the source has no gain processor."""
        setter = getattr(self._source, "set_gain", None)
        if setter is not None:
            setter(gain_db, auto)

    @staticmethod
    def _spawn(target, name: str, *args) -> threading.Thread:
        thread = threading.Thread(target=target, args=args, name=name, daemon=True)
        thread.start()
        return thread

    def _terminate(self, q: queue.Queue, thread: threading.Thread | None) -> None:
        if thread is None:
            return
        # Deliver the stop sentinel. Stop flag is set, so no producer is adding;
        # on a full queue drop stale items to make room -- the sentinel MUST
        # land so a worker abandoned by the join timeout can drain and exit.
        delivered = False
        for _ in range(q.maxsize + 2):
            try:
                q.put_nowait(None)
                delivered = True
                break
            except queue.Full:
                try:
                    q.get_nowait()
                except queue.Empty:
                    pass
        if not delivered:
            try:
                q.put(None, timeout=self._join_timeout_s)
            except queue.Full:
                logger.warning("could not deliver stop sentinel to %s", thread.name)
        thread.join(timeout=self._join_timeout_s)
        if thread.is_alive():
            logger.warning(
                "%s did not exit within %.1fs (an engine call is still "
                "running); abandoning it -- it will drain its own queue and "
                "exit when the call returns",
                thread.name,
                self._join_timeout_s,
            )

    # -- audio callback ----------------------------------------------------

    def _on_frame(self, frame: np.ndarray) -> None:
        """Audio-callback-thread entry point: enqueue a frame, fast. Drops
        the oldest frame (logging once) when the queue is full."""
        q = self._frame_queue
        try:
            q.put_nowait(frame)
        except queue.Full:
            try:
                q.get_nowait()  # drop oldest
            except queue.Empty:
                pass
            try:
                q.put_nowait(frame)
            except queue.Full:
                pass
            self._note_dropped_frame()

    def _note_dropped_frame(self) -> None:
        self._dropped_frames += 1
        if self._dropped_frames == 1:
            logger.warning(
                "frame queue full (>%d); dropping oldest frames -- the "
                "pipeline is falling behind real time (further drops counted)",
                FRAME_QUEUE_MAX,
            )

    # -- segmenter thread --------------------------------------------------

    def _seg_loop(self, q: queue.Queue, stop: threading.Event) -> None:
        while True:
            frame = q.get()
            if frame is None:  # stop sentinel
                return
            if stop.is_set():
                continue  # stopping/abandoned: drain to the sentinel
            pipeline_frames.process_frame(self, frame)

    def _on_seg_event(self, event: object) -> None:
        """Dispatch one segmenter event. **Documented test seam** -- tests
        call this directly with synthetic ``Seg*`` events."""
        if isinstance(event, SegLevel):
            self._bus.publish(MicLevel(rms=event.rms, vad_prob=event.vad_prob))
        elif isinstance(event, SegSpeechStart):
            self._bus.publish(SpeechStarted(utterance_id=event.utterance_id))
        elif isinstance(event, SegSpeculative):
            pipeline_jobs.handle_speculative(self, event)
        elif isinstance(event, SegFinal):
            pipeline_jobs.handle_final(self, event)
        elif isinstance(event, SegDiscard):
            pipeline_jobs.handle_discard(self, event)
        elif isinstance(event, SegPartial):
            pipeline_jobs.handle_partial(self, event)
        # Unknown event types are ignored.

    def _should_caption(self) -> bool:
        if not self._captioning:
            return False
        if self._swapping:  # paused mid model-swap
            return False
        return not self.mute_gated()

    def mute_gated(self) -> bool:
        """Whether mute sync is currently holding captions back (GUI-polled
        so the capture label can name the reason for the pause). The segmenter
        worker's listen gate also reads it per frame."""
        mute = self._mute
        return mute is not None and not mute.should_caption()

    # -- STT worker --------------------------------------------------------

    def _stt_loop(self, q: queue.Queue, stop: threading.Event) -> None:
        while True:
            job = q.get()
            if job is None:  # stop sentinel
                return
            if stop.is_set():
                continue  # stopping/abandoned: drain to the sentinel
            try:
                pipeline_jobs.process_stt_job(self, job, stop)
            except Exception as exc:  # noqa: BLE001 -- one bad job must not stop the worker
                logger.exception("STT job failed")
                self._bus.publish(AppError("STT_JOB_FAILED", str(exc)))

    def _transcribe(self, samples: np.ndarray) -> "SttResult | None | object":
        """Transcribe under _stt_lock so a concurrent detach_stt waits before
        unloading (returns ``_NO_ENGINE`` when swapped out). Only _stt_lock is
        held (never SpecCache/TypingTracker), so no lock-order cycle."""
        with self._stt_lock:
            engine = self._stt
            if engine is None:
                return _NO_ENGINE
            return engine.transcribe(samples)

    # -- MT worker ---------------------------------------------------------

    def _mt_loop(self, q: queue.Queue, stop: threading.Event) -> None:
        while True:
            job = q.get()
            if job is None:  # stop sentinel
                return
            if stop.is_set():
                continue  # stopping/abandoned: drain to the sentinel
            try:
                pipeline_jobs.process_mt_job(self, job, stop)
            except Exception as exc:  # noqa: BLE001 -- one bad job must not stop the worker
                logger.exception("MT job failed")
                self._bus.publish(AppError("MT_JOB_FAILED", str(exc)))
                if job.manage_typing:
                    self._resolve_typing(job.utterance_id)

    # -- typed text --------------------------------------------------------

    def submit_typed(self, text: str) -> bool:
        """Send typed text straight through translation to the chatbox,
        bypassing STT and mute/captioning gating (utterance id 0). Returns False
        (``PIPELINE_NOT_RUNNING``) when not started, keeping the text uncaptured."""
        return pipeline_jobs.submit_typed(self, text)

    # -- typing helpers ------------------------------------------------------

    def _set_typing(self, value: bool) -> None:
        if not self._config.osc.send_to_vrchat:
            return
        try:
            self._chatbox.set_typing(value)
        except Exception:
            logger.warning("chatbox.set_typing raised; ignoring", exc_info=True)

    def _begin_typing(self, utterance_id: int) -> None:
        self._typing.begin(utterance_id)
        self._set_typing(True)

    def _resolve_typing(self, utterance_id: int) -> None:
        if self._typing.resolve(utterance_id):
            self._set_typing(False)

    # -- shared-state helpers ----------------------------------------------

    def _mark_finalized(self, utterance_id: int) -> None:
        """Bound the speculative caches, then defensively prune typing
        orphans below the cutoff (see TypingTracker.prune_orphans)."""
        cutoff = self._spec.mark_finalized(utterance_id)
        orphaned, emptied = self._typing.prune_orphans(cutoff)
        if orphaned:
            logger.warning(
                "pruned orphaned typing entries %s (segmenter invariant "
                "violated?)",
                sorted(orphaned),
            )
        if emptied:
            self._set_typing(False)

    def _enqueue(self, q: queue.Queue, job) -> None:
        """Put a job, applying backpressure (blocking) but waking to drop it if
        stop() is requested, so a full downstream queue never deadlocks stop."""
        while not self._stop_flag.is_set():
            try:
                q.put(job, timeout=_PUT_POLL_S)
                return
            except queue.Full:
                continue
