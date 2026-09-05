"""Transcribe and translate what the speakers are playing, for you to read.

The other half of :class:`~vrcc.core.pipeline.Pipeline`: that one captions the
voice going INTO VRChat, this one captions the voices coming out of it, so a
user can follow someone speaking a language they do not read.

Deliberately not a second Pipeline. Two properties are easier to guarantee with
a smaller dedicated path than with a flag threaded through the existing one:

It never reaches VRChat. This class holds no chatbox, no sender and no OSC of
any kind, so "other people's words are never broadcast under your name" is a
property of the construction rather than a branch someone could invert later.

It never runs two decodes at once. The STT and MT engines are SHARED with the
main pipeline rather than duplicated, under locks the caller passes in, because
a second copy of large-v3-turbo costs another 2.5 GB of VRAM and a 12 GB card
has no room for it. The cost is latency instead of memory: when both streams
speak together, one waits.

Source language is always detected. You cannot know in advance what someone
else will speak, which is the whole reason for the feature.

It also refuses to caption the user back to themselves. Loopback captures
everything the output device plays, and on a lot of setups that includes the
user's own voice: hardware direct monitoring (an Elgato Wave XLR does this),
sidetone on a headset, or any virtual "mix" output. Those are audio-routing
facts VRCC cannot change, and the result reads as the feature being broken,
so an utterance heard while the microphone was live is dropped rather than
shown. It costs the moments when someone talks over you, which is also when
the transcription would have been poor.
"""

from __future__ import annotations

import logging
import queue
import threading
import time

import numpy as np

from vrcc.audio.frames import SAMPLE_RATE
from vrcc.audio.segmenter import SegFinal, SegLevel
from vrcc.core.events import HeardLevel, HeardPhrase
from vrcc.core.languages import from_whisper, get

logger = logging.getLogger("vrcc.core.heard")

# Utterances waiting to be transcribed. Small on purpose: the heard stream is
# the lower-priority one, and a backlog means captions describing what was said
# a minute ago, which is worse than dropping the oldest.
_QUEUE_MAX = 4

_AUTO = "auto"

# How long after the user stops speaking their voice may still be arriving on
# the loopback, covering the monitoring path's own delay. It does not have to
# cover vad.finalize_silence_ms (600 ms by default, up to 5 s in Advanced)
# because _handle checks the utterance's whole capture window.
_SELF_ECHO_GRACE_S = 1.0

# Microphone loudness that counts as the user making a sound. Deliberately low:
# suppressing a little of someone else costs a caption, while captioning the
# user back to themselves makes the feature look broken.
#
# RMS rather than the VAD probability because MicLevel's vad_prob is ZERO
# whenever the pipeline is gated (pipeline_frames.process_frame returns before
# the VAD runs when captioning is off or mute sync is holding), which is
# exactly the state this feature is used in. RMS is real in every state.
_MIC_ACTIVE_RMS = 0.01


class HeardStream:
    """Capture -> segment -> transcribe -> translate, published for display."""

    def __init__(
        self,
        config,
        bus,
        source,
        segmenter,
        stt,
        mt,
        stt_lock: threading.Lock,
        mt_lock: threading.Lock,
    ) -> None:
        self._config = config
        self._bus = bus
        self._source = source
        self._segmenter = segmenter
        self._stt = stt
        self._mt = mt
        # The MAIN pipeline's locks, not fresh ones: sharing an engine is only
        # safe if both callers serialise on the same object.
        self._stt_lock = stt_lock
        self._mt_lock = mt_lock

        # Monotonic timestamp of the last frame the MICROPHONE judged to be
        # speech, written by the bus thread and read by the worker. A float
        # assignment is atomic under the GIL, so no lock is warranted.
        self._user_spoke_at = 0.0
        self._suppressed = 0

        self._queue: queue.Queue = queue.Queue(maxsize=_QUEUE_MAX)
        self._thread: threading.Thread | None = None
        self._running = False
        # Identity of the current run. A join that times out (a decode can hold
        # the shared STT lock for seconds) leaves a worker alive, and comparing
        # this token is how that worker learns it is no longer the one: it
        # publishes nothing and exits rather than racing its replacement.
        self._run_token: object | None = None
        self.dropped = 0

    # -- lifecycle ---------------------------------------------------------

    def note_mic_level(self, rms: float, vad_prob: float = 0.0) -> None:
        """Called for every microphone frame, so the stream knows when the user
        is making sound.

        Either signal counts. vad_prob is the better judgement but is published
        as 0.0 whenever the pipeline is gated, and captioning starts off every
        launch, so relying on it alone left this guard inert in the one state
        the feature is normally used in.
        """
        if rms >= _MIC_ACTIVE_RMS or vad_prob >= self._config.vad.threshold:
            self._user_spoke_at = time.monotonic()

    @property
    def bus(self):
        return self._bus

    def set_stt(self, engine) -> None:
        """Point the stream at the voice engine the pipeline now holds.

        The engines are shared, and only the pipeline is told when one is
        swapped. Without this the stream keeps the object the reloader already
        unloaded, every decode raises into :meth:`_work`'s handler, and the
        feature goes silent for the rest of the session with its toggle still
        lit. Taken under the shared lock, so the swap cannot land while a
        decode is running on the engine about to be unloaded.
        """
        with self._stt_lock:
            self._stt = engine

    def set_mt(self, engine) -> None:
        """Point the stream at the translator the pipeline now holds. Also the
        path by which translation switched on after launch reaches it."""
        with self._mt_lock:
            self._mt = engine

    def reconfigure_vad(self, cfg) -> None:
        """Adopt new VAD timings, as the main pipeline's segmenter does.

        Its own segmenter means its own precomputed frame counts, and left
        behind they silently diverged: the same silence ended a microphone
        utterance but not a speaker one, and this stream's echo grace is
        reasoned about in terms of the finalize timing it was no longer using.
        """
        self._segmenter.reconfigure(cfg)

    def set_source(self, source) -> None:
        """Swap the capture source. Only while stopped: the running one owns a
        thread reading a device. Lets a speaker change take effect without a
        relaunch, since the source is bound to one device for its lifetime."""
        if self._running:
            self.stop()
        self._source = source

    @property
    def running(self) -> bool:
        return self._running

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._segmenter.reset()
        # A fresh queue per run. What was queued before the user switched this
        # off is stale by the time they switch it back on, and captioning it
        # as if it had just been said is the backlog _QUEUE_MAX exists to
        # prevent.
        self._queue = queue.Queue(maxsize=_QUEUE_MAX)
        token = self._run_token = object()
        self._thread = threading.Thread(
            target=self._work, args=(token, self._queue),
            name="HeardStream", daemon=True,
        )
        self._thread.start()
        self._source.start(self._on_frame)

    def stop(self) -> None:
        self._running = False
        self._run_token = None
        try:
            self._source.stop()
        except Exception:
            logger.warning("heard source failed to stop", exc_info=True)
        # Wake the worker off its blocking get.
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass
        thread, self._thread = self._thread, None
        if thread is not None:
            thread.join(timeout=2.0)

    # -- capture -----------------------------------------------------------

    def _on_frame(self, frame: np.ndarray) -> None:
        """Segment on the capture thread; only whole utterances are queued.

        Frames arrive about 31 times a second and the VAD is cheap, so handing
        every one to a queue would cost more in wakeups than it saves.
        """
        if not self._running:
            return
        try:
            events = self._segmenter.process(frame)
        except Exception:
            logger.warning("heard segmenter raised", exc_info=True)
            return
        for event in events:
            if isinstance(event, SegLevel):
                # Every frame, so the meter shows what the speakers are
                # actually feeding this stream even when nothing is loud
                # enough to become an utterance.
                self._bus.publish(
                    HeardLevel(rms=event.rms, vad_prob=event.vad_prob)
                )
                continue
            if not isinstance(event, SegFinal):
                continue
            # Both stamped at capture. captured_at lets the echo check ask
            # about the whole window rather than only the instant it was
            # dequeued, and spoke_at freezes what the microphone was doing
            # while this utterance was being captured: read later it would also
            # catch the user's REPLY, which is normal turn-taking and would
            # suppress the other person systematically.
            item = (event.samples, time.monotonic(), self._user_spoke_at)
            try:
                self._queue.put_nowait(item)
            except queue.Full:
                # Oldest first: a caption of what was said a minute ago is
                # worse than not captioning it.
                self.dropped += 1
                logger.debug("heard queue full; dropping the oldest utterance")
                try:
                    self._queue.get_nowait()
                    self._queue.put_nowait(item)
                except (queue.Empty, queue.Full):
                    # The worker drained or refilled it in between; either way
                    # the backlog is no longer this frame's problem.
                    pass

    # -- worker ------------------------------------------------------------

    def _work(self, token: object, q: queue.Queue) -> None:
        while self._run_token is token:
            item = q.get()
            if item is None or self._run_token is not token:
                continue
            try:
                self._handle(token, *item)
            except Exception:
                logger.warning("heard utterance failed", exc_info=True)

    def _handle(
        self, token: object, samples: np.ndarray, captured_at: float, spoke_at: float
    ) -> None:
        # The whole capture window, not just its end. The segmenter will not
        # close an utterance until finalize_silence_ms of silence, so a reply
        # arriving sooner is welded onto the user's own speech and an
        # end-of-utterance check would pass the merged audio through.
        window_start = captured_at - len(samples) / SAMPLE_RATE
        if spoke_at >= window_start - _SELF_ECHO_GRACE_S:
            self._suppressed += 1
            logger.debug("dropped a heard utterance that overlapped your speech")
            return
        with self._stt_lock:
            # This stream must never inherit the user's configured spoken
            # language: an English speaker's setting would decode every
            # Japanese speaker in the room as English, which produces
            # confident nonsense rather than an error.
            #
            # Read inside the lock: a hot swap sets it to None here first, so
            # an engine that is about to be unloaded is never entered.
            engine = self._stt
            if engine is None:
                return
            result = engine.transcribe(samples, detect_language=True)
        if result is None or not result.text.strip():
            return

        translations = self._translate(result)
        if self._run_token is not token:
            return  # switched off while this was decoding
        self._bus.publish(
            HeardPhrase(
                text=result.text,
                language=result.language,
                translations=translations,
            )
        )

    def _translate(self, result) -> list[tuple[str, str]]:
        """Translate into the user's own languages, which is the direction that
        helps: they need what was said rendered into something they read."""
        cfg = self._config.translate
        if self._mt is None or not cfg.enabled:
            return []
        # from_whisper, not get: engines report the language they detected as
        # a code, and get() keys on display names, so every utterance raised
        # KeyError, was caught here, and returned no translations at all. The
        # transcription still appeared, which is why it read as "translation is
        # just slow" rather than as a failure.
        source = from_whisper(result.language)
        if source is None:
            return []
        try:
            # get() inside the try, not above it. It raises on a display name
            # the registry does not know (a hand-edited config, a name a later
            # build renamed), and outside the try that KeyError unwinds past
            # the publish below, costing the transcription as well as the
            # translation.
            targets = [
                get(name)
                for name in self._heard_targets()
                if name != source.display
            ]
            if not targets:
                return []
            with self._mt_lock:
                engine = self._mt
                if engine is None:
                    return []
                return engine.translate(result.text, source, targets)
        except Exception:
            logger.warning("heard translation failed", exc_info=True)
            return []

    def _heard_targets(self) -> list[str]:
        """What to render their speech INTO.

        A language picked for this stream wins. Otherwise the language the user
        speaks, which is right for almost everyone and needs no setting.

        Never translate.targets, which is the outbound direction (what the room
        reads). Someone speaking Japanese at an English speaker should be shown
        English, and translate.targets would send it back to Japanese.
        """
        chosen = self._config.audio.hear_others_language
        if chosen:
            return [chosen]
        spoken = self._config.stt.spoken_languages
        if spoken:
            return list(spoken)
        source = self._config.stt.source_language
        return [] if source == _AUTO else [source]
