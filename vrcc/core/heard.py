"""Transcribe and translate what the speakers are playing, for you to read.

The other half of :class:`~vrcc.core.pipeline.Pipeline`: that one captions the
voice going INTO VRChat, this one captions the voices coming out of it, so a
user can follow someone speaking a language they do not read.

Deliberately not a second Pipeline. Two properties are easier to guarantee with
a smaller dedicated path than with a flag threaded through the existing one:

It never reaches VRChat. This class holds no chatbox, no sender and no OSC of
any kind, so "other people's words are never broadcast under your name" is a
property of the construction rather than a branch someone could invert later.
The room already heard them; what the user lacks is understanding, not a relay.

It never runs two decodes at once. The STT and MT engines are SHARED with the
main pipeline rather than duplicated, under locks the caller passes in, because
a second copy of large-v3-turbo costs another 2.5 GB of VRAM and a 12 GB card
has no room for it. The cost is latency instead of memory: when both streams
speak together, one waits. Conversation alternates most of the time, so that is
usually free, and when it is not, a late caption beats a card that cannot load
the model.

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

from vrcc.audio.segmenter import SegFinal, SegLevel
from vrcc.core.events import HeardLevel, HeardPhrase
from vrcc.core.languages import get

logger = logging.getLogger("vrcc.core.heard")

# Utterances waiting to be transcribed. Small on purpose: the heard stream is
# the lower-priority one, and a backlog means captions describing what was said
# a minute ago, which is worse than dropping the oldest.
_QUEUE_MAX = 4

_AUTO = "auto"

# How long after the user stops speaking their voice may still be arriving on
# the loopback. Covers the monitoring path's own delay plus the tail of an
# utterance already in the segmenter, without swallowing a reply that lands
# immediately afterwards.
_SELF_ECHO_GRACE_S = 1.0


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
        self.dropped = 0

    # -- lifecycle ---------------------------------------------------------

    def note_mic_level(self, vad_prob: float) -> None:
        """Called for every microphone frame, so the stream knows when the user
        is talking. Fed from MicLevel rather than reaching into the pipeline:
        the mic's own VAD has already made this judgement."""
        if vad_prob >= self._config.vad.threshold:
            self._user_spoke_at = time.monotonic()

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
        self._thread = threading.Thread(
            target=self._work, name="HeardStream", daemon=True
        )
        self._thread.start()
        self._source.start(self._on_frame)

    def stop(self) -> None:
        self._running = False
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
            try:
                self._queue.put_nowait(event.samples)
            except queue.Full:
                # Oldest first: a caption of what was said a minute ago is
                # worse than not captioning it.
                self.dropped += 1
                logger.debug("heard queue full; dropping an utterance")

    # -- worker ------------------------------------------------------------

    def _work(self) -> None:
        while self._running:
            samples = self._queue.get()
            if samples is None or not self._running:
                continue
            try:
                self._handle(samples)
            except Exception:
                logger.warning("heard utterance failed", exc_info=True)

    def _handle(self, samples: np.ndarray) -> None:
        if time.monotonic() - self._user_spoke_at < _SELF_ECHO_GRACE_S:
            # The user was talking while this was captured, so it is most
            # likely their own voice arriving back through monitoring.
            self._suppressed += 1
            logger.debug("dropped a heard utterance that overlapped your speech")
            return
        with self._stt_lock:
            result = self._stt.transcribe(samples)
        if result is None or not result.text.strip():
            return

        translations = self._translate(result)
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
        try:
            source = get(result.language) if result.language != _AUTO else None
        except KeyError:
            source = None
        if source is None:
            return []
        targets = [get(name) for name in self._heard_targets() if name != source.display]
        if not targets:
            return []
        try:
            with self._mt_lock:
                return self._mt.translate(result.text, source, targets)
        except Exception:
            logger.warning("heard translation failed", exc_info=True)
            return []

    def _heard_targets(self) -> list[str]:
        """What to render their speech INTO: the language the user speaks.

        Not translate.targets, which is the outbound direction (what the room
        reads). Someone speaking Japanese at an English speaker should be shown
        English, and translate.targets would send it back to Japanese.
        """
        spoken = self._config.stt.spoken_languages
        if spoken:
            return list(spoken)
        source = self._config.stt.source_language
        return [] if source == _AUTO else [source]
