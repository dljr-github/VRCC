"""Adapts the headless EventBus to Qt signals: the single seam between the
threaded engine and the GUI.

BusBridge lives on the GUI thread, so Qt auto-connection turns every cross-thread
emit into a queued GUI-thread delivery -- slots never touch engine threads or need
a lock (never force DirectConnection). MicLevel is time-gated to ~10 Hz (segmenter
emits ~31 Hz) so mic frames can't flood the event queue.
"""

from __future__ import annotations

import time
from typing import Callable

from PySide6.QtCore import QObject, Signal

from vrcc.core.bus import EventBus
from vrcc.core.events import (
    AppError,
    ChatboxSent,
    DownloadProgress,
    EngineStateChanged,
    MicLevel,
    MuteChanged,
    PhraseRecognized,
    HeardLevel,
    HeardPhrase,
    PhraseTranslated,
    UpdateCheckResult,
    VrchatDetected,
)

# Minimum spacing between forwarded level emits (~10 Hz). A segmenter produces
# ~31 Hz; anything faster than this is dropped. Both meters share the gate: two
# ungated streams would put three times the traffic on the GUI thread that one
# was tuned down to.
_LEVEL_MIN_INTERVAL_S = 0.1


class BusBridge(QObject):
    """Re-emits :class:`EventBus` events as Qt signals on the GUI thread.

    Construct on the GUI thread. Signals carry the event object unchanged
    (``Signal(object)``), except ``mic_level`` which is unpacked to
    ``(rms, vad_prob)`` floats for the meter/VAD widgets. Call :meth:`detach`
    on teardown to unsubscribe from the bus.
    """

    mic_level = Signal(float, float)  # rms, vad_prob
    phrase_recognized = Signal(object)  # PhraseRecognized
    phrase_translated = Signal(object)  # PhraseTranslated
    chatbox_sent = Signal(object)  # ChatboxSent
    mute_changed = Signal(object)  # MuteChanged
    download_progress = Signal(object)  # DownloadProgress
    engine_state = Signal(object)  # EngineStateChanged
    app_error = Signal(object)  # AppError
    vrchat_detected = Signal(object)  # VrchatDetected
    update_result = Signal(object)  # UpdateCheckResult
    heard_phrase = Signal(object)  # HeardPhrase
    heard_level = Signal(float, float)  # rms, vad_prob

    def __init__(self, bus: EventBus, clock: Callable[[], float] = time.monotonic) -> None:
        super().__init__()
        self._bus = bus
        self._clock = clock
        # Monotonic time of the last forwarded level event per meter; None until
        # the first (always emits).
        self._last_level_emit: dict[str, float] = {}

        # Keep the unsubscribe callables so detach() can undo every wiring.
        self._unsubs: list[Callable[[], None]] = [
            bus.subscribe(MicLevel, self._on_mic_level),
            bus.subscribe(PhraseRecognized, self.phrase_recognized.emit),
            bus.subscribe(PhraseTranslated, self.phrase_translated.emit),
            bus.subscribe(HeardPhrase, self.heard_phrase.emit),
            bus.subscribe(HeardLevel, self._on_heard_level),
            bus.subscribe(ChatboxSent, self.chatbox_sent.emit),
            bus.subscribe(MuteChanged, self.mute_changed.emit),
            bus.subscribe(DownloadProgress, self.download_progress.emit),
            bus.subscribe(EngineStateChanged, self.engine_state.emit),
            bus.subscribe(AppError, self.app_error.emit),
            bus.subscribe(VrchatDetected, self.vrchat_detected.emit),
            bus.subscribe(UpdateCheckResult, self.update_result.emit),
        ]

    def _on_heard_level(self, event) -> None:
        self._emit_level("heard", self.heard_level, event)

    def _on_mic_level(self, event: MicLevel) -> None:
        self._emit_level("mic", self.mic_level, event)

    def _emit_level(self, meter: str, signal, event) -> None:
        """Time-gate a level event to ~10 Hz, then unpack it for the meter.
        Runs on a segmenter thread; the subsequent ``emit`` queues onto the GUI
        thread. Each meter keeps its own clock, so one stream falling silent
        does not hold the other's next frame."""
        now = self._clock()
        last = self._last_level_emit.get(meter)
        if last is not None and (now - last) < _LEVEL_MIN_INTERVAL_S:
            return
        self._last_level_emit[meter] = now
        signal.emit(event.rms, event.vad_prob)

    def detach(self) -> None:
        """Unsubscribe every handler from the bus. Idempotent."""
        for unsub in self._unsubs:
            unsub()
        self._unsubs = []
