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
    PhraseTranslated,
    VrchatDetected,
)

# Minimum spacing between forwarded MicLevel emits (~10 Hz). The segmenter
# produces ~31 Hz; anything faster than this is dropped.
_MIC_MIN_INTERVAL_S = 0.1


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

    def __init__(self, bus: EventBus, clock: Callable[[], float] = time.monotonic) -> None:
        super().__init__()
        self._bus = bus
        self._clock = clock
        # Monotonic time of the last forwarded MicLevel; None until the first (always emits).
        self._last_mic_emit: float | None = None

        # Keep the unsubscribe callables so detach() can undo every wiring.
        self._unsubs: list[Callable[[], None]] = [
            bus.subscribe(MicLevel, self._on_mic_level),
            bus.subscribe(PhraseRecognized, self.phrase_recognized.emit),
            bus.subscribe(PhraseTranslated, self.phrase_translated.emit),
            bus.subscribe(ChatboxSent, self.chatbox_sent.emit),
            bus.subscribe(MuteChanged, self.mute_changed.emit),
            bus.subscribe(DownloadProgress, self.download_progress.emit),
            bus.subscribe(EngineStateChanged, self.engine_state.emit),
            bus.subscribe(AppError, self.app_error.emit),
            bus.subscribe(VrchatDetected, self.vrchat_detected.emit),
        ]

    def _on_mic_level(self, event: MicLevel) -> None:
        """Time-gate MicLevel to ~10 Hz. Runs on the segmenter thread; the
        subsequent ``emit`` queues onto the GUI thread."""
        now = self._clock()
        if self._last_mic_emit is not None and (now - self._last_mic_emit) < _MIC_MIN_INTERVAL_S:
            return
        self._last_mic_emit = now
        self.mic_level.emit(event.rms, event.vad_prob)

    def detach(self) -> None:
        """Unsubscribe every handler from the bus. Idempotent."""
        for unsub in self._unsubs:
            unsub()
        self._unsubs = []
