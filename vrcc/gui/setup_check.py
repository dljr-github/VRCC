"""Owns the setup check panel's facts and decides when to show it.

Lives in the composition root beside VrchatDetector, not inside MainWindow.
Eleven existing tests call window.show() (test_caption_feed.py, test_main_window_ui.py,
test_raise_window.py); a panel parented to the window would leave each of them
an untorn-down second top level. A UI-language change destroys and rebuilds
the main window (window_swap.py), carrying only geometry, _engine_states and
_capture_ok, so a controller owned by the window would lose every fact the
bus already reported. And this has to exist before VrchatDetector.start(), to
catch the one VrchatDetected publish of the session; nothing replays it for a
late subscriber.
"""

from __future__ import annotations

from PySide6.QtCore import QObject, QTimer, Signal
from PySide6.QtWidgets import QApplication

from vrcc.core.bus import EventBus
from vrcc.core.config import ConfigStore
from vrcc.core.events import (
    ChatboxSent,
    EngineStateChanged,
    HeardLevel,
    MicLevel,
    PhraseRecognized,
    VrchatDetected,
)
from vrcc.gui.setup_panel import SetupPanel
from vrcc.gui.setup_steps import SetupFacts, evaluate, required_passed

# Two facts have no bus event of their own (setup_steps.py's own docstring):
# Pipeline.set_captioning publishes nothing, and osc.send_to_vrchat is a
# Settings field that can change while the panel is open. Polling both is
# also what survives a window rebuild for free: a `toggled` connection to a
# button dies with the window it belongs to, while `window._pipeline` is the
# one object every rebuilt window shares.
_POLL_MS = 250


class SetupCheck(QObject):
    """Subscribes to the bus directly rather than through BusBridge:
    disconnect_bridge (main_window.py) unbinds only its own fixed table of
    pairs, and BusBridge.detach() undoes only its own _unsubs, so anything
    subscribed here needs its own teardown in stop().
    """

    _event = Signal(object)

    def __init__(self, bus: EventBus, store: ConfigStore, window, detector) -> None:
        super().__init__()
        self._store = store
        # Outlives every window rebuild: make_window() closes over the same
        # Pipeline for the life of the process, only the window is replaced.
        self._pipeline = window._pipeline
        # Kept only for a later place_beside(), never read otherwise: a
        # UI-language change destroys and recreates this window, and this
        # controller deliberately survives that rebuild, so the reference
        # can outlive the C++ object it points at. _show_panel() below
        # checks shiboken6.isValid() before using it for that reason.
        self._window = window
        self._panel = SetupPanel()

        self._facts = SetupFacts(vrchat_found=detector.detected)
        self._mic_forwarded = False
        self._heard_forwarded = False
        self._last_states: dict[str, str] | None = None
        # Tracked apart from the persisted flag: a flag later reset to False
        # (reopening the check from Settings) while the facts are still all
        # passed must not be flipped straight back to True before the user
        # ever sees the reopened panel. Only a fresh transition into "passed"
        # writes the flag.
        self._was_required_passed = required_passed(self._facts)
        # Seeded from the STORED value rather than zero: a request written by
        # an earlier session would otherwise read as fresh here and pop the
        # panel open on a launch nobody asked it to.
        self._last_request = store.config.gui.setup_check_requests
        # Set on a request, acted on (and cleared) once no modal is active:
        # Settings writes the request while it is still the modal dialog on
        # screen, so it can arrive well before it is safe to act on. A launch
        # that has never completed the check opens the panel on the first tick
        # below, which is the same path a later request takes.
        self._reshow_pending = not store.config.gui.setup_check_done

        # Qt's queued auto-connection hops a worker-thread emit onto this
        # object's own (GUI) thread, the same mechanism BusBridge relies on.
        self._event.connect(self._on_event)
        self._unsubs = [
            bus.subscribe(MicLevel, self._on_mic_level),
            bus.subscribe(HeardLevel, self._on_heard_level),
            bus.subscribe(PhraseRecognized, self._event.emit),
            bus.subscribe(ChatboxSent, self._event.emit),
            bus.subscribe(VrchatDetected, self._event.emit),
            bus.subscribe(EngineStateChanged, self._event.emit),
        ]

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._recompute)
        self._timer.start(_POLL_MS)

        self._recompute()

    # -- worker-thread handlers: latch once, forward once -------------------

    def _on_mic_level(self, event: MicLevel) -> None:
        # Only "has sound ever arrived" matters here, unlike the live meter's
        # 10 Hz gate in BusBridge, so latch and stop forwarding rather than
        # keep re-emitting a stream this row never needs again.
        if self._mic_forwarded or event.rms <= 0:
            return
        self._mic_forwarded = True
        self._event.emit(event)

    def _on_heard_level(self, event: HeardLevel) -> None:
        if self._heard_forwarded or event.rms <= 0:
            return
        self._heard_forwarded = True
        self._event.emit(event)

    # -- GUI thread -----------------------------------------------------------

    def _on_event(self, event) -> None:
        if self._panel is None:
            # stop() already ran; a cross-thread emit queued before it can
            # still arrive here on a later event-loop turn.
            return
        if isinstance(event, MicLevel):
            self._facts.mic_seen = True
        elif isinstance(event, HeardLevel):
            self._facts.heard_seen = True
        elif isinstance(event, PhraseRecognized):
            # A typed Send carries a negative utterance id (pipeline_typed.py);
            # only a spoken one may satisfy this row.
            if event.utterance_id > 0:
                self._facts.spoken_utterance = True
        elif isinstance(event, ChatboxSent):
            self._facts.any_chatbox_send = True
            if event.utterance_id > 0:
                self._facts.spoken_chatbox_send = True
        elif isinstance(event, VrchatDetected):
            self._facts.vrchat_found = event.detected
        elif isinstance(event, EngineStateChanged):
            self._facts.engine_states[event.engine] = event.state
        self._recompute()

    def _recompute(self) -> None:
        if self._panel is None:
            return
        # Watches the request counter, not setup_check_done: the flag answers
        # "should this open by itself at launch?", which is a different
        # question from "the user just asked for it", and a user who dismissed
        # the panel before finishing holds the flag at False already, so a
        # press would change nothing. Not level-triggered on "unmet and
        # hidden" either: that re-fires every tick for the rest of an unmet
        # session, forcing the panel back within one poll of a user closing
        # it. Closing the panel changes neither field, so it stays closed.
        req = self._store.config.gui.setup_check_requests
        if req != self._last_request:
            self._last_request = req
            self._reshow_pending = True
        # A modeless panel must not appear under a modal dialog (Settings is
        # application-modal): act on a pending request only once none is
        # active, rather than dropping it if the poll lands mid-dialog.
        if self._reshow_pending and QApplication.activeModalWidget() is None:
            self._show_panel()
            self._reshow_pending = False

        # Read fresh every time rather than caching a value from
        # construction: neither field has a bus event to invalidate a cache.
        self._facts.captioning = self._pipeline.captioning_enabled
        self._facts.send_enabled = self._store.config.osc.send_to_vrchat

        states = evaluate(self._facts)
        if states != self._last_states:
            self._last_states = states
            self._panel.apply(states)

        now_passed = required_passed(self._facts)
        if now_passed and not self._was_required_passed:
            if not self._store.config.gui.setup_check_done:
                self._store.config.gui.setup_check_done = True
                self._store.save_soon()
        self._was_required_passed = now_passed

    def _show_panel(self) -> None:
        import shiboken6

        # Both must already be shown before place_beside: it reads
        # frameGeometry(), which under-reports on an unshown window (no
        # title bar yet). Shown unconditionally either way: a stale window
        # reference must not leave the panel undisplayed, only unpositioned.
        self._panel.show()
        if shiboken6.isValid(self._window):
            self._panel.place_beside(self._window)

    def stop(self) -> None:
        """Unsubscribe from the bus, stop polling, hide the panel. Idempotent
        even across an event-loop turn: the second call sees `_panel` already
        None and returns before touching the by-then-deleted widget."""
        if self._panel is None:
            return
        for unsub in self._unsubs:
            unsub()
        self._unsubs = []
        self._timer.stop()
        self._panel.close_panel()
        self._panel.deleteLater()
        self._panel = None


def start(bus: EventBus, store: ConfigStore, window, detector) -> SetupCheck:
    return SetupCheck(bus, store, window, detector)
