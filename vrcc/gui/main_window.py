"""The single top-level window: live captions, meters and quick controls.

A thin view over a BusBridge (Qt signals for live updates), a ConfigStore
(read to fill controls, written back + save_soon() on edits) and a pipeline
(submit_typed/set_captioning/captioning_enabled). Threading: every bridge
signal is delivered on the GUI thread, so slots mutate widgets without locks.
"""

from __future__ import annotations

import base64
import logging
from typing import Callable

from PySide6.QtCore import QByteArray
from PySide6.QtWidgets import QComboBox, QMainWindow, QMessageBox, QVBoxLayout, QWidget

from vrcc.core.config import ConfigStore, apply_profile
from vrcc.gui import main_heard, main_targets, model_prompts, status_render
from vrcc.gui.bridge import BusBridge
from vrcc.gui.caption_log import (
    CaptionModel,
    empty_state_html,
    empty_state_text,
    render_row_html,
)
from vrcc.gui.icons import FRIENDLY_ERRORS as _FRIENDLY_ERRORS
from vrcc.gui.icons import dots_svg as _dots_svg  # re-exported: tests import it from here
from vrcc.gui.log_follow import LogFollower
from vrcc.gui.main_parts import (
    build_caption_log,
    build_compose_row,
    build_status_strip,
    build_top_bar,
)
from vrcc.gui.style import PALETTE, resolve_theme
from vrcc.gui.widgets import set_combo_value
from vrcc.i18n import tr, tr_noop

logger = logging.getLogger("vrcc.gui.main_window")

# Transient status/error text lingers this long before clearing.
_TRANSIENT_MS = 5000

# The three target-language slots. Slot 0 is always active (no checkbox).
_NUM_TARGET_SLOTS = 3

# Plain, user-facing engine names, so failure/fallback messages never leak
# "STT"/"MT" jargon. tr_noop: translated at the point of use, not import time.
_ENGINE_NAMES = {"stt": tr_noop("Voice model"), "mt": tr_noop("Translation model")}
# Sentence-internal lowercase forms (never lowercase a translated string in code).
_ENGINE_NAMES_LOWER = {"stt": tr_noop("voice model"), "mt": tr_noop("translation model")}


class MainWindow(QMainWindow):
    def __init__(
        self,
        bridge: BusBridge,
        config_store: ConfigStore,
        pipeline,
        on_open_settings: Callable[[], None],
        on_open_models: Callable[[], None],
        mt_available: bool = True,
        download_manager=None,
        on_model_change=None,
        on_check_updates=None,
        on_hear_others=None,
    ) -> None:
        super().__init__()
        self._bridge = bridge
        self._store = config_store
        self._pipeline = pipeline
        # Optional collaborators (None when headless): both feed the
        # language-change model nudge, mirroring SettingsDialog's pair.
        self._download_manager = download_manager
        self._on_model_change = on_model_change
        # Language-nudge state: one queued prompt at a time, and the declined
        # (model, language) pair so config reloads do not re-ask it.
        self._nudge_pending = False
        self._nudge_declined: tuple[str, str] | None = None
        # Resolved once at construction (theme + text size are restart-applied).
        self._p = PALETTE[resolve_theme(config_store.config.gui.theme)]
        self._scale = max(0.5, min(2.0, config_store.config.gui.font_scale))
        self._on_open_settings = on_open_settings
        self._on_open_models = on_open_models
        self._on_check_updates = on_check_updates
        # Starts/stops the speaker capture live (None when headless), so the
        # toggle never needs a relaunch to take effect.
        self._on_hear_others = on_hear_others
        # Kept for caller compat but no longer read: engines hot-swap mid-session,
        # so a launch-time "was MT built?" snapshot would wrongly suppress the
        # "translating…" row. Live config is the only correct source of truth.
        self._mt_available = mt_available

        # Guards config writes while we push config values INTO widgets during
        # construction, so setCurrentText/setValue don't echo back to disk.
        self._loading = True
        # Re-entrancy guard: _rebuild_targets re-points a pill that cannot
        # stand, and that edit would otherwise call it straight back in.
        self._rebuilding = False
        # Latest engine states, rendered together in the status bar.
        self._engine_states: dict[str, str] = {}
        # Engine kinds already shown a failure modal, cleared when one goes
        # ready again. Without it a kind that retries stacks one dialog a try.
        self._engine_failures_reported: set[str] = set()
        # Per-utterance caption rows with delivery status (pure model, re-rendered).
        self._caption_model = CaptionModel()

        self.setWindowTitle("VRCC")
        self._build_ui()
        self._restore_geometry()
        self._load_from_config()
        self._connect_bridge()

        self._loading = False

    # -- construction ------------------------------------------------------

    def _build_ui(self) -> None:
        central = QWidget()
        root = QVBoxLayout(central)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)
        root.addWidget(build_top_bar(self))
        root.addWidget(build_status_strip(self))
        root.addWidget(build_caption_log(self), stretch=1)
        root.addWidget(build_compose_row(self))
        self.setCentralWidget(central)
        self._log_follow = LogFollower(self._log)
        # statusBar() builds the bar on first call, and building it takes its
        # height out of the caption feed. Left lazy, the first status flash of
        # a session shifts the feed under the user (measured 184px -> 162px).
        self.statusBar()

        # The capture label is the honest "is the app working?" answer: it stays
        # red "Not listening" after any load/mic/engine failure, so a broken app
        # never looks healthy.
        self.set_capture_status(None)
        # OSC is fire-and-forget with no ack, so VRChat reachability comes from
        # mDNS discovery of its OSCQuery service.
        self._on_vrchat_detected(None)

        # Render the initial empty/loading state so the log never opens blank.
        self._render_log()

    # -- target add/remove -------------------------------------------------

    def _add_target(self) -> None:
        main_targets.add_target(self, _NUM_TARGET_SLOTS)

    def _seed_free_target(self, slot: int) -> None:
        main_targets.seed_free_target(self, slot)

    def _remove_target(self, slot: int) -> None:
        main_targets.remove_target(self, slot, _NUM_TARGET_SLOTS)

    def _sync_target_visibility(self) -> None:
        main_targets.sync_target_visibility(self, _NUM_TARGET_SLOTS)

    # -- initial values from config ----------------------------------------

    def _load_from_config(self) -> None:
        cfg = self._store.config

        # Grey the spoken languages the active voice model can't serve; while
        # translation is on that includes "auto" for onnx-asr models, which
        # detect but can't report the language. Point the combo at the stored
        # language after (re-run on every re-sync so model or translation
        # changes made in Settings re-enable the right entries).
        model_prompts.grey_unsupported_languages(
            self._source_combo, cfg.stt.model, translating=cfg.translate.enabled
        )
        set_combo_value(self._source_combo, cfg.stt.source_language)
        # A stored language the greying just disabled would caption wrongly in
        # silence; offer a better downloaded model once construction settles.
        model_prompts.schedule_language_nudge(self)

        main_targets.load_targets(self, cfg)

        # Re-read on every reload, so ticking it in Settings moves the button
        # and vice versa: two controls for one setting must never disagree.
        self._hear_btn.setChecked(bool(cfg.audio.hear_others_enabled))
        self._heard_meter.set_active(bool(cfg.audio.hear_others_enabled))

        self._captioning_btn.setChecked(bool(self._pipeline.captioning_enabled))
        # setChecked only emits toggled on a state change, so sync the meter's
        # active/dimmed state directly, else it stays bright when captioning loads off.
        self._mic_meter.set_active(self._captioning_btn.isChecked())
        self._sync_target_visibility()
        # Settings can move the mute gate (mode/enable edits) through the
        # shared config with no bus event; re-derive the capture label.
        self._render_capture_status()

    @staticmethod
    def _set_combo_text(combo: QComboBox, text: str) -> None:
        idx = combo.findText(text)
        if idx >= 0:
            combo.setCurrentIndex(idx)

    # -- bridge signal wiring ----------------------------------------------

    def _bridge_bindings(self):
        b = self._bridge
        return (
            (b.mic_level, self._on_mic_level),
            (b.phrase_recognized, self._on_phrase_recognized),
            (b.phrase_translated, self._on_phrase_translated),
            (b.chatbox_sent, self._on_chatbox_sent),
            (b.mute_changed, self._on_mute_changed),
            (b.engine_state, self._on_engine_state),
            (b.download_progress, self._on_download_progress),
            (b.app_error, self._on_app_error),
            (b.vrchat_detected, self._on_vrchat_detected),
            (b.update_result, self._on_update_result),
            (b.heard_phrase, self._on_heard_phrase),
            (b.heard_level, self._on_heard_level),
        )

    def _connect_bridge(self) -> None:
        for signal, slot in self._bridge_bindings():
            signal.connect(slot)

    def disconnect_bridge(self) -> None:
        """Detach every bridge slot so a window replaced on a UI-language change
        stops receiving events (the shared BusBridge outlives this window)."""
        for signal, slot in self._bridge_bindings():
            signal.disconnect(slot)

    # -- bridge slots (GUI thread) -----------------------------------------

    def _on_mic_level(self, rms: float, vad_prob: float) -> None:
        self._mic_meter.set_level(rms)

    def _translate_active(self) -> bool:
        # Live config AND a live engine: an MT engine can hot-swap in
        # mid-session, so the toggle alone (an engine existed at launch or
        # not) can't decide this. But the toggle alone also can't tell a
        # genuinely loaded engine from one that never finished loading or was
        # swapped out -- marking the row TRANSLATING then would leave it with
        # no event that will ever resolve it (forward_final's else-branch
        # with send_to_vrchat off publishes neither PhraseTranslated nor
        # ChatboxSent), stuck on "translating…" forever.
        return self._store.config.translate.enabled and self._pipeline.mt_active

    def _send_active(self) -> bool:
        return bool(self._store.config.osc.send_to_vrchat)

    def _on_phrase_recognized(self, event) -> None:
        self._caption_model.recognized(
            event.utterance_id,
            event.text,
            translate_enabled=self._translate_active(),
            send_enabled=self._send_active(),
        )
        self._render_log()

    def _on_phrase_translated(self, event) -> None:
        self._caption_model.translated(
            event.utterance_id,
            event.translations,
            send_enabled=self._send_active(),
        )
        self._render_log()

    def _on_chatbox_sent(self, event) -> None:
        self._caption_model.sent(event.utterance_id, getattr(event, "truncated", False))
        self._render_log()

    def _on_mute_changed(self, event) -> None:
        self._set_mute_chip(event.muted)
        # A mute transition moves the pipeline's caption gate, and the capture
        # label folds that gate in; repaint it alongside the chip.
        self._render_capture_status()

    def _on_vrchat_detected(self, event) -> None:
        # event is None only for the initial "checking" render at construction.
        detected = bool(event.detected) if event is not None else None
        status_render.render_vrchat(self, detected)

    def _on_heard_level(self, rms: float, _vad_prob: float) -> None:
        main_heard.on_level(self, rms)

    def _on_hear_others_toggled(self, checked: bool) -> None:
        main_heard.on_toggled(self, checked)

    def _on_heard_phrase(self, event) -> None:
        main_heard.on_phrase(self, event)

    def _on_engine_state(self, event) -> None:
        # State drives the caption feed's loading message via _engine_states; it
        # is no longer shown as jargon text on the main screen.
        self._engine_states[event.engine] = event.state
        # _render_log reads that state to choose the empty-state text, and only
        # a caption event would otherwise redraw it: without this the feed still
        # says the model is loading long after it is ready.
        self._render_log()
        known = event.engine in _ENGINE_NAMES
        name = tr(_ENGINE_NAMES[event.engine]) if known else event.engine.title()
        if event.state == "fallback_cpu":
            # Transient state (immediately followed by "ready"), so surface the
            # CPU drop as a status flash before it's overwritten. Plain name, no
            # jargon; cause-neutral -- fallback_cpu also covers a CUDA provider
            # that failed to initialize, not just VRAM exhaustion.
            self._flash_status(
                tr("{name} could not stay on the GPU. Switched to CPU (slower).", name=name)
            )
        if event.state == "ready":
            # A kind that recovers may fail again later and deserves to be heard
            # a second time.
            self._engine_failures_reported.discard(event.engine)
        if event.state == "failed":
            # Capture is the voice model's business alone. A dead translator
            # leaves transcription running, so claiming "Not listening" would be
            # a lie the user has no way to clear: no later event repaints it.
            if event.engine == "stt":
                self.set_capture_status(False, tr("{name} failed to load", name=name))
            elif known:
                self._flash_status(
                    tr("{name} failed to load. Captions still work.", name=name)
                )
            # One engine can republish "failed" (a retried swap, a second
            # warm-up), and a modal per attempt buries the screen in dialogs.
            if event.engine in self._engine_failures_reported:
                return
            self._engine_failures_reported.add(event.engine)
            if known:
                lower = tr(_ENGINE_NAMES_LOWER[event.engine])
                body = tr("The {name} failed to start.", name=lower)
            else:
                body = tr("A model failed to start.")
            body += "\n\n" + tr("Open Models to re-download it, then restart VRCC.")
            if event.detail:
                body += f"\n\n{event.detail}"
            QMessageBox.warning(self, tr("Model failed to load"), body)

    def _on_download_progress(self, event) -> None:
        if event.done:
            self._flash_status(tr("Download complete: {model_id}", model_id=event.model_id))
            return
        if event.total <= 0:
            return
        pct = int(100 * event.downloaded / event.total)
        self._flash_status(tr("Downloading {model_id}: {pct}%", model_id=event.model_id, pct=pct))

    def _on_app_error(self, event) -> None:
        # All AppErrors are transient status text (5 s); the only modal alert is
        # a failed engine (in _on_engine_state). The raw code+message go to the
        # log so diagnostics are never lost, and never to the status bar: a code
        # with no sentence of its own used to render as "WHAT_IS_THIS: ct2
        # assertion failed at src/layers/attention.cc:88", which tells a user
        # nothing and points them nowhere.
        logger.warning("AppError %s: %s", event.code, event.message)
        friendly = _FRIENDLY_ERRORS.get(event.code, _FRIENDLY_ERRORS["HANDLER_ERROR"])
        self._flash_status(tr(friendly))

    def _on_update_result(self, event) -> None:
        from vrcc.gui import updates_ui
        updates_ui.handle_result(self, event)

    # -- caption log helpers -----------------------------------------------

    def _render_log(self) -> None:
        # Row HTML is handed over per row, not as one document: the follower
        # writes only what changed, and holds the reading position across the
        # model's cap eviction. Both belong to it, so this stays a pure
        # model-to-markup step.
        rows = self._caption_model.rows()
        if rows:
            self._log_follow.set_rows(
                [(row.key, render_row_html(row, self._p, self._scale)) for row in rows]
            )
            return
        msg, sub = empty_state_text(self._engine_states.get("stt"))
        self._log_follow.set_html(empty_state_html(msg, sub, self._p, self._scale))

    # -- mute chip / status rendering --------------------------------------

    def _set_mute_chip(self, muted) -> None:
        status_render.set_mute_chip(self, muted)

    def _flash_status(self, text: str) -> None:
        # showMessage's own timer clears this after _TRANSIENT_MS.
        self.statusBar().showMessage(text, _TRANSIENT_MS)

    def set_capture_status(self, capturing, reason: str = "") -> None:
        """Persistent, honest "is the app capturing?" indicator.

        None = starting up (gray), True = running (green "Listening", amber
        "Paused" if the toggle is off), False = not listening (red) with a
        reason. Called on pipeline start / mic failure / engine failure so a
        failed startup can't look healthy while it hears nothing.
        """
        self._capture_ok = capturing
        self._capture_reason = reason
        self._render_capture_status()

    def _render_capture_status(self) -> None:
        status_render.render_capture_status(self)

    def reload_from_config(self) -> None:
        """Re-sync the toolbar controls to config (e.g. after the modal Settings
        dialog edits shared fields like source language or the profile)."""
        self._loading = True
        try:
            self._load_from_config()
        finally:
            self._loading = False

    # -- user-edit slots (write config, save_soon) -------------------------

    def _on_source_changed(self, text: str) -> None:
        if self._loading:
            return
        self._store.config.stt.source_language = text
        # A target equal to the new source would translate a language into itself
        # (sending the original twice); rebuild drops it (and persists via save_soon).
        self._rebuild_targets()
        model_prompts.run_language_nudge(self)

    def _on_target_slot_changed(self, slot: int, text: str) -> None:
        # The combo can be showing a substitute for what the user asked for, so
        # only a real edit may overwrite the recorded intent.
        if not self._loading and not self._rebuilding:
            self._target_intent[slot] = text
        self._rebuild_targets()

    def _on_target_enabled_changed(self, _checked: bool) -> None:
        self._rebuild_targets()

    def _rebuild_targets(self) -> None:
        main_targets.rebuild_targets(self, _NUM_TARGET_SLOTS)

    def _on_profile_toggled(self, checked: bool) -> None:
        if self._loading:
            return
        # apply_profile writes the full bundle (STT beam, VAD timings) + gui.profile;
        # the engine reads beam size per job, so it takes effect next utterance.
        apply_profile(self._store.config, "quality" if checked else "latency")
        self._store.save_soon()

    def _on_captions_toggled(self, checked: bool) -> None:
        # Pause/resume captioning live: the pipeline and mic keep running, it just
        # stops producing captions. Reflect in the button, meter, and capture status.
        self._captioning_btn.setText(tr("Captioning on") if checked else tr("Start captioning"))
        self._mic_meter.set_active(checked)
        if not self._loading:
            self._pipeline.set_captioning(checked)
        self._render_capture_status()

    def _on_send_clicked(self) -> None:
        text = self._text_input.text()
        if not text.strip():
            # Whitespace only: there is nothing to send, and leaving it in the
            # box makes the press look ignored. Clearing is the answer.
            self._text_input.clear()
            return
        # Only clear the input if the pipeline accepted the message; otherwise
        # (engines still loading / failed) preserve the user's typed text.
        if self._pipeline.submit_typed(text):
            self._text_input.clear()

    # -- menu actions ------------------------------------------------------

    def _check_for_updates(self) -> None:
        if self._on_check_updates is not None:
            self._flash_status(tr("Checking for updates…"))
            self._on_check_updates()

    def _show_about(self) -> None:
        from vrcc.gui import updates_ui

        updates_ui.show_about(self)

    # -- geometry persistence ----------------------------------------------

    def _restore_geometry(self) -> None:
        raw = self._store.config.gui.window_geometry
        if not raw:
            return
        try:
            self.restoreGeometry(QByteArray(base64.b64decode(raw)))
        except Exception:  # noqa: BLE001 -- a corrupt geometry blob must not crash startup
            logger.warning("could not restore window geometry; ignoring", exc_info=True)

    def closeEvent(self, event) -> None:  # noqa: N802 -- Qt override
        try:
            encoded = base64.b64encode(bytes(self.saveGeometry())).decode("ascii")
            self._store.config.gui.window_geometry = encoded
            self._store.save_now()
        except Exception:  # noqa: BLE001 -- never block window close on a save failure
            logger.warning("could not save window geometry", exc_info=True)
        super().closeEvent(event)
