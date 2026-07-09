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

from vrcc import __version__
from vrcc.core.config import ConfigStore, apply_profile
from vrcc.gui.bridge import BusBridge
from vrcc.gui.caption_log import CaptionModel, empty_state_html, render_rows_html
from vrcc.gui.icons import FRIENDLY_ERRORS as _FRIENDLY_ERRORS
from vrcc.gui.icons import dots_svg as _dots_svg  # re-exported: tests import it from here
from vrcc.gui.main_parts import (
    build_caption_log,
    build_compose_row,
    build_status_strip,
    build_top_bar,
)
from vrcc.gui.style import PALETTE, resolve_theme
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
    ) -> None:
        super().__init__()
        self._bridge = bridge
        self._store = config_store
        self._pipeline = pipeline
        # Resolved once at construction (theme + text size are restart-applied).
        self._p = PALETTE[resolve_theme(config_store.config.gui.theme)]
        self._scale = max(0.5, min(2.0, config_store.config.gui.font_scale))
        self._on_open_settings = on_open_settings
        self._on_open_models = on_open_models
        # Kept for caller compat but no longer read: engines hot-swap mid-session,
        # so a launch-time "was MT built?" snapshot would wrongly suppress the
        # "translating…" row. Live config is the only correct source of truth.
        self._mt_available = mt_available

        # Guards config writes while we push config values INTO widgets during
        # construction, so setCurrentText/setValue don't echo back to disk.
        self._loading = True
        # Latest engine states, rendered together in the status bar.
        self._engine_states: dict[str, str] = {}
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
        # Enable the lowest hidden target slot; its checkbox drives the rebuild.
        for slot in range(1, _NUM_TARGET_SLOTS):
            check = self._target_checks[slot]
            if check is not None and not check.isChecked():
                check.setChecked(True)
                break
        self._sync_target_visibility()

    def _remove_target(self, slot: int) -> None:
        check = self._target_checks[slot]
        if check is not None:
            check.setChecked(False)
        self._sync_target_visibility()

    def _sync_target_visibility(self) -> None:
        # Show a slot's pill iff its (hidden) checkbox is on; offer "+ Language"
        # only while a slot is still free.
        any_free = False
        for slot in range(1, _NUM_TARGET_SLOTS):
            check = self._target_checks[slot]
            cont = self._target_conts[slot]
            if check is None or cont is None:
                continue
            cont.setVisible(check.isChecked())
            if not check.isChecked():
                any_free = True
        self._add_target_btn.setVisible(any_free)

    # -- initial values from config ----------------------------------------

    def _load_from_config(self) -> None:
        cfg = self._store.config

        self._set_combo_text(self._source_combo, cfg.stt.source_language)

        targets = list(cfg.translate.targets)
        for slot, combo in enumerate(self._target_combos):
            check = self._target_checks[slot]
            if slot < len(targets):
                self._set_combo_text(combo, targets[slot])
                if check is not None:
                    check.setChecked(True)
            elif check is not None:
                check.setChecked(False)

        self._captioning_btn.setChecked(bool(self._pipeline.captioning_enabled))
        # setChecked only emits toggled on a state change, so sync the meter's
        # active/dimmed state directly, else it stays bright when captioning loads off.
        self._mic_meter.set_active(self._captioning_btn.isChecked())
        self._sync_target_visibility()

    @staticmethod
    def _set_combo_text(combo: QComboBox, text: str) -> None:
        idx = combo.findText(text)
        if idx >= 0:
            combo.setCurrentIndex(idx)

    # -- bridge signal wiring ----------------------------------------------

    def _connect_bridge(self) -> None:
        self._bridge.mic_level.connect(self._on_mic_level)
        self._bridge.phrase_recognized.connect(self._on_phrase_recognized)
        self._bridge.phrase_translated.connect(self._on_phrase_translated)
        self._bridge.chatbox_sent.connect(self._on_chatbox_sent)
        self._bridge.mute_changed.connect(self._on_mute_changed)
        self._bridge.engine_state.connect(self._on_engine_state)
        self._bridge.download_progress.connect(self._on_download_progress)
        self._bridge.app_error.connect(self._on_app_error)
        self._bridge.vrchat_detected.connect(self._on_vrchat_detected)

    # -- bridge slots (GUI thread) -----------------------------------------

    def _on_mic_level(self, rms: float, vad_prob: float) -> None:
        self._mic_meter.set_level(rms)

    def _translate_active(self) -> bool:
        # Live config only: an MT engine can hot-swap in mid-session, so show
        # "translating…" whenever the toggle is on, not just if one existed at launch.
        return self._store.config.translate.enabled

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

    def _on_vrchat_detected(self, event) -> None:
        # event is None only for the initial "checking" render at construction.
        detected = bool(event.detected) if event is not None else None
        tip = tr(
            "Enable OSC in VRChat: Action Menu > Options > OSC > Enabled. "
            "VRChat must be running on this PC."
        )
        if detected is True:
            self._vrchat_label.setText(tr("VRChat: connected"))
            self._vrchat_label.setStyleSheet(f"color: {self._p['good']}; padding: 2px 8px;")
            self._vrchat_label.setToolTip(tr("VRChat's OSC service was found on this network."))
        elif detected is False:
            self._vrchat_label.setText(tr("VRChat: not detected — enable OSC in-game"))
            self._vrchat_label.setStyleSheet(f"color: {self._p['warn']}; padding: 2px 8px;")
            self._vrchat_label.setToolTip(tip)
        else:
            self._vrchat_label.setText(tr("VRChat: checking…"))
            self._vrchat_label.setStyleSheet(f"color: {self._p['muted']}; padding: 2px 8px;")
            self._vrchat_label.setToolTip(tip)

    def _on_engine_state(self, event) -> None:
        # State drives the caption feed's loading message via _engine_states; it
        # is no longer shown as jargon text on the main screen.
        self._engine_states[event.engine] = event.state
        known = event.engine in _ENGINE_NAMES
        name = tr(_ENGINE_NAMES[event.engine]) if known else event.engine.title()
        if event.state == "fallback_cpu":
            # Transient state (immediately followed by "ready"), so surface the
            # CPU drop as a status flash before it's overwritten. Plain name, no jargon.
            self._flash_status(
                tr("{name} ran out of GPU memory — switched to CPU (slower).", name=name)
            )
        if event.state == "failed":
            self.set_capture_status(False, tr("{name} failed to load", name=name))
            lower = tr(_ENGINE_NAMES_LOWER[event.engine]) if known else event.engine.title().lower()
            body = tr("The {name} failed to start.", name=lower)
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
        # All AppErrors are transient status text (5 s); the only modal alert is a
        # failed engine (in _on_engine_state). Known codes show a human sentence;
        # the raw code+message go to the log so diagnostics are never lost.
        friendly = _FRIENDLY_ERRORS.get(event.code)
        if friendly is not None:
            logger.warning("AppError %s: %s", event.code, event.message)
            self._flash_status(tr(friendly))
        else:
            self._flash_status(f"{event.code}: {event.message}")

    # -- caption log helpers -----------------------------------------------

    def _render_log(self) -> None:
        # Full re-render from the row model: caption events are low-frequency and
        # the model is capped, so it's cheap. setHtml resets the scrollbar to the
        # top, so preserve position -- pin to bottom if there, else hold (setValue
        # clamps to the new maximum).
        scrollbar = self._log.verticalScrollBar()
        at_bottom = scrollbar.value() >= scrollbar.maximum() - 2
        previous = scrollbar.value()
        rows = self._caption_model.rows()
        colors = self._p
        if rows:
            self._log.setHtml(render_rows_html(rows, colors, self._scale))
        else:
            loading = self._engine_states.get("stt") in (None, "loading")
            if loading:
                msg, sub = tr("Getting the voice model ready…"), tr("usually takes a few seconds")
            else:
                msg, sub = (
                    tr("Say something — captions appear here"),
                    tr("then in your VRChat chatbox"),
                )
            self._log.setHtml(empty_state_html(msg, sub, colors, self._scale))
        scrollbar.setValue(scrollbar.maximum() if at_bottom else previous)

    # -- mute chip / status rendering --------------------------------------

    def _set_mute_chip(self, muted) -> None:
        # None (mute-sync state unknown yet) hides the chip entirely rather than
        # showing an empty "–" box.
        if muted is None:
            self._mute_chip.setVisible(False)
            return
        if muted:
            self._mute_chip.setText(tr("MUTED"))
            color = self._p["bad"]
        else:
            self._mute_chip.setText(tr("LIVE"))
            color = self._p["good"]
        self._mute_chip.setStyleSheet(
            f"color: {self._p['on_badge']}; background: {color}; padding: 2px 8px;"
        )
        self._mute_chip.setVisible(True)

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
        ok = getattr(self, "_capture_ok", None)
        if ok is None:
            text, color = tr("Starting…"), self._p["muted"]
        elif ok is False:
            reason = getattr(self, "_capture_reason", "")
            if reason:
                text = tr("Not listening — {reason}", reason=reason)
            else:
                text = tr("Not listening")
            color = self._p["bad"]
        elif getattr(self, "_captioning_btn", None) is not None and not self._captioning_btn.isChecked():
            text, color = tr("Paused — not listening"), self._p["warn"]
        else:
            text, color = tr("Listening"), self._p["good"]
        self._capture_label.setText(text)
        self._capture_label.setStyleSheet(f"color: {color}; padding: 2px 8px;")

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

    def _on_targets_changed(self, _text: str) -> None:
        self._rebuild_targets()

    def _on_target_enabled_changed(self, _checked: bool) -> None:
        self._rebuild_targets()

    def _rebuild_targets(self) -> None:
        if self._loading:
            return
        # Dedupe across slots (first-occurrence order) and drop any target equal
        # to the source -- translating a language into itself just re-sends the original.
        source = self._store.config.stt.source_language
        targets: list[str] = []
        for slot, combo in enumerate(self._target_combos):
            check = self._target_checks[slot]
            lang = combo.currentText()
            enabled = check is None or check.isChecked()
            if enabled and lang not in targets and lang != source:
                targets.append(lang)
        self._store.config.translate.targets = targets[:_NUM_TARGET_SLOTS]
        self._store.save_soon()

    def _on_profile_toggled(self, checked: bool) -> None:
        if self._loading:
            return
        # apply_profile writes the full bundle (STT/MT beam, VAD timings) + gui.profile;
        # engines read beam sizes per job, so it takes effect on the next utterance.
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
            return
        # Only clear the input if the pipeline accepted the message; otherwise
        # (engines still loading / failed) preserve the user's typed text.
        if self._pipeline.submit_typed(text):
            self._text_input.clear()

    # -- menu actions ------------------------------------------------------

    def _show_about(self) -> None:
        blurb = tr("Live voice captioning and translation for the VRChat chatbox.")
        credit = tr('Created by <a href="https://github.com/dljr-github">dljr-github</a>')
        QMessageBox.about(
            self,
            tr("About VRCC"),
            f"<p><b>VRCC</b> v{__version__}</p><p>{blurb}</p><p>{credit}</p>",
        )

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
