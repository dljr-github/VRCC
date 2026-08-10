"""Model manager dialog: download and delete the two models VRCC uses.

Two cards (Voice / Translation), each row a friendly name, blurb, and one
contextual action (download / progress / "In use" or "Downloaded" + trash).
Picking which model to use happens in Settings; this only fetches/removes files.
Downloads run one at a time on a background thread; the thread only touches the
manager and emits a Qt Signal on completion -- all widget mutation is GUI-thread.
"""

from __future__ import annotations

import logging
import threading

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from vrcc.core import calibrate, recommend
from vrcc.gui.bridge import BusBridge
from vrcc.i18n import tr
from vrcc.gui import firstrun_languages, model_fit
from vrcc.gui.model_row import ModelRow
from vrcc.gui.model_labels import fmt_size, mt_display_name, whisper_display_name, model_blurb
from vrcc.gui.style import PALETTE, resolve_theme
from vrcc.gui.widgets import Card, icon_label, mic_svg
from vrcc.stt.registry import WHISPER_MODELS
from vrcc.translate.registry import MT_MODELS

logger = logging.getLogger("vrcc.gui.models_dialog")


def _globe_svg(color: str) -> str:
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" '
        f'viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" '
        'stroke-linecap="round" stroke-linejoin="round">'
        '<circle cx="12" cy="12" r="10"/><line x1="2" y1="12" x2="22" y2="12"/>'
        '<path d="M12 2a15 15 0 0 1 0 20a15 15 0 0 1 0-20"/></svg>'
    )


class ModelsDialog(QDialog):
    """Download/delete models. ``download_manager`` may be a real
    :class:`DownloadManager` or a test fake exposing the same methods."""

    _op_finished = Signal(str, bool, str)  # model_id, success, error

    def __init__(
        self, download_manager, bridge: BusBridge, config_store=None, parent=None
    ) -> None:
        super().__init__(parent)
        self._dm = download_manager
        self._bridge = bridge
        # Read for the active model behind the "In use" pill; selecting which
        # model to use still lives in Settings. Written by the language picker
        # below, which sets stt.spoken_languages and through them
        # stt.source_language, so a caller that opens this window has to re-sync
        # whatever else shows those two fields.
        self._store = config_store
        self._downloading_id: str | None = None
        # Resolved once at construction (theme + text size are restart-applied).
        theme = config_store.config.gui.theme if config_store is not None else "dark"
        self._p = PALETTE[resolve_theme(theme)]
        scale = config_store.config.gui.font_scale if config_store is not None else 1.0
        self._scale = max(0.5, min(2.0, scale))
        # Tier resolved once here, following the configured device (a forced-CPU
        # config badges CPU picks even on a GPU machine); the badge tracks this
        # preset, not the active model. Also reranked by spoken_languages, so
        # this agrees with the wizard's recommendation for the same tier and
        # languages instead of only the tier.
        # The machine factor is read, never probed, so opening this dialog
        # stays free; an unprobed config ranks at reference speed.
        self._recommended_ids = self._recommendation()
        self._fit_notes = self._compute_fit_notes()

        self.setWindowTitle(tr("Models"))
        self.resize(660, 620)

        self._rows: list[ModelRow] = []
        self._row_by_id: dict[str, ModelRow] = {}
        self._build_ui()
        self._render_all()

        self._bridge.download_progress.connect(self._on_progress)
        self._op_finished.connect(self._on_op_finished)

    # -- construction --------------------------------------------------------

    def _recommendation(self) -> tuple[str, str]:
        """The preset this machine and these languages should use."""
        cfg = self._store.config if self._store is not None else None
        if cfg is None:
            return recommend.preset_for_tier(recommend.detect_tier())
        return recommend.preset_for_tier(
            recommend.tier_for_config(cfg),
            recommend.spoken_whisper_codes(cfg),
            calibrate.stored_factor(cfg),
            recommend.detected_vram_mb(cfg.stt.device_index),
        )

    def _compute_fit_notes(self) -> dict[str, str]:
        """Graphics-card warnings, resolved once: neither the card nor the
        configured device can change while this window is open, and a
        per-render probe would cost an NVML lookup per row on every tick."""
        if self._store is None:
            return {}
        return model_fit.fit_notes(self._store.config)

    def _on_spoken_changed(self) -> None:
        """Re-badge for the languages just ticked, and persist them.

        Order matches the wizard's: the recommendation depends on the
        languages, and the derived source language depends on the model.

        The model handed to resolve_source_language is the ACTIVE one, not the
        recommended one. This window only badges a recommendation; it never
        installs it, so judging what the source language may become against a
        model that is not running would set a source the running model cannot
        serve. The wizard passes its recommendation because it writes it to
        cfg.stt.model in the same breath.
        """
        cfg = self._store.config
        cfg.stt.spoken_languages = firstrun_languages.checked_in(self._spoken_list)
        self._recommended_ids = self._recommendation()
        firstrun_languages.resolve_source_language(
            cfg, cfg.stt.spoken_languages, cfg.stt.model, cfg.translate.enabled,
        )
        self._render_all()
        self._store.save_soon()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)

        title = QLabel(tr("Models"))
        title.setStyleSheet(
            f"font-weight: 700; font-size: {round(16 * self._scale)}px; "
            f"color: {self._p['text']};"
        )
        root.addWidget(title)
        self._title = title  # test seam: scaled-title assertion

        lead = QLabel(
            tr(
                "VRCC uses two models: one to hear your speech and one to translate "
                "it. Download the ones you want here, then choose which to use "
                "in Settings."
            )
        )
        lead.setWordWrap(True)
        lead.setStyleSheet(f"color: {self._p['muted']}; background: transparent;")
        root.addWidget(lead)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        container = QWidget()
        col = QVBoxLayout(container)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(12)

        # Which voice model suits a user depends on what they speak, and the
        # wizard is the only place that ever asked. Someone who skipped it, or
        # who starts speaking another language, would otherwise be stuck on the
        # language-blind pick with nowhere to correct it.
        if self._store is not None:
            heading = QLabel(tr("Pick your languages"))
            heading.setStyleSheet(
                f"font-weight: 700; font-size: {round(14 * self._scale)}px; "
                f"color: {self._p['text']}; background: transparent;"
            )
            col.addWidget(heading)
            spoken_label = QLabel(tr("You speak (tick every language you use)"))
            spoken_label.setStyleSheet(
                f"color: {self._p['muted']}; background: transparent;"
            )
            col.addWidget(spoken_label)
            self._spoken_list = firstrun_languages.build_picker(
                self._scale, self._store.config, self._on_spoken_changed
            )
            col.addWidget(self._spoken_list)
            # Unticking everything is allowed, and nothing on screen said what
            # VRCC does then, so an empty list read as "no language at all".
            # Text set in _render_spoken_hint from the source that ends up in
            # force, not from a prediction of it.
            self._spoken_hint = QLabel()
            self._spoken_hint.setWordWrap(True)
            self._spoken_hint.setStyleSheet(
                f"color: {self._p['warn']}; background: transparent;"
            )
            col.addWidget(self._spoken_hint)

        col.addWidget(
            self._build_card(
                mic_svg(self._p["accent"]),
                tr("Voice model"),
                tr("Recognizes what you say and turns it into text."),
                self._voice_rows(),
            )
        )
        col.addWidget(
            self._build_card(
                _globe_svg(self._p["accent"]),
                tr("Translation model"),
                tr("Translates your speech into the languages you chose."),
                self._translation_rows(),
            )
        )
        col.addStretch(1)
        scroll.setWidget(container)
        root.addWidget(scroll, 1)

        footer = QHBoxLayout()
        # Says why Close and the row buttons are off during a download. Without
        # it the window refused to close and disabled everything with no
        # explanation on screen, which reads as a hang.
        self._status = QLabel()
        self._status.setWordWrap(True)
        self._status.setStyleSheet(
            f"color: {self._p['muted']}; background: transparent;"
        )
        self._status.setVisible(False)
        footer.addWidget(self._status, 1)
        self._close_btn = QPushButton(tr("Close"))
        self._close_btn.clicked.connect(self.reject)
        footer.addWidget(self._close_btn)
        root.addLayout(footer)

    def _build_card(self, icon_svg: str, header: str, role: str, rows: list[ModelRow]) -> Card:
        card = Card(colors=self._p)
        head_row = QHBoxLayout()
        head_row.setSpacing(6)
        head_row.addWidget(icon_label(icon_svg, colors=self._p, fallback_text="*"))
        title = QLabel(header)
        title.setStyleSheet(
            f"font-weight: 700; font-size: {round(15 * self._scale)}px; "
            "background: transparent;"
        )
        head_row.addWidget(title)
        head_row.addStretch(1)
        card.body.addLayout(head_row)
        role_lbl = QLabel(role)
        role_lbl.setWordWrap(True)
        role_lbl.setStyleSheet(f"color: {self._p['muted']}; background: transparent;")
        card.body.addWidget(role_lbl)
        for row in rows:
            card.body.addWidget(row)
        return card

    def _voice_rows(self) -> list[ModelRow]:
        rows = []
        for spec in WHISPER_MODELS.values():
            row = self._make_row(
                "whisper", spec.id, spec,
                whisper_display_name(spec.id), model_blurb("whisper", spec.id), fmt_size(spec.size_mb),
            )
            rows.append(row)
        return rows

    def _translation_rows(self) -> list[ModelRow]:
        rows = []
        for spec in MT_MODELS.values():
            row = self._make_row(
                "mt", spec.id, spec,
                mt_display_name(spec.id), model_blurb("mt", spec.id), fmt_size(spec.size_mb),
            )
            rows.append(row)
        return rows

    def _make_row(self, kind, model_id, spec, name, blurb, size_text) -> ModelRow:
        row = ModelRow(
            kind, model_id, spec, name, blurb, size_text, self._p,
            self._download, self._delete, scale=self._scale,
        )
        self._rows.append(row)
        self._row_by_id[model_id] = row
        return row

    # -- state ---------------------------------------------------------------

    def _is_downloaded(self, row: ModelRow) -> bool:
        if row.kind == "whisper":
            return self._dm.is_whisper_downloaded(row.model_id)
        return self._dm.is_mt_downloaded(row.spec)

    def _is_active(self, row: ModelRow) -> bool:
        """Whether config currently points the app at this model."""
        if self._store is None:
            return False
        cfg = self._store.config
        if row.kind == "whisper":
            return row.model_id == cfg.stt.model
        return row.model_id == cfg.translate.model

    def _is_recommended(self, row: ModelRow) -> bool:
        whisper_id, mt_id = self._recommended_ids
        return row.model_id == (whisper_id if row.kind == "whisper" else mt_id)

    def _row_note(self, row: ModelRow) -> str:
        """What this row warns about before the user spends a download."""
        if self._store is None:
            return ""
        return model_fit.row_note(
            self._store.config, row.kind, row.model_id, row.display_name,
            self._fit_notes,
        )

    def _render_all(self) -> None:
        """Re-render every row and apply the download-in-flight action guard."""
        downloading = self._downloading_id
        for row in self._rows:
            row.render(
                downloaded=self._is_downloaded(row),
                active=self._is_active(row),
                downloading=row.model_id == downloading,
                recommended=self._is_recommended(row),
            )
            row.set_note(self._row_note(row))
            # While any download runs, every row's actions are disabled; the
            # running row shows only its progress bar.
            row.set_actions_enabled(downloading is None)
        if self._store is not None:
            self._render_spoken_hint()
        self._render_status(downloading)

    def _render_spoken_hint(self) -> None:
        """With nothing ticked, name the language VRCC is left listening for.

        Read off the config after resolve_source_language has had its say
        rather than restating that function's rule, so this cannot claim a
        source the engines are not running.
        """
        empty = not firstrun_languages.checked_in(self._spoken_list)
        if empty:
            source = self._store.config.stt.source_language
            self._spoken_hint.setText(
                tr("No language ticked, so VRCC tries to detect what it hears.")
                if source == "auto"
                else tr(
                    "No language ticked, so VRCC keeps listening for {language}.",
                    language=source,
                )
            )
        self._spoken_hint.setVisible(empty)

    def _render_status(self, downloading: str | None) -> None:
        """Explain the download lock, or clear it."""
        row = self._row_by_id.get(downloading) if downloading else None
        if row is None:
            self._status.setVisible(False)
        else:
            self._status.setText(
                tr(
                    "Downloading {name}. A big model can take several minutes. "
                    "A download cannot be stopped once it starts, so Close and "
                    "the other buttons wait until this one finishes.",
                    name=row.display_name,
                )
            )
            self._status.setVisible(True)
        self._close_btn.setEnabled(row is None)

    # -- download ------------------------------------------------------------

    def _download(self, row: ModelRow) -> None:
        if self._downloading_id is not None or self._is_downloaded(row):
            return
        msg = model_fit.disk_warning(getattr(self._dm, "models_dir", None), row.spec.size_mb)
        if msg:
            answer = QMessageBox.question(
                self, tr("Low disk space"), msg + "\n\n" + tr("Download anyway?"),
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
        self._start_download(row)

    def _start_download(self, row: ModelRow) -> None:
        self._downloading_id = row.model_id
        # Only faster-whisper downloads lack byte progress; MT and onnx-asr
        # snapshots publish real percentages.
        row.begin_progress(
            indeterminate=row.kind == "whisper"
            and getattr(row.spec, "backend", "whisper") == "whisper"
        )
        self._render_all()

        def worker() -> None:
            error = ""
            success = True
            try:
                if row.kind == "whisper":
                    self._dm.ensure_whisper(row.model_id)
                else:
                    self._dm.ensure_mt(row.spec)
            except Exception as exc:  # noqa: BLE001 -- surfaced via the signal
                success = False
                error = str(exc)
                logger.exception("download failed for %s", row.model_id)
            self._op_finished.emit(row.model_id, success, error)

        threading.Thread(target=worker, name=f"Download-{row.model_id}", daemon=True).start()

    def _on_progress(self, event) -> None:
        if event.model_id != self._downloading_id:
            return
        row = self._row_by_id.get(event.model_id)
        if row is None:
            return
        if event.done:
            row.set_progress_value(100)
            return
        if event.total > 0:
            row.set_progress_value(int(100 * event.downloaded / event.total))

    def _on_op_finished(self, model_id: str, success: bool, error: str) -> None:
        self._downloading_id = None
        row = self._row_by_id.get(model_id)
        if row is not None:
            row.reset_progress()
        self._render_all()
        if not success:
            name = row.display_name if row is not None else model_id
            QMessageBox.warning(
                self,
                tr("Download failed"),
                tr("Could not download {name}:\n\n{error}", name=name, error=error),
            )

    # -- delete --------------------------------------------------------------

    def _delete(self, row: ModelRow) -> None:
        if self._downloading_id is not None or not self._is_downloaded(row):
            return
        warning = ""
        if self._is_active(row):
            warning = "\n\n" + tr(
                "This is the model VRCC is currently using. Captions "
                "stop until you choose another in Settings."
            )
        body = tr("Delete the downloaded files for {name}?", name=row.display_name) + warning
        reply = QMessageBox.question(
            self, tr("Delete model"), body,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        try:
            self._dm.delete(row.kind, row.model_id)
        except Exception as exc:  # noqa: BLE001
            logger.exception("delete failed for %s", row.model_id)
            QMessageBox.warning(
                self,
                tr("Delete failed"),
                tr("Could not delete {name}:\n\n{error}", name=row.display_name, error=exc),
            )
        self._render_all()

    # -- lifecycle guards ----------------------------------------------------
    # Refuses to close while `_downloading_id` is set (Close/Esc/titlebar X all
    # route through reject()/closeEvent), else the daemon download thread could
    # outlive the dialog and its completion emit would hit a deleted QObject.
    # Neither downloader (snapshot_download, faster-whisper's download_model)
    # takes a cancel token, so there is nothing to offer but the wait; the
    # footer status says so while Esc and the X keep being swallowed here.

    def reject(self) -> None:  # noqa: N802 -- Qt override
        if self._downloading_id is not None:
            logger.info("ignoring close request while %s is downloading", self._downloading_id)
            return
        super().reject()

    def closeEvent(self, event) -> None:  # noqa: N802 -- Qt override
        if self._downloading_id is not None:
            logger.info("ignoring close request while %s is downloading", self._downloading_id)
            event.ignore()
            return
        super().closeEvent(event)
