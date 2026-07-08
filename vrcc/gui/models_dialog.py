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
    QProgressBar,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from vrcc.core import recommend
from vrcc.gui.bridge import BusBridge
from vrcc.gui import model_fit
from vrcc.gui.model_labels import fmt_size, mt_display_name, whisper_display_name, model_blurb
from vrcc.gui.style import PALETTE, resolve_theme
from vrcc.gui.widgets import Card, IconButton, icon_label, mic_svg
from vrcc.stt.registry import WHISPER_MODELS
from vrcc.translate.registry import MT_MODELS

logger = logging.getLogger("vrcc.gui.models_dialog")


def _trash_svg(color: str) -> str:
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" '
        f'viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" '
        'stroke-linecap="round" stroke-linejoin="round">'
        '<polyline points="3 6 5 6 21 6"/>'
        '<path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6"/>'
        '<path d="M8 6V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/>'
        '<line x1="10" y1="11" x2="10" y2="17"/>'
        '<line x1="14" y1="11" x2="14" y2="17"/></svg>'
    )


def _globe_svg(color: str) -> str:
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" '
        f'viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" '
        'stroke-linecap="round" stroke-linejoin="round">'
        '<circle cx="12" cy="12" r="10"/><line x1="2" y1="12" x2="22" y2="12"/>'
        '<path d="M12 2a15 15 0 0 1 0 20a15 15 0 0 1 0-20"/></svg>'
    )


class _ModelRow(QWidget):
    """One model's row: name (+ recommended badge), blurb, and a single
    contextual action area with its own progress bar. The dialog owns the
    state; :meth:`render` just reflects it. ``kind`` is ``"whisper"`` or
    ``"mt"``; ``model_id`` matches ``DownloadProgress`` events.
    """

    def __init__(
        self, kind: str, model_id: str, spec, name: str, blurb: str, size_text: str,
        colors: dict, on_download, on_delete, parent=None, scale: float = 1.0,
    ) -> None:
        super().__init__(parent)
        self.kind = kind
        self.model_id = model_id
        self.spec = spec
        self.display_name = name
        self._on_download = on_download
        self._on_delete = on_delete

        row = QHBoxLayout(self)
        row.setContentsMargins(0, 6, 0, 6)
        row.setSpacing(10)

        # -- left: name (+ recommended badge) and the muted descriptor --------
        left = QVBoxLayout()
        left.setSpacing(2)
        name_row = QHBoxLayout()
        name_row.setSpacing(6)
        name_lbl = QLabel(name)
        name_lbl.setStyleSheet("font-weight: 600; background: transparent;")
        name_row.addWidget(name_lbl)
        self._badge = QLabel("Recommended for your PC")
        accent = colors["accent"]
        self._badge.setStyleSheet(
            f"color: {accent}; border: 1px solid {accent}; border-radius: 8px; "
            f"padding: 0 6px; font-size: {round(10 * scale)}px; background: transparent;")
        self._badge.setVisible(False)
        name_row.addWidget(self._badge)
        name_row.addStretch(1)
        left.addLayout(name_row)
        if blurb:
            blurb_lbl = QLabel(blurb)
            blurb_lbl.setWordWrap(True)
            blurb_lbl.setStyleSheet(
                f"color: {colors['muted']}; font-size: {round(12 * scale)}px; "
                "background: transparent;"
            )
            left.addWidget(blurb_lbl)
        row.addLayout(left, 1)

        # -- right: exactly one of these shows at a time ----------------------
        self._progress = QProgressBar()
        self._progress.setFixedWidth(170)
        self._progress.setTextVisible(True)
        self._progress.setVisible(False)
        row.addWidget(self._progress)

        self._download_btn = QPushButton(f"Download · {size_text}")
        self._download_btn.setToolTip("Download this model")
        self._download_btn.clicked.connect(lambda: self._on_download(self))
        self._download_btn.setVisible(False)
        row.addWidget(self._download_btn)

        self._inuse_pill = QLabel("In use")
        self._inuse_pill.setStyleSheet(f"color: {colors['good']}; font-weight: 600; background: transparent;")
        self._inuse_pill.setVisible(False)
        row.addWidget(self._inuse_pill)

        # Shown for a downloaded, non-active model: read-only, no action.
        self._downloaded_pill = QLabel("Downloaded")
        self._downloaded_pill.setStyleSheet(f"color: {colors['muted']}; background: transparent;")
        self._downloaded_pill.setVisible(False)
        row.addWidget(self._downloaded_pill)

        self._trash_btn = IconButton(
            _trash_svg(colors["muted"]), "Delete download", fallback_text="Del"
        )
        self._trash_btn.setFixedSize(30, 30)
        self._trash_btn.clicked.connect(lambda: self._on_delete(self))
        self._trash_btn.setVisible(False)
        row.addWidget(self._trash_btn)

    # -- rendering -----------------------------------------------------------

    def render(
        self, *, downloaded: bool, active: bool, downloading: bool, recommended: bool = False
    ) -> None:
        """Show exactly one contextual action. ``recommended`` (the tier
        preset) drives the badge; ``active`` drives the "In use" pill -- the
        two are independent (a model can be active without being the preset)."""
        self._badge.setVisible(recommended)
        self._progress.setVisible(downloading)
        self._download_btn.setVisible(not downloading and not downloaded)
        self._inuse_pill.setVisible(not downloading and downloaded and active)
        self._downloaded_pill.setVisible(not downloading and downloaded and not active)
        self._trash_btn.setVisible(not downloading and downloaded)

    def set_actions_enabled(self, enabled: bool) -> None:
        """Enable/disable the interactive buttons (the download-in-flight guard)."""
        self._download_btn.setEnabled(enabled)
        self._trash_btn.setEnabled(enabled)

    # -- progress ------------------------------------------------------------

    def begin_progress(self, *, indeterminate: bool) -> None:
        if indeterminate:
            # faster-whisper's download exposes no byte-progress hook, so a
            # %-bar would sit frozen at 0%. Use a busy/indeterminate bar.
            self._progress.setRange(0, 0)
            self._progress.setFormat("Downloading…")
        else:
            self._progress.setRange(0, 100)
            self._progress.setValue(0)
            self._progress.setFormat("%p%")

    def set_progress_value(self, pct: int) -> None:
        if self._progress.maximum() > 0:  # ignore for the busy/indeterminate bar
            self._progress.setValue(pct)

    def reset_progress(self) -> None:
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        self._progress.setFormat("%p%")


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
        # The config store is read-only here (the currently-active model, for the
        # "In use" pill); selecting which model to use lives in Settings.
        self._store = config_store
        self._downloading_id: str | None = None
        # Resolved once at construction (theme + text size are restart-applied).
        theme = config_store.config.gui.theme if config_store is not None else "dark"
        self._p = PALETTE[resolve_theme(theme)]
        scale = config_store.config.gui.font_scale if config_store is not None else 1.0
        self._scale = max(0.5, min(2.0, scale))
        # Tier resolved once here, following the configured device (a forced-CPU
        # config badges CPU picks even on a GPU machine); the badge tracks this
        # preset, not the active model.
        tier = (
            recommend.tier_for_config(config_store.config)
            if config_store is not None
            else recommend.detect_tier()
        )
        self._recommended_ids = recommend.PRESETS[tier]

        self.setWindowTitle("Models")
        self.resize(660, 620)

        self._rows: list[_ModelRow] = []
        self._row_by_id: dict[str, _ModelRow] = {}
        self._build_ui()
        self._render_all()

        self._bridge.download_progress.connect(self._on_progress)
        self._op_finished.connect(self._on_op_finished)

    # -- construction --------------------------------------------------------

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)

        title = QLabel("Models")
        title.setStyleSheet(
            f"font-weight: 700; font-size: {round(16 * self._scale)}px; "
            f"color: {self._p['text']};"
        )
        root.addWidget(title)
        self._title = title  # test seam: scaled-title assertion

        lead = QLabel(
            "VRCC uses two models: one to hear your speech and one to translate "
            "it. Download the ones you want here — choose which to use in "
            "Settings."
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

        col.addWidget(
            self._build_card(
                mic_svg(self._p["accent"]),
                "Voice model",
                "Recognizes what you say and turns it into text.",
                self._voice_rows(),
            )
        )
        col.addWidget(
            self._build_card(
                _globe_svg(self._p["accent"]),
                "Translation model",
                "Translates your speech into the languages you chose.",
                self._translation_rows(),
            )
        )
        col.addStretch(1)
        scroll.setWidget(container)
        root.addWidget(scroll, 1)

        footer = QHBoxLayout()
        footer.addStretch(1)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.reject)
        footer.addWidget(close_btn)
        root.addLayout(footer)

    def _build_card(self, icon_svg: str, header: str, role: str, rows: list[_ModelRow]) -> Card:
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

    def _voice_rows(self) -> list[_ModelRow]:
        rows = []
        for spec in WHISPER_MODELS.values():
            row = self._make_row(
                "whisper", spec.id, spec,
                whisper_display_name(spec.id), model_blurb("whisper", spec.id), fmt_size(spec.size_mb),
            )
            rows.append(row)
        return rows

    def _translation_rows(self) -> list[_ModelRow]:
        rows = []
        for spec in MT_MODELS.values():
            row = self._make_row(
                "mt", spec.id, spec,
                mt_display_name(spec.id), model_blurb("mt", spec.id), fmt_size(spec.size_mb),
            )
            rows.append(row)
        return rows

    def _make_row(self, kind, model_id, spec, name, blurb, size_text) -> _ModelRow:
        row = _ModelRow(
            kind, model_id, spec, name, blurb, size_text, self._p,
            self._download, self._delete, scale=self._scale,
        )
        self._rows.append(row)
        self._row_by_id[model_id] = row
        return row

    # -- state ---------------------------------------------------------------

    def _is_downloaded(self, row: _ModelRow) -> bool:
        if row.kind == "whisper":
            return self._dm.is_whisper_downloaded(row.model_id)
        return self._dm.is_mt_downloaded(row.spec)

    def _is_active(self, row: _ModelRow) -> bool:
        """Whether config currently points the app at this model."""
        if self._store is None:
            return False
        cfg = self._store.config
        if row.kind == "whisper":
            return row.model_id == cfg.stt.model
        return row.model_id == cfg.translate.model

    def _is_recommended(self, row: _ModelRow) -> bool:
        whisper_id, mt_id = self._recommended_ids
        return row.model_id == (whisper_id if row.kind == "whisper" else mt_id)

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
            # While any download runs, every row's actions are disabled; the
            # running row shows only its progress bar.
            row.set_actions_enabled(downloading is None)

    # -- download ------------------------------------------------------------

    def _download(self, row: _ModelRow) -> None:
        if self._downloading_id is not None or self._is_downloaded(row):
            return
        msg = model_fit.disk_warning(getattr(self._dm, "models_dir", None), row.spec.size_mb)
        if msg:
            answer = QMessageBox.question(
                self, "Low disk space", msg + "\n\nDownload anyway?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
        self._start_download(row)

    def _start_download(self, row: _ModelRow) -> None:
        self._downloading_id = row.model_id
        row.begin_progress(indeterminate=row.kind == "whisper")
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
            QMessageBox.warning(self, "Download failed", f"Could not download {name}:\n\n{error}")

    # -- delete --------------------------------------------------------------

    def _delete(self, row: _ModelRow) -> None:
        if self._downloading_id is not None or not self._is_downloaded(row):
            return
        warning = ""
        if self._is_active(row):
            warning = (
                "\n\nThis is the model VRCC is currently using — captions "
                "stop until you choose another in Settings."
            )
        reply = QMessageBox.question(
            self, "Delete model", f"Delete the downloaded files for {row.display_name}?{warning}",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        try:
            self._dm.delete(row.kind, row.model_id)
        except Exception as exc:  # noqa: BLE001
            logger.exception("delete failed for %s", row.model_id)
            QMessageBox.warning(self, "Delete failed", f"Could not delete {row.display_name}:\n\n{exc}")
        self._render_all()

    # -- lifecycle guards ----------------------------------------------------
    # Refuses to close while `_downloading_id` is set (Close/Esc/titlebar X all
    # route through reject()/closeEvent), else the daemon download thread could
    # outlive the dialog and its completion emit would hit a deleted QObject.

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
