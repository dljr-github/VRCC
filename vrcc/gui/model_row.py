"""One model's row inside the Models dialog.

Split out of :mod:`vrcc.gui.models_dialog` for the 500-line cap. The dialog owns
all the state; this widget only reflects what it is told in :meth:`ModelRow.render`.
"""

from __future__ import annotations

from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from vrcc.gui.widgets import IconButton
from vrcc.i18n import tr


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


class ModelRow(QWidget):
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
        self._badge = QLabel(tr("Recommended for your PC"))
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

        # Warnings that depend on this machine and this config, so they cannot
        # live in the static blurb: the model may not fit the graphics card, or
        # it may not be able to write a language the captions are set to.
        self._note = QLabel()
        self._note.setWordWrap(True)
        self._note.setStyleSheet(
            f"color: {colors['warn']}; font-size: {round(12 * scale)}px; "
            "background: transparent;"
        )
        self._note.setVisible(False)
        left.addWidget(self._note)
        row.addLayout(left, 1)

        # -- right: exactly one of these shows at a time ----------------------
        self._progress = QProgressBar()
        self._progress.setFixedWidth(170)
        self._progress.setTextVisible(True)
        self._progress.setVisible(False)
        row.addWidget(self._progress)

        self._download_btn = QPushButton(tr("Download · {size}", size=size_text))
        self._download_btn.setToolTip(tr("Download this model"))
        self._download_btn.clicked.connect(lambda: self._on_download(self))
        self._download_btn.setVisible(False)
        row.addWidget(self._download_btn)

        self._inuse_pill = QLabel(tr("In use"))
        self._inuse_pill.setStyleSheet(f"color: {colors['good']}; font-weight: 600; background: transparent;")
        self._inuse_pill.setVisible(False)
        row.addWidget(self._inuse_pill)

        # Shown for a downloaded, non-active model: read-only, no action.
        self._downloaded_pill = QLabel(tr("Downloaded"))
        self._downloaded_pill.setStyleSheet(f"color: {colors['muted']}; background: transparent;")
        self._downloaded_pill.setVisible(False)
        row.addWidget(self._downloaded_pill)

        self._trash_btn = IconButton(
            _trash_svg(colors["muted"]), tr("Delete download"), fallback_text="Del"
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

    def set_note(self, text: str) -> None:
        """Warning line under the blurb; ``""`` hides it."""
        self._note.setText(text)
        self._note.setVisible(bool(text))

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
            self._progress.setFormat(tr("Downloading…"))
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
