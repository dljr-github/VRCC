"""The frameless panel shown before the slow imports, so a launch reads as
progress rather than a hang.

Built as a splash, not a dialog: it carries no buttons, cannot be dismissed
by the user, and must not pull focus away from whatever else is coming up
behind it. The bar is stepped once per phase rather than swept, because a
sweep freezes mid-frame while a blocking import runs, which looks exactly
like the hang this panel exists to hide.

Must be constructed after apply_ui_language, apply_theme_guarded and
apply_font_scale. tr() returns English verbatim until the catalog loads, and
this codebase has no retranslate path, so a label resolved a moment early
stays English for the life of the process.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFrame, QLabel, QProgressBar, QVBoxLayout, QWidget

from vrcc.core.progress import PHASES, phase_label
from vrcc.gui.style import PALETTE, resolve_theme

# The main window carries the same literal, untranslated for the same reason:
# a product name is not a phrase that needs a catalog entry in every language.
_TITLE = "VRCC"


class BootPanel(QWidget):
    """A determinate, five-step splash. ``start(key)`` is the only thing a
    caller needs during the import walk; ``close_panel()`` (or ``close()``,
    which only forwards to it) is the only thing it needs once the real
    window is ready to take over."""

    def __init__(self, theme: str = "dark", parent=None) -> None:
        super().__init__(
            parent, Qt.WindowType.FramelessWindowHint | Qt.WindowType.SplashScreen
        )
        p = PALETTE[resolve_theme(theme)]
        self.setWindowTitle(_TITLE)
        self.setFixedWidth(340)
        self.setStyleSheet(f"background: {p['ground']};")

        outer = QVBoxLayout(self)
        outer.setContentsMargins(18, 18, 18, 18)

        # A bare QFrame rather than vrcc.gui.widgets.Card: Card pulls in
        # QtSvg through vrcc.gui.icons, and this module's whole reason to
        # exist is to appear before that weight is paid.
        card = QFrame(self)
        card.setObjectName("BootCard")
        card.setStyleSheet(
            f"#BootCard {{ background: {p['surface']}; "
            f"border: 1px solid {p['border']}; border-radius: 12px; }}"
        )
        body = QVBoxLayout(card)
        body.setContentsMargins(14, 12, 14, 12)
        body.setSpacing(8)
        outer.addWidget(card)

        heading = QLabel(_TITLE)
        heading.setStyleSheet(f"color: {p['text']}; font-weight: 600; background: transparent;")
        body.addWidget(heading)

        self._label = QLabel("")
        self._label.setWordWrap(True)
        self._label.setStyleSheet(f"color: {p['muted']}; background: transparent;")
        body.addWidget(self._label)

        self._bar = QProgressBar()
        self._bar.setRange(0, len(PHASES))
        self._bar.setValue(0)
        # No format string: a translated "%p%" would have to survive all 17
        # catalogs untouched, for a number nobody reads on a five-step bar.
        self._bar.setTextVisible(False)
        self._bar.setStyleSheet(
            f"QProgressBar {{ background: {p['surface_2']}; "
            f"border: 1px solid {p['border']}; border-radius: 6px; }} "
            f"QProgressBar::chunk {{ background: {p['accent']}; border-radius: 6px; }}"
        )
        body.addWidget(self._bar)

        self._centre_on_screen()

    def _centre_on_screen(self) -> None:
        """Move the panel to the middle of its screen's usable area.

        QSplashScreen centres itself inside its own constructor; a bare
        QWidget carrying Qt.WindowType.SplashScreen does not, so this does
        it by hand. availableGeometry() excludes the taskbar, so the panel
        does not land partly behind it the way the full screen geometry
        would allow. A widget whose screen cannot be resolved is left
        wherever the platform put it rather than moved: a boot panel that
        cannot work out where it is must still appear.
        """
        screen = self.screen()
        if screen is None:
            return
        self.adjustSize()
        geo = self.frameGeometry()
        geo.moveCenter(screen.availableGeometry().center())
        self.move(geo.topLeft())

    def start(self, key: str) -> None:
        """Name the phase starting now and step the bar toward it.

        A key outside the table still steps the bar (a caller must never be
        refused for naming a phase this panel does not know) and clamps at
        the maximum, since a stray extra call is not a sixth phase.
        """
        self._label.setText(phase_label(key))
        self._bar.setValue(min(self._bar.value() + 1, self._bar.maximum()))

    def close_panel(self) -> None:
        """Hide the panel. Safe to call twice: the caller that built this
        may close it from more than one place depending on how launch went."""
        self.hide()

    def close(self) -> None:
        # QWidget.close() already has its own meaning; this only exists so a
        # caller holding either a BootPanel or a LogProgress reporter can
        # call close() without knowing which one it has.
        self.close_panel()
