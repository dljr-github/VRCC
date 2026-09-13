"""The setup check panel: draws the six rows `setup_steps.evaluate` scores.

A separate top level, not a child of the main window's central column. That
column's height and width are pinned by tests/test_caption_feed.py and
tests/test_main_window_ui.py, so grafting rows onto it would move numbers
those tests own. This widget holds no `SetupFacts` and runs none of the row
logic itself; a later controller owns the facts and calls `apply()` with
whatever `evaluate()` returns whenever the bus reports something new.

Built as a tool window, not a dialog: `Qt.WindowType.Tool` keeps it off the
taskbar and floating above the app it reports on, and
`WA_ShowWithoutActivating` means showing it never steals focus from whatever
the user is doing in VRChat or the main window.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QHBoxLayout, QLabel, QVBoxLayout, QWidget

from vrcc.gui.icons import alert_svg, circle_svg, tick_svg
from vrcc.gui.setup_steps import ROWS, row_text
from vrcc.gui.style import PALETTE, resolve_theme
from vrcc.gui.widgets import Card, svg_pixmap

# The product name, not a phrase that needs a catalog entry in every
# language: the same choice boot_panel.py makes for the same reason.
_TITLE = "VRCC"

_ICON_SIZE = 20

# Breathing room between the panel and the window it sits beside, applied
# the same way on both sides so a fallback to the left never reads as a
# different placement rule than the default to the right.
_GAP = 8


class SetupPanel(QWidget):
    """Six rows, each showing `row_text`'s headline and detail for whatever
    state `apply()` last gave it. `close_panel()` is the teardown callers
    use; `QWidget.close()` is left alone, since nothing here holds this
    panel and a BootPanel interchangeably (boot.py's reporter does, which is
    why BootPanel carries a close() override and this does not)."""

    def __init__(self, theme: str = "dark", parent=None) -> None:
        super().__init__(parent, Qt.WindowType.Tool)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        p = PALETTE[resolve_theme(theme)]
        self._colors = p
        self.setWindowTitle(_TITLE)
        self.setFixedWidth(320)
        self.setStyleSheet(f"background: {p['ground']};")

        outer = QVBoxLayout(self)
        outer.setContentsMargins(12, 12, 12, 12)

        card = Card(self, colors=p)
        outer.addWidget(card)

        self._icon_labels: dict[str, QLabel] = {}
        self._headline_labels: dict[str, QLabel] = {}
        self._detail_labels: dict[str, QLabel] = {}

        for row_id in ROWS:
            card.body.addLayout(self._build_row(row_id, p))

        # apply() has not been called yet at this point; every row starts
        # pending rather than blank, so the panel never shows empty labels.
        self.apply({row_id: "pending" for row_id in ROWS})

    def _build_row(self, row_id: str, p: dict) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(10)

        icon = QLabel()
        icon.setFixedSize(_ICON_SIZE, _ICON_SIZE)
        row.addWidget(icon)
        self._icon_labels[row_id] = icon

        text = QVBoxLayout()
        text.setSpacing(2)
        headline = QLabel()
        headline.setStyleSheet(
            f"color: {p['text']}; font-weight: 600; background: transparent;"
        )
        detail = QLabel()
        detail.setWordWrap(True)
        detail.setStyleSheet(f"color: {p['muted']}; background: transparent;")
        text.addWidget(headline)
        text.addWidget(detail)
        row.addLayout(text, 1)

        self._headline_labels[row_id] = headline
        self._detail_labels[row_id] = detail
        return row

    def apply(self, states: dict[str, str]) -> None:
        """Redraw every row from `states`, as returned by
        `setup_steps.evaluate`. A row id missing from `states` reads as
        "pending" rather than raising: a caller handed a partial dict must
        still get a panel that draws, not a crash."""
        p = self._colors
        for row_id in ROWS:
            state = states.get(row_id, "pending")
            headline, detail = row_text(row_id, state)
            self._headline_labels[row_id].setText(headline)
            self._detail_labels[row_id].setText(detail)
            pixmap = svg_pixmap(self._icon_svg(state, p), _ICON_SIZE)
            icon_label = self._icon_labels[row_id]
            if pixmap is not None:
                icon_label.setPixmap(pixmap)
            else:
                icon_label.clear()

    def _icon_svg(self, state: str, p: dict) -> str:
        if state == "pass":
            return tick_svg(p["good"])
        if state == "attention":
            # Not x_svg: "attention" covers "you haven't turned this on
            # yet" as much as a real failure (setup_steps.py's own
            # docstring says so), and captioning is off on every fresh
            # launch by design. A cross there would read as broken on
            # the very first row a new user sees, so this draws a notice
            # mark instead of a mark of failure.
            return alert_svg(p["warn"])
        return circle_svg(p["muted"])

    def place_beside(self, window) -> None:
        """Anchor to the right of `window`'s frame, falling back to the left
        if there is no room, then clamp inside the screen's usable area.
        Only ever reads `window`'s geometry; this panel never moves the
        window it sits beside.

        Both `window` and this panel must already be shown before this is
        called. `frameGeometry()` on a widget that has never been shown
        reports no title bar on Windows, so calling this first and showing
        second computes a position that is short by the title bar's height
        and ends up overlapping the window instead of sitting beside it."""
        # window's own screen, not this panel's: before the first move a
        # top-level's screen() is always the primary one, so on a second
        # monitor self.screen() alone would clamp against the wrong bounds.
        screen = window.screen() or self.screen()
        if screen is None:
            return
        self.adjustSize()
        avail = screen.availableGeometry()
        host = window.frameGeometry()
        geo = self.frameGeometry()
        width, height = geo.width(), geo.height()

        x = host.right() + 1 + _GAP
        if x + width > avail.right():
            x = host.left() - _GAP - width
        x = max(avail.left(), min(x, avail.right() - width))
        y = max(avail.top(), min(host.top(), avail.bottom() - height))
        self.move(x, y)

    def close_panel(self) -> None:
        """Hide the panel. Safe to call twice."""
        self.hide()
