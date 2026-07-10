"""Small reusable themed widgets, so main_window/settings stay focused."""

from __future__ import annotations

from PySide6.QtCore import QByteArray, Qt, Signal
from PySide6.QtGui import QIcon, QPainter, QPixmap
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget,
)

from vrcc.gui.icons import arrow_svg, mic_svg  # re-exported: icons.py is the single home
from vrcc.gui.style import PALETTE


def no_wheel(widget: QWidget) -> QWidget:
    """Make ``widget`` ignore the scroll wheel. Spin boxes and combo boxes
    step their value when the pointer crosses them mid-scroll; ignoring the
    event lets it bubble up to the enclosing scroll area instead."""
    widget.wheelEvent = lambda event: event.ignore()
    return widget


def svg_pixmap(svg: str, size: int) -> QPixmap | None:
    """Render an inline SVG to a transparent pixmap; ``None`` if invalid."""
    renderer = QSvgRenderer(QByteArray(svg.encode("utf-8")))
    if not renderer.isValid():
        return None
    pm = QPixmap(size, size)
    pm.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pm)
    renderer.render(painter)
    painter.end()
    return pm


def icon_label(
    svg: str, size: int = 18, colors: dict | None = None, fallback_text: str = ""
) -> QLabel:
    """Static icon as a QLabel (no button chrome). When the SVG can't render,
    shows ``fallback_text`` (plain ASCII, never emoji) in the muted color."""
    label = QLabel()
    pm = svg_pixmap(svg, size)
    if pm is not None:
        label.setPixmap(pm)
    elif fallback_text:
        p = colors or PALETTE["dark"]
        label.setText(fallback_text)
        label.setStyleSheet(f"color: {p['muted']}; background: transparent;")
    return label


class Card(QFrame):
    # `colors` is a PALETTE dict; None keeps the dark default for old callers.
    def __init__(self, parent=None, colors: dict | None = None) -> None:
        super().__init__(parent)
        p = colors or PALETTE["dark"]
        self.setObjectName("Card")
        self.body = QVBoxLayout(self)
        self.body.setContentsMargins(14, 12, 14, 12)
        self.body.setSpacing(8)
        self.setStyleSheet(
            f"#Card {{ background: {p['surface']}; "
            f"border: 1px solid {p['border']}; border-radius: 12px; }}"
        )


class SegmentedControl(QWidget):
    # Emits the segment's VALUE. An option may be a plain string (value ==
    # label) or a (value, label) pair, so a translated label can sit on a
    # stable value -- values are what value()/set_value()/changed carry, and
    # they must never be translated (callers compare and persist them).
    changed = Signal(str)

    def __init__(self, options: list[str | tuple[str, str]], selected: str) -> None:
        super().__init__()
        self._value = selected
        self._buttons: dict[str, QPushButton] = {}
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(4)
        for opt in options:
            value, label = opt if isinstance(opt, tuple) else (opt, opt)
            b = QPushButton(label)
            b.setCheckable(True)
            b.setChecked(value == selected)
            b.clicked.connect(lambda _=False, o=value: self.set_value(o))
            self._buttons[value] = b
            row.addWidget(b)
        self._repolish()

    def value(self) -> str:
        return self._value

    def set_value(self, v: str) -> None:
        if v not in self._buttons:
            return
        changed = v != self._value
        self._value = v
        for opt, b in self._buttons.items():
            b.setChecked(opt == v)
        self._repolish()
        if changed:
            self.changed.emit(v)

    def set_option_enabled(self, option: str, enabled: bool, tooltip: str = "") -> None:
        """Enable/disable one segment; an optional tooltip explains why."""
        b = self._buttons.get(option)
        if b is None:
            return
        b.setEnabled(enabled)
        b.setToolTip(tooltip)

    def _repolish(self) -> None:
        for opt, b in self._buttons.items():
            b.setProperty("segActive", opt == self._value)
            b.style().unpolish(b)
            b.style().polish(b)


class IconButton(QPushButton):
    """A square icon button that renders an inline SVG reliably.

    Uses QSvgRenderer (via svg_pixmap) rather than QPixmap.loadFromData(..., "SVG"),
    which needs the optional qsvg plugin and silently produced blank icons when it
    was missing. Falls back to fallback_text (plain text, never emoji) if the SVG
    can't render, so an icon is never blank.
    """

    def __init__(self, svg: str, tooltip: str = "", fallback_text: str = "") -> None:
        super().__init__()
        self.setFixedSize(38, 38)
        # QSS min-height overrides a fixed size on that axis; opt out so the
        # shared control-kit min-height rule doesn't inflate this square button.
        self.setProperty("compact", True)
        self.setToolTip(tooltip)
        pm = svg_pixmap(svg, 20)
        if pm is not None:
            self.setIcon(QIcon(pm))
            self.setIconSize(pm.size())
        elif fallback_text:
            self.setText(fallback_text)


class MicMeter(QWidget):
    def __init__(self, parent=None, colors: dict | None = None) -> None:
        super().__init__(parent)
        self._colors = colors or PALETTE["dark"]
        self._level = 0.0
        self._active = True
        self.setFixedHeight(18)
        self.setMinimumWidth(60)

    def set_level(self, rms: float) -> None:
        self._level = max(0.0, min(1.0, rms * 20.0))
        self.update()

    def set_active(self, active: bool) -> None:
        self._active = active
        self.update()

    _BARS = 8
    _GAP = 3

    def _segment_rects(self):
        # All segments share one size (full widget height); only the fill
        # count encodes level, so the meter never reads as a bar chart. The
        # floor-division remainder is centered (half margin each side) rather
        # than left entirely as a stray gap after the last segment.
        from PySide6.QtCore import QRect

        gap = self._GAP
        n = self._BARS
        total_gap = gap * (n - 1)
        bw = max(2, (self.width() - total_gap) // n)
        remainder = max(0, self.width() - (bw * n + total_gap))
        offset = remainder // 2
        return [QRect(offset + i * (bw + gap), 0, bw, self.height()) for i in range(n)]

    def _segment_style(self, i: int, filled_count: int) -> tuple[str, float]:
        """(fill color, opacity) for segment ``i``. Unlit segments always use
        the ``border`` token at full opacity -- the old low-opacity accent/muted
        fill blended to near-invisible against ``ground``, especially now that
        the meter starts dimmed (captioning off) at launch."""
        if i < filled_count:
            return self._colors["accent"], 0.9
        return self._colors["border"], 1.0

    def paintEvent(self, event) -> None:  # noqa: N802 -- Qt override
        from PySide6.QtGui import QColor, QPainter

        painter = QPainter(self)
        filled_count = round(self._level * self._BARS) if self._active else 0
        for i, rect in enumerate(self._segment_rects()):
            color, opacity = self._segment_style(i, filled_count)
            painter.setOpacity(opacity)
            painter.fillRect(rect, QColor(color))
        painter.end()
