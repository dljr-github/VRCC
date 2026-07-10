"""Keeps the caption feed pinned to its newest entry while the user follows.

A proximity check (is the scrollbar within a couple of pixels of maximum,
sampled around setHtml) cannot decide this reliably: QTextDocument lays out
large documents progressively, so maximum() read right after setHtml can
undershoot the settled height, and a resize that rewraps rows grows it
later still. A pin that lands short then reads as a user scroll on the next
sample and the feed freezes (tests/test_main_window_ui.py measured the
stall offscreen: value stuck at 2561 against a maximum that grew to 3546).
Following is therefore an explicit flag: on by default, cleared when the
user scrolls away from the bottom, restored when they scroll back. While
following, the view re-pins after every render and on every scrollbar range
change, so layout that settles late can never strand it. A user reading
history is never moved.
"""

from __future__ import annotations

from PySide6.QtWidgets import QTextBrowser

# How close to maximum (scrollbar units) still counts as "at the bottom"
# when a user scroll decides whether following stays on or resumes.
_BOTTOM_SLOP = 2


class LogFollower:
    """Owns the follow flag and every programmatic scroll of one QTextBrowser.

    valueChanged emits that arrive while no programmatic adjustment is in
    flight are user scrolls; the _adjusting guard (MainWindow's _loading
    idiom) keeps the follower's own setValue calls, and setHtml's internal
    scroll reset, from being mistaken for one.
    """

    def __init__(self, view: QTextBrowser) -> None:
        self._view = view
        self._bar = view.verticalScrollBar()
        self.following = True
        self._adjusting = False
        self._bar.valueChanged.connect(self._on_value_changed)
        self._bar.rangeChanged.connect(self._on_range_changed)

    def set_html(self, html: str) -> None:
        """Replace the document, keeping the reading position: pinned to the
        bottom while following, else exactly where the user left it (setValue
        clamps if the new document is shorter)."""
        previous = self._bar.value()
        was = self._adjusting
        self._adjusting = True
        try:
            self._view.setHtml(html)
            self._bar.setValue(self._bar.maximum() if self.following else previous)
        finally:
            self._adjusting = was

    def _on_value_changed(self, value: int) -> None:
        if self._adjusting:
            return
        self.following = value >= self._bar.maximum() - _BOTTOM_SLOP

    def _on_range_changed(self, _minimum: int, maximum: int) -> None:
        # Deferred layout and rewrapping resizes grow the range after a pin;
        # while following, chase the new bottom. A held reading position needs
        # no correction: Qt clamps it only if the document shrank.
        if not self.following:
            return
        was = self._adjusting
        self._adjusting = True
        try:
            self._bar.setValue(maximum)
        finally:
            self._adjusting = was
