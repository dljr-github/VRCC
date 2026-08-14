"""Writes the caption feed's document and keeps the reading position.

Rows go in one at a time and only where they changed. Replacing the whole
document per event costs a full re-layout, which grows with the session: one
utterance publishes recognized, translated and sent, and `setHtml` on all
three measured 16.8 ms on a fresh window against 180 ms once the model sat at
its 200-row cap. Writing through a cursor measures 3.1 ms and 5.9 ms. Building
the HTML was never the cost; Qt's layout was.

Scroll position is the other half of the same job. A proximity check (is the
scrollbar within a couple of pixels of maximum, sampled around a render)
cannot decide whether the user is following: QTextDocument lays out large
documents progressively, so maximum() read right after a write can undershoot
the settled height, and a resize that rewraps rows grows it later still. A pin
that lands short then reads as a user scroll on the next sample and the feed
freezes (tests/test_main_window_ui.py measured the stall offscreen: value
stuck at 2561 against a maximum that grew to 3546). Following is therefore an
explicit flag: on by default, cleared when the user scrolls away from the
bottom, restored when they scroll back. While following, the view re-pins after
every write and on every scrollbar range change, so layout that settles late
can never strand it.

A user reading history is never moved, which is why rows the model trims off
the top are removed here rather than re-rendered away: the removal's height is
measured, and the held offset moves down by exactly that much, so the text
under their eyes stays under their eyes.
"""

from __future__ import annotations

from PySide6.QtGui import QTextCursor
from PySide6.QtWidgets import QTextBrowser

# How close to maximum (scrollbar units) still counts as "at the bottom"
# when a user scroll decides whether following stays on or resumes.
_BOTTOM_SLOP = 2


class LogFollower:
    """Owns one QTextBrowser's document content, the follow flag and every
    programmatic scroll of it.

    valueChanged emits that arrive while no programmatic adjustment is in
    flight are user scrolls; the _adjusting guard (MainWindow's _loading
    idiom) keeps the follower's own setValue calls, and the scroll reset a
    document edit can cause, from being mistaken for one.
    """

    def __init__(self, view: QTextBrowser) -> None:
        self._view = view
        self._bar = view.verticalScrollBar()
        self.following = True
        self._adjusting = False
        # Every row currently in the document, in order: its key, the HTML it
        # was written from (the change check) and the character position it
        # starts at (where a rewrite cuts).
        self._rows: list[tuple[int, str, int]] = []
        self._bar.valueChanged.connect(self._on_value_changed)
        self._bar.rangeChanged.connect(self._on_range_changed)

    def set_html(self, html: str) -> None:
        """Replace the whole document (the empty state), keeping the reading
        position: pinned to the bottom while following, else exactly where the
        user left it (setValue clamps if the new document is shorter)."""
        previous = self._bar.value()
        was = self._adjusting
        self._adjusting = True
        try:
            self._rows = []
            self._view.setHtml(html)
            self._bar.setValue(self._bar.maximum() if self.following else previous)
        finally:
            self._adjusting = was

    def set_rows(self, rows: list[tuple[int, str]]) -> None:
        """Bring the document in line with ``rows`` ((key, html), oldest
        first), rewriting only what differs from what is already there."""
        if not rows:
            self.set_html("")
            return
        previous = self._bar.value()
        was = self._adjusting
        self._adjusting = True
        try:
            shift = self._drop_evicted(rows)
            self._write_changed(rows)
            if self.following:
                self._bar.setValue(self._bar.maximum())
            else:
                self._bar.setValue(max(0, previous - shift))
        finally:
            self._adjusting = was

    # -- document writing --------------------------------------------------

    def _drop_evicted(self, rows: list[tuple[int, str]]) -> int:
        """Remove the rows the model has trimmed off the top. Returns how many
        pixels of content that took out: everything still on screen moved up by
        exactly that, which is what a held reading offset owes."""
        keys = {key for key, _ in rows}
        drop = 0
        while drop < len(self._rows) and self._rows[drop][0] not in keys:
            drop += 1
        if drop == 0:
            return 0
        doc = self._view.document()
        if drop >= len(self._rows):
            # Nothing on screen survives, so there is no position to hold.
            self._rows = []
            return 0
        cut = self._rows[drop][2]
        layout = doc.documentLayout()
        before = layout.blockBoundingRect(doc.findBlock(cut)).top()
        cursor = QTextCursor(doc)
        cursor.setPosition(0)
        cursor.setPosition(cut, QTextCursor.MoveMode.KeepAnchor)
        cursor.removeSelectedText()
        # The cut is on a block boundary, so the removal is exactly `cut`
        # characters and every surviving position moves down by that much.
        self._rows = [(key, html, pos - cut) for key, html, pos in self._rows[drop:]]
        # Measured after the fact rather than assumed: the document margin sits
        # above the first row either way, and block margins can collapse.
        return round(before - layout.blockBoundingRect(doc.firstBlock()).top())

    def _write_changed(self, rows: list[tuple[int, str]]) -> None:
        same = 0
        while (
            same < len(self._rows)
            and same < len(rows)
            and self._rows[same][0] == rows[same][0]
            and self._rows[same][1] == rows[same][1]
        ):
            same += 1
        if same == len(rows) == len(self._rows):
            return
        doc = self._view.document()
        cursor = QTextCursor(doc)
        if not self._rows:
            # The document still holds the empty-state placeholder (or an
            # evicted-out feed). Select-all rather than clear(): clear() drops
            # the root frame format the document margin lives in.
            cursor.select(QTextCursor.SelectionType.Document)
            cursor.removeSelectedText()
        elif same < len(self._rows):
            # A row that changed is rewritten together with everything after
            # it. Rows only ever change at the tail (a status resolving), so
            # that span is short; the head is left untouched, which is the
            # whole point.
            cursor.setPosition(self._rows[same][2])
            cursor.movePosition(
                QTextCursor.MoveOperation.End, QTextCursor.MoveMode.KeepAnchor
            )
            cursor.removeSelectedText()
            del self._rows[same:]
        for key, html in rows[same:]:
            cursor.movePosition(QTextCursor.MoveOperation.End)
            start = cursor.position()
            cursor.insertHtml(html)
            self._rows.append((key, html, start))

    # -- scroll ------------------------------------------------------------

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
