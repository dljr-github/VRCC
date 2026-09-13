"""The setup check panel: does it draw whatever `evaluate()` hands it,
without deciding anything on its own, and does it behave as a tool window
that never steals focus or blocks the offscreen suite.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QWidget

from vrcc.gui.setup_panel import _GAP, SetupPanel
from vrcc.gui.setup_steps import ROWS, SetupFacts, evaluate, row_text


@pytest.fixture(scope="module")
def qapp():
    # QApplication, never a bare QGuiApplication: the process-wide Qt
    # singleton cannot be upgraded once claimed, and a bare QGuiApplication
    # would poison every widget test that runs later in this process.
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture
def panel(qapp):
    p = SetupPanel()
    try:
        yield p
    finally:
        p.close_panel()
        p.deleteLater()


# -- what apply() draws -------------------------------------------------


def test_panel_starts_with_every_row_pending(panel):
    for row_id in ROWS:
        headline, detail = row_text(row_id, "pending")
        assert panel._headline_labels[row_id].text() == headline
        assert panel._detail_labels[row_id].text() == detail


def test_apply_draws_the_evaluator_output_directly():
    """The panel must not re-derive state; feeding it real evaluate() output
    for a facts snapshot that passes everything required must show each
    row's own pass text, not something the panel worked out itself."""
    facts = SetupFacts(
        captioning=True,
        engine_states={"stt": "ready"},
        spoken_utterance=True,
        vrchat_found=True,
        spoken_chatbox_send=True,
    )
    states = evaluate(facts)
    assert states["heard"] != "pass"  # sanity: heard is untouched by this snapshot

    p = SetupPanel()
    try:
        p.apply(states)
        for row_id in ROWS:
            headline, detail = row_text(row_id, states[row_id])
            assert p._headline_labels[row_id].text() == headline
            assert p._detail_labels[row_id].text() == detail
    finally:
        p.close_panel()
        p.deleteLater()


def test_apply_covers_every_state_a_row_can_reach(panel):
    """Each state actually reachable for its row, drawn through the same
    evaluator copy tests/test_setup_steps.py already covers."""
    combos = [
        SetupFacts(),
        SetupFacts(captioning=True),
        SetupFacts(engine_states={"stt": "failed"}),
        SetupFacts(captioning=True, mic_seen=False, spoken_utterance=False),
        SetupFacts(captioning=True, spoken_utterance=True),
        SetupFacts(vrchat_found=True),
        SetupFacts(spoken_chatbox_send=True),
        SetupFacts(heard_seen=True),
    ]
    for facts in combos:
        states = evaluate(facts)
        panel.apply(states)
        for row_id in ROWS:
            headline, detail = row_text(row_id, states[row_id])
            assert panel._headline_labels[row_id].text() == headline
            assert panel._detail_labels[row_id].text() == detail


def test_apply_treats_a_missing_row_as_pending(panel):
    """A caller handing a partial dict must get a panel that still draws
    every row, not a KeyError."""
    panel.apply({})
    for row_id in ROWS:
        _, detail = row_text(row_id, "pending")
        assert panel._detail_labels[row_id].text() == detail


def test_every_row_gets_a_real_icon_pixmap(panel):
    """pass/attention/pending must each render something (a blank icon slot
    would silently drop the only non-text signal a row carries), and the
    three must actually differ: a fixed icon regardless of state would pass
    a mere non-null check while telling the user nothing."""
    images = {}
    for state in ("pass", "attention", "pending"):
        panel.apply({row_id: state for row_id in ROWS})
        images[state] = {}
        for row_id in ROWS:
            pm = panel._icon_labels[row_id].pixmap()
            assert pm is not None and not pm.isNull()
            images[state][row_id] = pm.toImage()

    for row_id in ROWS:
        assert images["pass"][row_id] != images["attention"][row_id]
        assert images["pass"][row_id] != images["pending"][row_id]
        assert images["attention"][row_id] != images["pending"][row_id]


# -- window identity ------------------------------------------------------


def test_panel_never_steals_focus(panel):
    # windowType(), not a bitwise AND on windowFlags(): Qt.WindowType.Tool's
    # value is a superset bitmask of Window and Dialog, so an AND check is
    # truthy for a plain window too and would not catch the flag being lost.
    assert panel.windowType() == Qt.WindowType.Tool
    assert panel.testAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)


def test_panel_title_is_the_product_name(panel):
    # "VRCC" is the same non-translated product name boot_panel.py titles
    # itself with; it needs no catalog entry in any of the 17 languages.
    assert panel.windowTitle() == "VRCC"


# -- close_panel ----------------------------------------------------------


def test_close_panel_hides_and_is_safe_to_call_twice(panel):
    panel.show()
    assert panel.isVisible()
    panel.close_panel()
    assert not panel.isVisible()
    panel.close_panel()  # must not raise
    assert not panel.isVisible()


# -- place_beside ----------------------------------------------------------


def test_place_beside_anchors_to_the_right_when_there_is_room(panel):
    host = QWidget()
    try:
        host.setGeometry(50, 50, 400, 300)
        panel.place_beside(host)
        host_frame = host.frameGeometry()
        panel_frame = panel.frameGeometry()
        assert panel_frame.left() == host_frame.right() + 1 + _GAP
        assert panel_frame.top() == host_frame.top()
    finally:
        host.deleteLater()


def test_place_beside_falls_back_to_the_left_without_room_on_the_right(panel):
    host = QWidget()
    try:
        avail = panel.screen().availableGeometry()
        # Parked hard against the right edge: nothing fits between the host
        # and the screen edge, so the panel must land on the host's left.
        host.setGeometry(avail.right() - 90, 50, 90, 300)
        panel.place_beside(host)
        host_frame = host.frameGeometry()
        panel_frame = panel.frameGeometry()
        assert panel_frame.right() == host_frame.left() - 1 - _GAP
    finally:
        host.deleteLater()


def test_place_beside_clamps_inside_the_screen(panel):
    host = QWidget()
    try:
        avail = panel.screen().availableGeometry()
        # A host near the bottom edge must not push the panel off-screen.
        host.setGeometry(50, avail.bottom() - 10, 400, 300)
        panel.place_beside(host)
        panel_frame = panel.frameGeometry()
        assert panel_frame.top() >= avail.top()
        assert panel_frame.bottom() <= avail.bottom()
    finally:
        host.deleteLater()


def test_place_beside_clamps_against_the_hosts_screen_not_the_panels(panel, monkeypatch):
    """A panel that has never been moved reports the primary screen from
    `self.screen()`; on a second monitor that is the wrong bounds to clamp
    against. `window.screen()` must win."""
    from PySide6.QtCore import QRect

    host = QWidget()
    try:
        host.setGeometry(50, 50, 400, 300)

        class _FakeScreen:
            def availableGeometry(self) -> QRect:
                return QRect(2000, 0, 800, 800)

        monkeypatch.setattr(host, "screen", lambda: _FakeScreen())
        panel.place_beside(host)
        assert panel.frameGeometry().left() >= 2000
    finally:
        host.deleteLater()


def test_place_beside_returns_early_without_a_screen(panel, monkeypatch):
    host = QWidget()
    try:
        before = panel.pos()
        # place_beside prefers the host's own screen (see the multi-monitor
        # comment on place_beside), so both must be gone for this to hit
        # the early return.
        monkeypatch.setattr(host, "screen", lambda: None)
        monkeypatch.setattr(panel, "screen", lambda: None)
        panel.place_beside(host)
        assert panel.pos() == before
    finally:
        host.deleteLater()
