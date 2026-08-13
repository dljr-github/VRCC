"""How the caption feed is written and how it behaves once it is full.

The feed used to be re-rendered whole on every bus event, which is a full
QTextDocument layout each time: an utterance publishes three events, and those
three measured 180 ms of blocked GUI thread once the model sat at its 200-row
cap. These tests pin the two things that fixed it, both observable without a
stopwatch: only the rows that changed are rewritten (Qt stamps a revision on
every block it edits), and a reader scrolled into history keeps the same text
under their eyes when the cap trims the top away.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from types import SimpleNamespace

import pytest

from vrcc.core.bus import EventBus
from vrcc.core.config import ConfigStore, default_paths


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


class _FakePipeline:
    captioning_enabled = False
    mt_active = True

    def __init__(self) -> None:
        self.typed: list[str] = []

    def submit_typed(self, text: str) -> bool:
        self.typed.append(text)
        return True

    def set_captioning(self, enabled: bool) -> None:
        self.captioning_enabled = bool(enabled)


def _window(tmp_path, store=None):
    from vrcc.gui.bridge import BusBridge
    from vrcc.gui.main_window import MainWindow

    if store is None:
        store = ConfigStore(default_paths(portable=True, app_dir=tmp_path).config_file)
    bridge = BusBridge(EventBus())
    window = MainWindow(
        bridge, store, _FakePipeline(),
        on_open_settings=lambda: None, on_open_models=lambda: None,
    )
    return window, bridge


def _say(window, utterance_id, text):
    window._on_phrase_recognized(
        SimpleNamespace(utterance_id=utterance_id, text=text)
    )


def _settle(qapp, window):
    qapp.processEvents()
    window._log.document().documentLayout().documentSize()
    qapp.processEvents()


def _revisions(window) -> list[int]:
    """Qt's per-block edit stamp, one entry per block in document order."""
    doc = window._log.document()
    out = []
    block = doc.begin()
    while block.isValid():
        out.append(block.revision())
        block = block.next()
    return out


def _block(window, prefix: str):
    doc = window._log.document()
    block = doc.begin()
    while block.isValid():
        if block.text().startswith(prefix):
            return block
        block = block.next()
    raise AssertionError(f"no block starting {prefix!r} in the feed")


def _block_top(window, prefix: str) -> float:
    doc = window._log.document()
    return doc.documentLayout().blockBoundingRect(_block(window, prefix)).top()


# -- writing only what changed -----------------------------------------------


def test_a_status_change_rewrites_only_its_own_row(qapp, tmp_path):
    window, bridge = _window(tmp_path)
    try:
        for i in range(1, 21):
            _say(window, i, f"row {i} spoken out loud")
        _settle(qapp, window)
        before = _revisions(window)

        window._on_chatbox_sent(SimpleNamespace(utterance_id=20, truncated=False))
        _settle(qapp, window)
        after = _revisions(window)

        assert len(before) == len(after)
        changed = [i for i, (a, b) in enumerate(zip(before, after)) if a != b]
        # The last row is four blocks; nothing above it may be touched, or the
        # cost of a status change would grow with the length of the session.
        assert changed and min(changed) >= len(before) - 6
    finally:
        window.close(); window.deleteLater(); bridge.detach()


def test_a_new_caption_at_the_cap_leaves_the_history_alone(qapp, tmp_path):
    # The regression this guards: a full setHtml per event restamps every
    # block in the document, which at the cap is the whole 60 ms a render.
    # Block index is no good here (the cap evicts from the top and shifts every
    # index), so the probe rows are found by their text.
    window, bridge = _window(tmp_path)
    try:
        for i in range(1, 201):
            _say(window, i, f"row {i} spoken out loud")
        _settle(qapp, window)
        assert len(_revisions(window)) > 500  # the document really is large
        probes = ["row 50", "row 100", "row 199"]
        before = [_block(window, p).revision() for p in probes]

        _say(window, 201, "one more caption")
        _settle(qapp, window)

        assert [_block(window, p).revision() for p in probes] == before
    finally:
        window.close(); window.deleteLater(); bridge.detach()


def test_feed_content_matches_a_full_render_after_the_cap_trims(qapp, tmp_path):
    # Incremental writing is only worth anything if it converges on the same
    # document: 220 utterances through a 200-row cap, each stepping
    # recognized -> translated -> sent.
    from PySide6.QtGui import QTextDocument

    from vrcc.gui.caption_log import render_rows_html

    store = ConfigStore(default_paths(portable=True, app_dir=tmp_path).config_file)
    store.config.translate.enabled = True
    window, bridge = _window(tmp_path, store=store)
    try:
        for i in range(1, 221):
            _say(window, i, f"row {i} spoken out loud")
            window._on_phrase_translated(
                SimpleNamespace(utterance_id=i, translations=[("Japanese", f"翻訳 {i}")])
            )
            window._on_chatbox_sent(
                SimpleNamespace(utterance_id=i, truncated=(i % 5 == 0))
            )
        _settle(qapp, window)

        fresh = QTextDocument()
        fresh.setHtml(
            render_rows_html(window._caption_model.rows(), window._p, window._scale)
        )
        live = [line for line in window._log.toPlainText().splitlines() if line]
        expected = [line for line in fresh.toPlainText().splitlines() if line]
        assert live == expected
        assert live[0].startswith("21:") or "row 21" in " ".join(live[:4])
    finally:
        window.close(); window.deleteLater(); bridge.detach()


# -- reading position across a cap eviction ----------------------------------


def test_reader_stays_on_the_same_text_when_the_cap_trims(qapp, tmp_path):
    # Holding the scrollbar VALUE only works while nothing leaves the top. Once
    # the model is at its cap every caption evicts one, and the held offset
    # then points at different content (measured: about a screenful every 20
    # utterances). The eviction's height has to come off the held offset.
    window, bridge = _window(tmp_path)
    try:
        window.resize(520, 240)
        window.show()
        qapp.processEvents()
        for i in range(1, 201):
            _say(window, i, f"row {i} history the user is reading back through")
        _settle(qapp, window)

        bar = window._log.verticalScrollBar()
        bar.setValue(bar.maximum() // 2)  # a user scroll
        assert window._log_follow.following is False
        _settle(qapp, window)
        on_screen = _block_top(window, "row 150") - bar.value()

        for i in range(201, 221):  # every one of these trims a row off the top
            _say(window, i, f"row {i} arriving while they read")
            _settle(qapp, window)

        assert _block_top(window, "row 150") - bar.value() == pytest.approx(on_screen, abs=1)
    finally:
        window.close(); window.deleteLater(); bridge.detach()


def test_following_reader_still_lands_on_the_newest_row_at_the_cap(qapp, tmp_path):
    window, bridge = _window(tmp_path)
    try:
        window.resize(520, 240)
        window.show()
        qapp.processEvents()
        for i in range(1, 221):
            _say(window, i, f"row {i} spoken out loud with enough words to wrap")
            _settle(qapp, window)
        bar = window._log.verticalScrollBar()
        assert bar.value() == bar.maximum() > 0
    finally:
        window.close(); window.deleteLater(); bridge.detach()


# -- status bar, errors, compose row -----------------------------------------


def test_first_status_flash_does_not_move_the_caption_feed(qapp, tmp_path):
    # statusBar() builds the bar on first call and takes its height out of the
    # central widget, so a lazily built one shifts the feed mid-session.
    window, bridge = _window(tmp_path)
    try:
        window.resize(700, 400)
        window.show()
        qapp.processEvents()
        before = window._log.height()
        window._flash_status("something happened")
        qapp.processEvents()
        assert window._log.height() == before
    finally:
        window.close(); window.deleteLater(); bridge.detach()


def test_unknown_error_code_is_not_shown_raw(qapp, tmp_path):
    window, bridge = _window(tmp_path)
    try:
        window._on_app_error(
            SimpleNamespace(
                code="WHAT_IS_THIS",
                message="ct2 assertion failed at src/layers/attention.cc:88",
            )
        )
        shown = window.statusBar().currentMessage()
        assert shown
        assert "WHAT_IS_THIS" not in shown
        assert "attention.cc" not in shown
    finally:
        window.close(); window.deleteLater(); bridge.detach()


def test_known_error_code_still_shows_its_own_sentence(qapp, tmp_path):
    window, bridge = _window(tmp_path)
    try:
        window._on_app_error(
            SimpleNamespace(code="MIC_OPEN_FAILED", message="device 3 busy")
        )
        assert "microphone" in window.statusBar().currentMessage()
    finally:
        window.close(); window.deleteLater(); bridge.detach()


def test_blank_send_clears_the_box_and_submits_nothing(qapp, tmp_path):
    window, bridge = _window(tmp_path)
    try:
        window._text_input.setText("   ")
        window._on_send_clicked()
        assert window._text_input.text() == ""
        assert window._pipeline.typed == []
    finally:
        window.close(); window.deleteLater(); bridge.detach()


def test_empty_translation_resolves_a_translating_row(qapp, tmp_path):
    # The GUI half of the pipeline's fallback: translation failed, nothing was
    # sent, and the empty PhraseTranslated is the only thing that can move the
    # row off "translating…".
    store = ConfigStore(default_paths(portable=True, app_dir=tmp_path).config_file)
    store.config.translate.enabled = True
    store.config.osc.send_to_vrchat = False
    window, bridge = _window(tmp_path, store=store)
    try:
        _say(window, 4, "hello there")
        assert window._caption_model.rows()[0].status == "translating"
        window._on_phrase_translated(
            SimpleNamespace(utterance_id=4, translations=())
        )
        assert window._caption_model.rows()[0].status == "not_sent"
    finally:
        window.close(); window.deleteLater(); bridge.detach()
