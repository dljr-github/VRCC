"""Tests for the caption-row model and its HTML rendering; the layout smoke
test uses an offscreen QTextDocument to verify Qt's actual rendering.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from vrcc.gui.caption_log import (
    LISTENING,
    NOT_SENT,
    QUEUED,
    SENT,
    TRANSLATING,
    TRUNCATED,
    CaptionModel,
    render_rows_html,
    status_markup,
)


class _Clock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


def _model():
    return CaptionModel(clock=_Clock(), time_label=lambda: "13:42")


def test_recognized_translated_sent_updates_one_row_in_place():
    m = _model()
    m.recognized(1, "hello", translate_enabled=True, send_enabled=True)
    assert len(m.rows()) == 1
    assert m.rows()[0].status == TRANSLATING

    m.translated(1, [("Japanese", "こんにちは")], send_enabled=True)
    assert len(m.rows()) == 1  # same row, not a new one
    assert m.rows()[0].translations == [("Japanese", "こんにちは")]
    assert m.rows()[0].status == QUEUED

    m.sent(1, truncated=False)
    assert m.rows()[0].status == SENT


def test_status_latency_computed_from_clock():
    clock = _Clock()
    m = CaptionModel(clock=clock, time_label=lambda: "13:42")
    m.recognized(1, "hi", translate_enabled=False, send_enabled=True)
    clock.t = 0.4  # 400 ms later
    m.sent(1, truncated=False)
    assert m.rows()[0].latency_ms == 400


def test_truncated_sent_marks_truncated():
    m = _model()
    m.recognized(2, "a very long line", translate_enabled=False, send_enabled=True)
    m.sent(2, truncated=True)
    assert m.rows()[0].status == TRUNCATED


def test_send_disabled_shows_not_sent():
    m = _model()
    m.recognized(3, "hi", translate_enabled=False, send_enabled=False)
    assert m.rows()[0].status == NOT_SENT


def test_reused_utterance_id_zero_creates_distinct_rows():
    # Defensive property of the model itself: a reused id (whatever the
    # caller's reason) always starts a fresh row, and a later event for that
    # id attaches to the most recent one, not an older row sharing the id.
    m = _model()
    m.recognized(0, "first typed", translate_enabled=False, send_enabled=True)
    m.recognized(0, "second typed", translate_enabled=False, send_enabled=True)
    m.sent(0, truncated=False)
    rows = m.rows()
    assert [r.original for r in rows] == ["first typed", "second typed"]
    assert rows[0].status == QUEUED  # first is untouched
    assert rows[1].status == SENT  # sent attached to the newest


def test_distinct_ids_do_not_cross_stamp_translated_or_sent():
    # Regression for the typed-message misattribution bug: with each
    # submission on its own id (as Pipeline now assigns), a translated()/
    # sent() for one row must never land on a different, still-pending row.
    m = _model()
    m.recognized(-1, "first typed", translate_enabled=True, send_enabled=True)
    m.recognized(-2, "second typed", translate_enabled=True, send_enabled=True)

    m.translated(-1, [("Japanese", "こんにちは")], send_enabled=True)
    m.sent(-1, truncated=False)

    rows = m.rows()
    assert rows[0].original == "first typed"
    assert rows[0].status == SENT
    assert rows[0].translations == [("Japanese", "こんにちは")]

    assert rows[1].original == "second typed"
    assert rows[1].status == TRANSLATING  # untouched: still awaiting its own translate
    assert rows[1].translations == []


def test_partial_then_partial_updates_one_row_in_place():
    m = _model()
    m.partial(1, "hel")
    assert len(m.rows()) == 1
    assert m.rows()[0].original == "hel"
    assert m.rows()[0].status == LISTENING

    m.partial(1, "hello there")
    assert len(m.rows()) == 1  # same row, not a new one
    assert m.rows()[0].original == "hello there"
    assert m.rows()[0].status == LISTENING


def test_partial_then_recognized_firms_the_same_row():
    m = _model()
    m.partial(1, "hel")
    assert len(m.rows()) == 1

    m.recognized(1, "hello there", translate_enabled=False, send_enabled=True)
    assert len(m.rows()) == 1  # no duplicate row
    assert m.rows()[0].original == "hello there"
    assert m.rows()[0].status == QUEUED


def test_partial_after_terminal_row_does_not_resurrect_it():
    m = _model()
    m.recognized(1, "hello", translate_enabled=False, send_enabled=True)
    m.sent(1, truncated=False)
    assert m.rows()[0].status == SENT

    m.partial(1, "a stale partial")
    rows = m.rows()
    # The terminal row is untouched: its text and status never move backward.
    assert rows[0].original == "hello"
    assert rows[0].status == SENT
    # A fresh row is started for the new activity on this utterance id (same
    # policy recognized() already applies to a reused id).
    assert len(rows) == 2
    assert rows[1].original == "a stale partial"
    assert rows[1].status == LISTENING


def test_cap_trims_oldest_rows():
    m = CaptionModel(cap=3, clock=_Clock(), time_label=lambda: "13:42")
    for i in range(5):
        m.recognized(i, f"utt{i}", translate_enabled=False, send_enabled=True)
    rows = m.rows()
    assert [r.original for r in rows] == ["utt2", "utt3", "utt4"]
    # A late event for an evicted utterance must be ignored, not crash.
    m.sent(0, truncated=False)
    assert len(m.rows()) == 3


def test_render_escapes_html_and_marks_status():
    m = _model()
    m.recognized(1, "<b>hi & bye</b>", translate_enabled=False, send_enabled=True)
    m.sent(1, truncated=True)
    html = render_rows_html(m.rows())
    assert "&lt;b&gt;hi &amp; bye&lt;/b&gt;" in html  # escaped, not injected
    assert "shortened to fit" in html


def test_truncated_status_reads_as_a_success():
    m = _model()
    m.recognized(1, "hi", translate_enabled=False, send_enabled=True)
    m.sent(1, truncated=True)
    text, _color = status_markup(m.rows()[0])
    assert "shortened to fit" in text
    assert text.startswith("sent")  # still a successful send, not an error


def test_status_lines_have_no_stranded_separators():
    # The truncated status is the one exception that stays multi-line --
    # "sent" / "shortened to fit" / "0.3s" -- with no middot stranded at a
    # line edge (they read as stray marks).
    clock = _Clock()
    m = CaptionModel(clock=clock, time_label=lambda: "13:42")
    m.recognized(1, "hi", translate_enabled=False, send_enabled=True)
    clock.t = 0.3
    m.sent(1, truncated=True)
    text, _color = status_markup(m.rows()[0])
    assert "·" not in text
    assert "sent<br/>shortened to fit" in text
    assert "0.3s" in text


def test_sent_status_with_latency_is_one_line_with_inline_middot():
    # Plain sent (not truncated) renders on ONE line: "sent · 0.3s". The
    # middot sits between two words, never at a line edge.
    clock = _Clock()
    m = CaptionModel(clock=clock, time_label=lambda: "13:42")
    m.recognized(1, "hi", translate_enabled=False, send_enabled=True)
    clock.t = 0.3
    m.sent(1, truncated=False)
    text, _color = status_markup(m.rows()[0])
    assert "<br/>" not in text
    assert "sent" in text and "0.3s" in text
    assert not text.strip().startswith("·")
    assert not text.strip().endswith("·")


def test_queued_and_not_sent_statuses_are_single_line():
    # No status but truncated ever stacks multiple lines.
    m = _model()
    m.recognized(1, "hi", translate_enabled=True, send_enabled=True)
    translating_text, _ = status_markup(m.rows()[0])
    assert "<br/>" not in translating_text

    m2 = _model()
    m2.recognized(2, "hi", translate_enabled=False, send_enabled=True)
    queued_text, _ = status_markup(m2.rows()[0])
    assert "<br/>" not in queued_text

    m3 = _model()
    m3.recognized(3, "hi", translate_enabled=False, send_enabled=False)
    not_sent_text, _ = status_markup(m3.rows()[0])
    assert "<br/>" not in not_sent_text


def test_status_lands_in_its_own_cell_not_inline_with_caption():
    # Qt's rich-text engine ignores white-space:nowrap (verified empirically:
    # it still hard-breaks at nbsp positions), so the anti-orphan guarantee is
    # structural: the status must live in its own table cell, never flowing
    # inline after the caption text.
    m = _model()
    m.recognized(1, "hello there", translate_enabled=False, send_enabled=True)
    m.sent(1, truncated=True)
    html = render_rows_html(m.rows())
    assert html.count("<td") == 3  # gutter | content | status
    between = html.split("hello there", 1)[1].split("shortened to fit", 1)[0]
    assert "<td" in between  # a cell boundary separates caption from status
    assert "white-space:nowrap" not in html  # dead CSS, dropped


def _rendered_lines(html: str, text_width: int) -> list[str]:
    """Text of every line Qt actually lays out at the given document width."""
    from PySide6.QtGui import QTextDocument
    from PySide6.QtWidgets import QApplication

    _ = QApplication.instance() or QApplication([])
    doc = QTextDocument()
    doc.setHtml(html)
    doc.setTextWidth(text_width)
    doc.documentLayout().documentSize()  # force layout
    lines = []
    block = doc.begin()
    while block.isValid():
        layout, text = block.layout(), block.text()
        for i in range(layout.lineCount()):
            ln = layout.lineAt(i)
            lines.append(text[ln.textStart(): ln.textStart() + ln.textLength()])
        block = block.next()
    return lines


def test_qt_layout_never_splits_status_mid_phrase():
    # Regression for the orphan-wrap bug: with the old inline status span, Qt
    # rendered 'sent · shortened ' / 'to fit · 0.0s' (breaking at nbsp
    # positions despite white-space:nowrap; verified empirically at 330px,
    # even mid-word at 360px). The fixed-width status cell must keep the
    # phrase together at any document width.
    m = _model()
    m.recognized(
        1,
        "a long caption line that a user said out loud " * 3,
        translate_enabled=False,
        send_enabled=True,
    )
    m.sent(1, truncated=True)
    html = render_rows_html(m.rows())
    for width in (330, 500):  # 330 reproduced the old bug; 500 is typical
        lines = _rendered_lines(html, text_width=width)
        offenders = [l for l in lines if "shortened" in l and "fit" not in l]
        assert not offenders, f"status split mid-phrase at {width}px: {offenders!r}"


def test_truncated_color_differs_from_sent():
    m = _model()
    m.recognized(1, "x", translate_enabled=False, send_enabled=True)
    m.sent(1, truncated=True)
    truncated_color = status_markup(m.rows()[0])[1]
    m2 = _model()
    m2.recognized(1, "x", translate_enabled=False, send_enabled=True)
    m2.sent(1, truncated=False)
    sent_color = status_markup(m2.rows()[0])[1]
    assert truncated_color != sent_color  # warn vs good


def test_render_rows_html_uses_a_table_per_entry_with_gutter():
    m = _model()
    m.recognized(1, "hello", translate_enabled=False, send_enabled=True)
    html = render_rows_html(m.rows())
    assert "<table" in html


def test_caption_text_is_not_bold():
    m = _model()
    m.recognized(1, "hello", translate_enabled=False, send_enabled=True)
    html = render_rows_html(m.rows())
    assert "font-weight: 600" not in html
    assert "font-weight:600" not in html
    assert "<b " not in html and "<b>" not in html


def test_translation_line_has_left_rule():
    m = _model()
    m.recognized(1, "hello", translate_enabled=True, send_enabled=True)
    m.translated(1, [("Japanese", "こんにちは")], send_enabled=True)
    html = render_rows_html(m.rows())
    assert "border-left" in html


def test_translation_line_shows_language_label():
    # Regression: render_rows_html iterated (display, text) tuples but only
    # rendered text, dropping which language each line was. With 2-3 targets
    # the log showed unlabeled indented lines with no way to tell them apart.
    m = _model()
    m.recognized(1, "hello", translate_enabled=True, send_enabled=True)
    m.translated(
        1,
        [("Japanese", "こんにちは"), ("Korean", "안녕하세요")],
        send_enabled=True,
    )
    html = render_rows_html(m.rows())
    assert "Japanese" in html
    assert "こんにちは" in html
    assert "Korean" in html
    assert "안녕하세요" in html


def test_translation_line_label_is_escaped():
    m = _model()
    m.recognized(1, "hello", translate_enabled=True, send_enabled=True)
    m.translated(1, [("<b>Lang</b>", "text")], send_enabled=True)
    html = render_rows_html(m.rows())
    assert "<b>Lang</b>" not in html
    assert "&lt;b&gt;Lang&lt;/b&gt;" in html


def test_listening_status_renders_a_muted_marker():
    m = _model()
    m.partial(1, "hel")
    text, color = status_markup(m.rows()[0])
    assert "listening" in text
    assert color == "#98a2b3"  # default muted


def test_status_markup_colors_distinct():
    m = _model()
    m.recognized(1, "x", translate_enabled=True, send_enabled=True)
    translating_color = status_markup(m.rows()[0])[1]
    m.sent(1, truncated=False)
    sent_color = status_markup(m.rows()[0])[1]
    assert translating_color != sent_color


def test_render_rows_html_accepts_theme_colors():
    from vrcc.gui.caption_log import CaptionModel, render_rows_html

    m = CaptionModel(time_label=lambda: "13:42")
    m.recognized(1, "hello", translate_enabled=False, send_enabled=True)
    html = render_rows_html(m.rows(), colors={"text": "#111111", "muted": "#222222",
                                              "accent": "#333333", "good": "#444444",
                                              "warn": "#555555"})
    assert "#111111" in html  # light-theme text color threaded through


def test_empty_state_html_contains_message():
    from vrcc.gui.caption_log import empty_state_html

    html = empty_state_html("Say something, your captions will appear here")
    assert "Say something" in html


def test_empty_state_html_contains_message_and_sub():
    from vrcc.gui.caption_log import empty_state_html

    html = empty_state_html("x", sub="y")
    assert "x" in html
    assert "y" in html


def test_render_rows_html_scales_widths_and_font():
    m = _model()
    m.recognized(1, "hello", translate_enabled=False, send_enabled=True)
    html = render_rows_html(m.rows(), None, 1.2)
    assert 'width="77"' in html  # round(64 * 1.2)
    assert 'width="228"' in html  # round(190 * 1.2)
    assert "font-size:13px" in html  # round(11 * 1.2)


def test_render_rows_html_default_scale_unchanged():
    m = _model()
    m.recognized(1, "hello", translate_enabled=False, send_enabled=True)
    html = render_rows_html(m.rows())
    assert 'width="64"' in html
    assert 'width="190"' in html
    assert "font-size:11px" in html


def test_empty_state_html_scales_font():
    from vrcc.gui.caption_log import empty_state_html

    html = empty_state_html("hi", sub="there", scale=1.2)
    assert "font-size:17px" in html  # round(14 * 1.2) headline
    assert "font-size:14px" in html  # round(12 * 1.2) sub


def test_render_partial_colors_dict_does_not_raise():
    # The documented 5-key override shape must not KeyError on border/bad for a
    # translated + not_sent row (regression for `colors or _DEFAULT_COLORS`).
    m = _model()
    m.recognized(1, "hi", translate_enabled=True, send_enabled=False)
    m.translated(1, [("Japanese", "こんにちは")], send_enabled=False)
    assert m.rows()[0].status == NOT_SENT
    colors = {"text": "#111", "muted": "#222", "accent": "#333",
              "good": "#444", "warn": "#555"}
    html = render_rows_html(m.rows(), colors)  # border + bad come from defaults
    assert "#222" in html


def test_status_markup_partial_colors_dict_does_not_raise():
    m = _model()
    m.recognized(1, "hi", translate_enabled=False, send_enabled=False)  # NOT_SENT
    text, _color = status_markup(m.rows()[0], {"text": "#111"})  # missing "bad"
    assert text == "not sent"
