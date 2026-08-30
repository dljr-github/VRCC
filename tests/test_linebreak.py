"""vrcc.osc.linebreak: script-level predicates and the boundary chooser.

Split from tests/test_chatbox_split.py's scope: these cover script knowledge
in isolation, not how a chatbox part is assembled from it.
"""

from __future__ import annotations

import unicodedata

from vrcc.osc.linebreak import _ends_clause, _legal, choose_cut, is_spaceless, safe_cut

# Same fixture tests/test_chatbox_split.py uses for its Thai case, reused
# here rather than inventing a second one.
_THAI = (
    "สวัสดีครับทุกคนวันนี้อากาศดีมากผมกำลังทดสอบระบบ"
    "คำบรรยายภาษาไทยอยู่ครับหวังว่าจะใช้งานได้ดี"
)


def test_safe_cut_now_respects_sara_am_and_sara_ii():
    # unicodedata.combining() is 0 for both, so the predicate it used to use
    # never moved the cut off either mark.
    for mark, index in (("ั", 2), ("ี", 5)):
        assert _THAI[index] == mark
        cut = safe_cut(_THAI, index)
        assert cut < index
        assert unicodedata.category(_THAI[cut]) not in ("Mn", "Mc")


def test_safe_cut_still_respects_mai_ek():
    # combining class 107: the one class the old predicate already caught.
    index = _THAI.index("่")
    cut = safe_cut(_THAI, index)
    assert cut < index
    assert unicodedata.category(_THAI[cut]) not in ("Mn", "Mc")


def test_safe_cut_falls_back_to_index_when_nudging_would_reach_zero():
    marks = "ัี่"  # nothing but combining marks
    assert safe_cut(marks, 2) == 2


def test_is_spaceless_unchanged_by_the_move():
    assert is_spaceless(_THAI) is True
    assert is_spaceless("これはテストです") is True
    assert is_spaceless("hello world") is False
    assert is_spaceless("안녕 하세요") is False  # Korean separates words


def test_legal_rejects_a_character_forbidden_at_line_start():
    text = "test。ing"
    assert _legal(text, text.index("。")) is False


def test_legal_rejects_a_predecessor_forbidden_at_line_end():
    text = "say「hi」"
    opener = text.index("「")
    assert _legal(text, opener + 1) is False  # 「 must not end a line
    assert _legal(text, opener) is True  # the opener itself may start one


def test_ends_clause_sees_through_a_run_of_closing_brackets():
    text = "彼は言った。」"
    assert _ends_clause(text, len(text)) is True


def test_ends_clause_false_without_a_clause_mark_behind_the_brackets():
    text = "彼は言った」"
    assert _ends_clause(text, len(text)) is False


def test_choose_cut_prefers_the_nearest_clause_boundary_in_window():
    text = "abc。defg、hijk"  # clause boundaries at index 4 and index 9
    assert choose_cut(text, index=8, floor=0, lo=0, hi=len(text) - 1) == 9


def test_choose_cut_falls_back_to_nearest_legal_position():
    text = "abc」def"  # no clause mark anywhere, so no candidate to prefer
    # index 3 (the closer) is illegal to start a line on; the walk lands on
    # the nearest legal position behind it.
    assert choose_cut(text, index=3, floor=0, lo=0, hi=len(text) - 1) == 2


def test_choose_cut_returns_index_unchanged_when_nothing_is_legal():
    text = "ab。cd"
    # Window restricted to the terminator's own position: no clause mark
    # ends there, and a terminator may not itself start a line.
    assert choose_cut(text, index=2, floor=2, lo=2, hi=2) == 2


def test_choose_cut_never_returns_below_floor():
    text = "a」bc"
    # Position 0 ("a") is legal but sits below floor; position 1 ("」") is
    # the only candidate at or above floor, and it is illegal. A search that
    # kept walking past floor would return 0.
    assert _legal(text, 0) is True
    assert choose_cut(text, index=1, floor=1, lo=1, hi=1) == 1


def test_choose_cut_clamps_the_final_position_of_a_terminated_text():
    text = "abcdefg。"
    # The position right after the terminator (len(text)) is a genuine
    # clause boundary, which would swallow the whole text into one slice.
    assert _ends_clause(text, len(text)) is True
    result = choose_cut(text, index=len(text) - 1, floor=0, lo=0, hi=len(text))
    assert result < len(text) - 1


def test_choose_cut_never_offers_the_last_character_as_a_cut():
    text = "何々何々何々。"
    for index in range(len(text)):
        result = choose_cut(text, index=index, floor=0, lo=0, hi=len(text))
        assert result < len(text) - 1
