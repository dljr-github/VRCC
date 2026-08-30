"""vrcc.osc.linebreak: script-level predicates and the boundary chooser.

Split from tests/test_chatbox_split.py's scope: these cover script knowledge
in isolation, not how a chatbox part is assembled from it.
"""

from __future__ import annotations

import unicodedata

from vrcc.osc.linebreak import (
    _BREAK_AFTER,
    _CLOSING,
    _ends_clause,
    _legal,
    choose_cut,
    is_spaceless,
    safe_cut,
)
from vrcc.translate.punctuation import _ALREADY, _CLOSERS

# Same fixture tests/test_chatbox_split.py uses for its Thai case, reused
# here rather than inventing a second one.
_THAI = (
    "สวัสดีครับทุกคนวันนี้อากาศดีมากผมกำลังทดสอบระบบ"
    "คำบรรยายภาษาไทยอยู่ครับหวังว่าจะใช้งานได้ดี"
)


def test_safe_cut_backs_up_past_mai_han_akat_and_sara_ii():
    # Both have unicodedata.combining() == 0, so a check keyed on that alone
    # would leave the cut sitting on the mark itself.
    for mark, index in (("ั", 2), ("ี", 5)):
        assert _THAI[index] == mark
        cut = safe_cut(_THAI, index)
        assert cut < index
        assert unicodedata.category(_THAI[cut]) not in ("Mn", "Mc")


def test_safe_cut_backs_up_past_mai_ek():
    # Combining class 107: unicodedata.combining() alone already identifies
    # this one, unlike MAI HAN-AKAT and SARA II above.
    index = _THAI.index("่")
    cut = safe_cut(_THAI, index)
    assert cut < index
    assert unicodedata.category(_THAI[cut]) not in ("Mn", "Mc")


def test_safe_cut_falls_back_to_index_when_nudging_would_reach_zero():
    marks = "ัี่"  # nothing but combining marks
    assert safe_cut(marks, 2) == 2


def test_is_spaceless_true_for_unspaced_scripts_false_for_spaced_ones():
    assert is_spaceless(_THAI) is True
    assert is_spaceless("\u3053\u308C\u306F\u30C6\u30B9\u30C8\u3067\u3059") is True
    assert is_spaceless("hello world") is False
    # Whitespace makes this False whatever the script. A lone hangul word is
    # spaceless like any other unspaced run, and stays on the word path
    # because it is short, not because this predicate turns it away.
    assert is_spaceless("\uC548\uB155 \uD558\uC138\uC694") is False
    assert is_spaceless("\uC548\uB155") is True


def test_is_spaceless_false_for_private_use_area():
    # U+8C48..U+FAFF once stood in for U+F900..U+FAFF here (an editor or
    # write path silently substituted the NFC-canonical decomposition of
    # U+F900), which widened this range to swallow the Private Use Area.
    # Nothing in vrcc/core/languages.py emits PUA text, so this is a
    # regression guard rather than a shipped-behavior test.
    assert is_spaceless("\ue000" * 80) is False
    assert is_spaceless("\uf8ff" * 80) is False
    assert is_spaceless("\uf900" * 80) is True  # the real range's own start
    assert is_spaceless("\ufaff" * 80) is True  # the real range's own end


def test_legal_rejects_a_character_forbidden_at_line_start():
    text = "test。ing"
    assert _legal(text, text.index("。")) is False


def test_legal_rejects_a_predecessor_forbidden_at_line_end():
    text = "say「hi」"
    opener = text.index("「")
    assert _legal(text, opener + 1) is False  # 「 must not end a line
    assert _legal(text, opener) is True  # the opener itself may start one


def test_legal_rejects_each_thai_sign_that_attaches_backward():
    # U+0E33 SARA AM, U+0E30 SARA A, U+0E46 MAIYAMOK, U+0E2F PAIYANNOI are
    # category Lo/Lm, not Mn/Mc, so only _NO_START membership catches them.
    # Dropping any one of the four from _NO_START would still leave this red.
    for mark in ("\u0e33", "\u0e30", "\u0e46", "\u0e2f"):
        text = "\u0e01" + mark
        assert _legal(text, 1) is False


def test_legal_rejects_a_position_after_each_thai_preposed_vowel():
    # U+0E40 through U+0E44 are written before the consonant they voice, so
    # the position right after one is illegal to start a line on.
    for mark in ("\u0e40", "\u0e41", "\u0e42", "\u0e43", "\u0e44"):
        text = mark + "\u0e01"
        assert _legal(text, 1) is False


def test_legal_rejects_a_cut_between_two_ascii_alphanumerics():
    # "without" -> "witho" / "ut" is the shape this rule exists to stop: a
    # code-switched translation's Latin word severed by the character path.
    text = "without"
    for i in range(1, len(text)):
        assert _legal(text, i) is False, f"index {i} split an ASCII word"


def test_legal_allows_a_cut_at_an_ascii_word_boundary():
    # The rule reads only text[i - 1] and text[i]; a space between two
    # words is not "one word" and must not be caught by it.
    text = "cat dog"
    assert _legal(text, text.index("d")) is True


def test_legal_ascii_word_rule_does_not_reach_across_punctuation():
    # A hyphen breaks the ASCII-alnum adjacency the rule keys on, so the
    # letter right after one is still a legal cut, same as any other
    # non-alnum predecessor.
    text = "cat-dog"
    assert _legal(text, text.index("d")) is True


def test_choose_cut_falls_back_to_the_raw_index_for_an_all_alnum_run():
    # A pathological over-long alphanumeric run has no legal position
    # anywhere in [floor, hi]: every interior index now fails the new rule,
    # same as a run of nothing but kinsoku-illegal characters already did.
    # Confirmed here rather than assumed: this must return the plain
    # clamped index, not stall or hand back something out of range.
    text = "9" * 40
    result = choose_cut(text, index=20, floor=5, lo=10, hi=30)
    assert result == 20
    assert all(not _legal(text, i) for i in range(6, 31))  # 0 is the only legal index


def test_ends_clause_sees_through_a_run_of_closing_brackets():
    text = "彼は言った。」"
    assert _ends_clause(text, len(text)) is True


def test_ends_clause_sees_through_a_run_of_two_closers():
    text = "text\u3002\u300d\uff09"  # period, then two different closers
    assert _ends_clause(text, len(text)) is True


def test_ends_clause_sees_through_a_run_of_three_closers():
    text = "text\u3002\u300d\uff09\u300f"  # period, then three closers
    assert _ends_clause(text, len(text)) is True


def test_ends_clause_false_without_a_clause_mark_behind_the_brackets():
    text = "彼は言った」"
    assert _ends_clause(text, len(text)) is False


def test_closing_matches_punctuation_closers():
    # A test may cross the package boundary the module itself deliberately
    # does not: if punctuation._CLOSERS ever gains a bracket, this must fail
    # rather than let the two definitions drift apart silently.
    assert _CLOSING == _CLOSERS


def test_already_is_a_subset_of_break_after():
    # punctuation.normalize refuses to convert a mark that already sits
    # next to one of _ALREADY; the line breaker welcomes a cut right after
    # one of _BREAK_AFTER. A terminator the normalizer respects but the
    # line breaker does not know is a sentence end no slice can snap to.
    assert set(_ALREADY) <= set(_BREAK_AFTER)


def test_choose_cut_prefers_the_nearest_clause_boundary_in_window():
    text = "abc。defg、hijk"  # clause boundaries at index 4 and index 9
    assert choose_cut(text, index=8, floor=0, lo=0, hi=len(text) - 1) == 9


def test_choose_cut_skips_a_clause_boundary_that_may_not_start_a_line():
    # A quoted terminator sits behind its closer, and _ends_clause looks
    # straight through the closer to reach it. That makes the closer's own
    # position a clause boundary, and a slice starting there opens on a
    # closer with nothing it closed. One position further out is both a
    # clause end and legal, so the window is not short of candidates.
    text = (
        "\u3042\u3044\u3046\u300C\u3059\u3054\u3044\uFF01"
        "\u300D\u3068\u8A00\u3063\u305F\u3002\u306F\u3044"
    )
    assert text[8] == "\u300D"
    assert _ends_clause(text, 8) is True and _legal(text, 8) is False
    assert _ends_clause(text, 9) is True and _legal(text, 9) is True
    assert choose_cut(text, index=8, floor=1, lo=6, hi=11) == 9


def test_choose_cut_never_starts_a_slice_on_a_closing_bracket():
    # Every clause mark here is quoted, so every position the clause branch
    # reaches through a closer is one _legal forbids. The window always holds
    # a legal candidate, which keeps that branch rather than the fallback
    # walk in charge of the answer.
    text = "\u300C\u306F\u3044\u3002\u300D" * 8
    for index in range(2, len(text) - 2):
        cut = choose_cut(text, index, 1, index - 3, index + 3)
        assert text[cut] not in _CLOSING, f"index {index} cut onto a closer"


def test_choose_cut_falls_back_to_nearest_legal_position():
    # CJK filler, not "abc"/"def": an ASCII word on either side of the closer
    # would also engage the ASCII-word rule and land this on index 0
    # instead, which is a different test. No clause mark anywhere, so no
    # candidate to prefer.
    # Escaped, not literal: U+3067, U+305A and U+304C are precomposed voiced
    # kana with canonical decompositions, so a normalizing write path could
    # substitute base plus U+3099 and leave this test green.
    text = "\u3042\u3044\u3046\u300D\u3067\u305A\u304C"
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
