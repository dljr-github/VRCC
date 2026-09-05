"""vrcc.osc.linebreak: script-level predicates and the boundary chooser.

Split from tests/test_chatbox_split.py's scope: these cover script knowledge
in isolation, not how a chatbox part is assembled from it.
"""

from __future__ import annotations

import unicodedata

from vrcc.core.charclass import is_closer, is_cjk
from vrcc.osc.linebreak import (
    _BREAK_AFTER,
    _ends_clause,
    _legal,
    choose_cut,
    is_spaceless,
    safe_cut,
    slice_cut,
)
from vrcc.translate.punctuation import _ALREADY

# Same fixture tests/test_chatbox_split.py uses for its Thai case, reused
# here rather than inventing a second one.
_THAI = (
    "\u0E2A\u0E27\u0E31\u0E2A\u0E14\u0E35\u0E04\u0E23\u0E31\u0E1A\u0E17\u0E38\u0E01\u0E04\u0E19\u0E27\u0E31\u0E19\u0E19\u0E35\u0E49\u0E2D\u0E32\u0E01\u0E32\u0E28\u0E14\u0E35\u0E21\u0E32\u0E01\u0E1C\u0E21\u0E01\u0E33\u0E25\u0E31\u0E07\u0E17\u0E14\u0E2A\u0E2D\u0E1A\u0E23\u0E30\u0E1A\u0E1A"
    "\u0E04\u0E33\u0E1A\u0E23\u0E23\u0E22\u0E32\u0E22\u0E20\u0E32\u0E29\u0E32\u0E44\u0E17\u0E22\u0E2D\u0E22\u0E39\u0E48\u0E04\u0E23\u0E31\u0E1A\u0E2B\u0E27\u0E31\u0E07\u0E27\u0E48\u0E32\u0E08\u0E30\u0E43\u0E0A\u0E49\u0E07\u0E32\u0E19\u0E44\u0E14\u0E49\u0E14\u0E35"
)


def test_safe_cut_backs_up_past_mai_han_akat_and_sara_ii():
    # Both have unicodedata.combining() == 0, so a check keyed on that alone
    # would leave the cut sitting on the mark itself.
    for mark, index in (("\u0E31", 2), ("\u0E35", 5)):
        assert _THAI[index] == mark
        cut = safe_cut(_THAI, index)
        assert cut < index
        assert unicodedata.category(_THAI[cut]) not in ("Mn", "Mc")


def test_safe_cut_backs_up_past_mai_ek():
    # Combining class 107: unicodedata.combining() alone already identifies
    # this one, unlike MAI HAN-AKAT and SARA II above.
    index = _THAI.index("\u0E48")
    cut = safe_cut(_THAI, index)
    assert cut < index
    assert unicodedata.category(_THAI[cut]) not in ("Mn", "Mc")


def test_safe_cut_falls_back_to_index_when_nudging_would_reach_zero():
    marks = "\u0E31\u0E35\u0E48"  # nothing but combining marks
    assert safe_cut(marks, 2) == 2


def test_safe_cut_backs_up_past_a_thai_sign_filed_as_a_letter():
    # SARA AM is category Lo, so a test keyed on combining marks alone leaves
    # the cut right in front of it and the next chunk opens on a dependent
    # vowel sign.
    text = "\u0E01\u0E01\u0E01\u0E17\u0E33\u0E01\u0E01\u0E01"
    assert text[4] == "\u0E33"
    assert safe_cut(text, 4) == 3


def test_slice_cut_backs_up_to_the_start_of_an_ascii_word():
    # The character path exists for spaceless scripts; a Latin island inside
    # one is still a word, and a plain slice boundary must not sever it.
    text = "中" * 5 + "VRChat" + "中" * 5
    assert slice_cut(text, 8) == 5
    assert slice_cut(text, 11) == 11  # the boundary right after the word


def test_slice_cut_falls_back_to_index_when_the_word_opens_the_text():
    text = "abcdefghij" + "中" * 5
    assert slice_cut(text, 4) == 4


def test_is_cjk_excludes_hangul_compatibility_jamo_and_hangul_syllables():
    # HANGUL LETTER KIYEOK sits inside the old single 0x3001-0x9FFF sweep,
    # and HANGUL SYLLABLE GA sits outside it: the same script was half in
    # and half out of the predicate. Both are Korean, so both are False now
    # that Hangul Compatibility Jamo is carved out.
    assert is_cjk("\u3131") is False  # HANGUL LETTER KIYEOK
    assert is_cjk("\uAC00") is False  # HANGUL SYLLABLE GA


def test_is_cjk_pins_the_bopomofo_hangul_jamo_kanbun_boundary():
    # Bopomofo (U+3100-U+312F) stays in; Hangul Compatibility Jamo
    # (U+3130-U+318F) is the carved-out gap; Kanbun (U+3190-U+319F) resumes
    # right after it. U+3130 and U+318F are themselves unassigned, but the
    # range check does not care, so they still pin the gap's own edges.
    assert is_cjk("\u312F") is True  # BOPOMOFO LETTER AH, last of that block
    assert is_cjk("\u3130") is False  # gap's own start
    assert is_cjk("\u318F") is False  # gap's own end
    assert is_cjk("\u3190") is True  # IDEOGRAPHIC ANNOTATION LINKING MARK


def test_is_cjk_excludes_enclosed_cjk_letters_and_months():
    # Enclosed CJK Letters and Months (U+3200-U+32FF) mixes parenthesized and
    # circled Hangul (U+3200 PARENTHESIZED HANGUL KIYEOK, U+3260 CIRCLED
    # HANGUL KIYEOK) into the same block as circled ideographs and squared
    # era names, so keeping any of it would reintroduce the same defect this
    # fix removes. The whole block is out, including its Japanese-only
    # SQUARE ERA NAME REIWA at the top end.
    assert is_cjk("\u3200") is False  # PARENTHESIZED HANGUL KIYEOK
    assert is_cjk("\u3260") is False  # CIRCLED HANGUL KIYEOK
    assert is_cjk("\u32FF") is False  # SQUARE ERA NAME REIWA, last of the block


def test_is_cjk_pins_the_katakana_phonetic_ext_to_cjk_compatibility_boundary():
    # Katakana Phonetic Extensions (U+31F0-U+31FF) stays in; Enclosed CJK
    # Letters and Months is the gap; CJK Compatibility (U+3300-U+33FF)
    # resumes right after it. That block carries no Hangul (its squared and
    # circled forms are Latin units and Japanese words, e.g. SQUARE APAATO),
    # so it stays in whole.
    assert is_cjk("\u31FF") is True  # KATAKANA LETTER SMALL RO, last of that block
    assert is_cjk("\u3300") is True  # SQUARE APAATO, first of CJK Compatibility


def test_is_spaceless_true_for_unspaced_scripts_false_for_spaced_ones():
    assert is_spaceless(_THAI) is True
    assert is_spaceless("\u3053\u308C\u306F\u30C6\u30B9\u30C8\u3067\u3059") is True
    assert is_spaceless("hello world") is False
    # Whitespace makes this False whatever the script, and hangul is not an
    # unspaced script at all: Korean stays on the word path however long a
    # token runs, so a lone hangul word is not spaceless either.
    assert is_spaceless("\uC548\uB155 \uD558\uC138\uC694") is False
    assert is_spaceless("\uC548\uB155") is False


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


def test_is_spaceless_true_for_extension_b_and_later():
    # The ranges are vrcc.core.charclass's, shared with the normalizer: a run
    # normalize treats as CJK reaches the character path here too, or it
    # travels whole in one slice and the other parts carry none of it.
    assert is_spaceless("\U00020000" * 5) is True


def test_legal_rejects_a_character_forbidden_at_line_start():
    text = "test。ing"
    assert _legal(text, text.index("。")) is False


def test_legal_rejects_a_predecessor_forbidden_at_line_end():
    text = "say「hi」"
    opener = text.index("「")
    assert _legal(text, opener + 1) is False  # 「 must not end a line
    assert _legal(text, opener) is True  # the opener itself may start one


def test_legal_rejects_halfwidth_middle_dot_at_line_start():
    # The halfwidth form separates the words on either side of it the same
    # way its fullwidth counterpart U+30FB does. Escaped: a write path could
    # silently widen it to U+30FB and leave this green for the wrong reason.
    text = "a\uFF65b"
    assert _legal(text, 1) is False


def test_legal_rejects_small_ka_and_ke_at_line_start():
    # The counter and place-name kana (three months, Kasumigaseki) are small
    # forms like the small tsu, and JIS X 4051 keeps them off a line start.
    for small in ("\u30f5", "\u30f6", "\u3095", "\u3096"):
        assert _legal("3" + small + "\u6708", 1) is False


def test_legal_rejects_the_decomposed_sara_am_tail():
    # U+0E33 SARA AM decomposes only under NFKC (its mapping is a
    # compatibility one), so decoded text can carry U+0E4D U+0E32 and the
    # tail has to be refused on its own. Escaped, not the precomposed
    # literal: a normalizing write path could collapse the two back into
    # U+0E33 and leave this test green for the wrong reason.
    text = "\u0e01\u0e4d\u0e32"
    assert _legal(text, 2) is False


def test_legal_rejects_each_thai_sign_that_attaches_backward():
    # SARA AM, SARA A, SARA AA, LAKKHANGYAO, MAIYAMOK and PAIYANNOI are
    # category Lo/Lm, not Mn/Mc, so only the following-sign ranges catch
    # them; any one missing from those ranges would leave this red.
    for mark in ("\u0e33", "\u0e30", "\u0e32", "\u0e45", "\u0e46", "\u0e2f"):
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
    # anywhere in [floor, hi]: every interior index splits the run, the same
    # dead end as a run of nothing but kinsoku-illegal characters. This must
    # return the plain clamped index, not stall or hand back something out
    # of range.
    text = "9" * 40
    result = choose_cut(text, index=20, floor=5, lo=10, hi=30)
    assert result == 20
    assert all(not _legal(text, i) for i in range(6, 31))  # 0 is the only legal index


def test_choose_cut_backward_walk_stays_within_lo_not_floor():
    # A walk that ran all the way to floor could find a legal position far
    # below lo and hand back a boundary outside the caller's own window.
    text = "あ" + "9" * 39  # only indices 0 and 1 are ever legal
    result = choose_cut(text, index=15, floor=0, lo=5, hi=20)
    assert result == 15
    assert 5 <= result <= 20


def test_choose_cut_forward_walk_stays_within_lo_not_floor():
    # The forward walk used to scan from start + 1 with no lower clamp of
    # its own, so when index sat below lo it could hand back a position
    # below both lo and floor. index=4 is below lo=5 here, and the first
    # legal position the unclamped scan reached was index 6, below floor=7.
    text = "\u3046\u3046\u304a\u3067\u3046\u3001\u304a\u3001\u3002"
    result = choose_cut(text, index=4, floor=7, lo=5, hi=7)
    assert result == 7
    assert 5 <= result <= 7


def test_choose_cut_forward_walk_never_returns_below_lo():
    # A single-position window with index sitting below it on both sides:
    # the unclamped forward scan reached index 1, two positions below the
    # lo=hi=3 window and below floor=3 as well.
    text = "\u3042\u3044\u3046\u3048\u304a"
    assert choose_cut(text, index=0, floor=3, lo=3, hi=3) == 3


def test_choose_cut_fallback_nudges_off_a_combining_mark():
    # The fallback is the one return no rule has vetted; safe_cut is what
    # keeps it off a mark orphaned from its base.
    text = "(" + "a" + "\u0301" * 25
    result = choose_cut(text, index=15, floor=1, lo=1, hi=1)
    assert result == 1
    assert unicodedata.category(text[result]) not in ("Mn", "Mc")


def test_choose_cut_fallback_never_returns_below_lo():
    # Backing off a mark cannot buy a position outside the window: the
    # caller's size bound outranks the attachment rule once a run of marks
    # fills the whole window.
    text = "(" + "a" + "\u0301" * 25
    assert choose_cut(text, index=15, floor=1, lo=5, hi=20) == 5


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


def test_break_after_leaves_the_ascii_terminators_out():
    # normalize never converts "!" or "?", so they sit inside Japanese with
    # an ASCII space after them or inside a URL, and a cut welcomed after
    # one opens the next slice on that space or inside that URL.
    assert "!" not in _BREAK_AFTER and "?" not in _BREAK_AFTER
    assert "\uFF01" in _BREAK_AFTER and "\uFF1F" in _BREAK_AFTER


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
        assert not is_closer(text[cut]), f"index {index} cut onto a closer"


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
