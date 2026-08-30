"""How an over-limit message is carved into parts.

Split from ``test_chatbox_overflow`` for the file-length cap. These cover which
part each language's share lands in, which is a different question from whether
a message overflows at all.
"""

from __future__ import annotations

from tests.test_chatbox_overflow import make_cfg
from vrcc.osc.chatbox import fit_message
from vrcc.osc.chatbox_format import CHATBOX_LIMIT, _balanced_slices, _join, _settle

# The regression report's Japanese translation, split into its three expected
# slices. Escaped per the branch-wide ruling: a raw glyph here survived one
# NFC-normalizing write path already (see tests/test_linebreak.py). Extracted
# character for character from design.md's own pinned example.
_JA_LINE_1 = "\u79c1\u306e\u53cb\u9054\u304c\u6559\u3048\u3066\u304f\u308c\u305f\u3093\u3060\u3051\u3069\u3001"
_JA_LINE_2 = (
    "\u99c5\u306e\u8fd1\u304f\u306e\u65b0\u3057\u3044\u30ab\u30d5\u30a7\u306f"
    "\u671d\u516b\u6642\u306b\u958b\u304f\u3089\u3057\u3044\u306e\u3067\u3001"
)
_JA_LINE_3 = (
    "\u660e\u65e5\u306e\u671d\u3054\u306f\u3093\u306e\u5f8c\u3067\u4e00\u7dd2"
    "\u306b\u884c\u3063\u3066\u307f\u307e\u305b\u3093\u304b\u3068\u601d\u3063"
    "\u3066\u3044\u307e\u3059\u3002"
)
_JA_TRANSLATION = _JA_LINE_1 + _JA_LINE_2 + _JA_LINE_3


def test_fit_message_split_carves_a_thai_translation_across_every_part():
    """Thai is written without spaces, so .split() returns one token that the
    word packer accepts whole and drops into a single slice, leaving every
    other part without a word of it. The same reasoning as CJK, which the
    character branch already handles."""
    cfg = make_cfg(overflow="split")
    original = " ".join(f"word{i}" for i in range(55))
    thai = (
        "สวัสดีครับทุกคนวันนี้อากาศดีมากผมกำลังทดสอบระบบ"
        "คำบรรยายภาษาไทยอยู่ครับหวังว่าจะใช้งานได้ดี"
    )

    parts = fit_message(original, [("TH", thai)], cfg)

    assert len(parts) >= 2
    assert all(len(part) <= CHATBOX_LIMIT for part in parts)
    assert all(any(ch in part for ch in thai) for part in parts), (
        "every part must carry a share of the Thai translation"
    )
    carried = "".join(part.split("\n")[-1] for part in parts)
    assert carried == thai, "the slices must reproduce the translation exactly"


def test_fit_message_split_leaves_a_sliced_translation_in_the_final_part():
    """Parts drain one at a time and a new utterance clears the queue, so the
    last part is the one still on screen once the user stops talking. A
    translation with fewer words than parts cannot both lead and persist, and
    persisting is the half worth having: a reader who is still being talked
    over never reaches the later parts either way. A translation long enough to
    fill every slice does both, which is what
    test_fit_message_split_puts_an_unrepeatable_translation_in_the_first_part
    pins."""
    cfg = make_cfg(overflow="split")
    original = " ".join(f"word{i}" for i in range(50))
    translation = " ".join("T" * 40 for _ in range(3))  # too long to repeat

    parts = fit_message(original, [("X", translation)], cfg)

    assert len(parts) >= 3
    assert all(len(part) <= CHATBOX_LIMIT for part in parts)
    assert "T" * 40 not in parts[0], "a 3-word translation cannot fill 4 parts"
    assert "T" * 40 in parts[-1], "the persisting part must carry the translation"


def test_end_anchoring_never_costs_a_repeat():
    """Anchoring changes which original slice a translation shares a part with,
    so a part count that only fits because of it would be chosen over a larger
    one that repeats the translation outright. Repeating measured better, so
    the part count is judged before anchoring is applied."""
    cfg = make_cfg(overflow="split")
    original = " ".join(f"word{i}" for i in range(60))
    translation = "これはテストです"

    parts = fit_message(original, [("JP", translation)], cfg)

    assert all(translation in part for part in parts), (
        "a translation short enough to repeat must still be in every part"
    )


# -- snap: clause-mark snapping, wired in after the part count is settled ----


def test_fit_message_split_pins_the_three_clause_cuts_end_to_end():
    """The bug report this task exists for: a Japanese translation with no
    ASCII space anywhere in it (`_JA_TRANSLATION`), cut mid-word by plain
    ceil-division. Each expected line is `design.md`'s own pinned example,
    reproduced end to end through `fit_message`. The original is long enough,
    and `_MAX_REPEAT_PARTS` low enough, that n=3 is the first count the search
    tries and the translation is too long to repeat, so it is always sliced."""
    original = " ".join(f"word{i}" for i in range(44))
    cfg = make_cfg(overflow="split")

    parts = fit_message(original, [("JA", _JA_TRANSLATION)], cfg)

    assert len(parts) == 3
    assert all(len(part) <= CHATBOX_LIMIT for part in parts)
    translation_lines = [part.split("\n")[-1] for part in parts]
    assert translation_lines == [_JA_LINE_1, _JA_LINE_2, _JA_LINE_3]
    assert "".join(translation_lines) == _JA_TRANSLATION


def test_balanced_slices_snap_true_lands_on_the_clause_marks():
    # Direct check of the snapped character path, isolated from the n-search
    # and from _settle's fallback logic that the end-to-end test above also
    # exercises.
    slices = _balanced_slices(_JA_TRANSLATION, 3, CHATBOX_LIMIT, snap=True)
    assert slices == [_JA_LINE_1, _JA_LINE_2, _JA_LINE_3]


def test_balanced_slices_snap_false_matches_todays_ceil_division():
    # size = ceil(69 / 3) = 23, with no combining mark to nudge off of, so
    # today's plain boundaries are 23 and 46, unrelated to the clause marks.
    default = _balanced_slices(_JA_TRANSLATION, 3, CHATBOX_LIMIT)
    explicit_false = _balanced_slices(_JA_TRANSLATION, 3, CHATBOX_LIMIT, snap=False)
    expected = [_JA_TRANSLATION[0:23], _JA_TRANSLATION[23:46], _JA_TRANSLATION[46:69]]
    assert default == explicit_false == expected
    assert [len(s) for s in expected] == [23, 23, 23]


def test_snap_leaves_unmarked_chinese_at_the_ceil_division_floor():
    """No clause mark exists anywhere in the text, so the fallback legal-
    position walk lands on the same index the plain ceil-division cut would
    have chosen. `[19, 19, 17]` is the design's own pinned floor: without a
    dictionary there is nothing better to know."""
    text = "\u4e2d" * 55
    snapped = _balanced_slices(text, 3, CHATBOX_LIMIT, snap=True)
    assert [len(s) for s in snapped] == [19, 19, 17]
    assert snapped == _balanced_slices(text, 3, CHATBOX_LIMIT, snap=False)


def test_snap_never_produces_an_empty_slice_for_terminated_text():
    # _JA_TRANSLATION ends in U+3002, the case choose_cut's `len(text) - 2`
    # ceiling exists for: a naive `len(text) - 1` ceiling would let the
    # position right after that terminator win a window near the end, and one
    # interior slice would swallow the rest.
    #
    # n starts at 3, not 2: at n=2, limit // n (72) exceeds len(_JA_TRANSLATION)
    # (69), so _balanced_slices' own spaceless-routing guard sends the whole
    # text down the word path as a single over-long token instead of the
    # character path snap touches. That routing threshold is a pre-existing
    # question this task does not change.
    for n in range(3, 6):
        slices = _balanced_slices(_JA_TRANSLATION, n, CHATBOX_LIMIT, snap=True)
        assert len(slices) == n
        assert all(slices), f"n={n} produced an empty slice: {slices!r}"
        assert "".join(slices) == _JA_TRANSLATION


def test_concatenation_reproduces_the_text_for_both_values_of_snap():
    texts = ["\u4e2d" * 55, _JA_TRANSLATION]
    for text in texts:
        for n in (2, 3, 4, 5):
            for snap in (False, True):
                slices = _balanced_slices(text, n, CHATBOX_LIMIT, snap=snap)
                assert len(slices) == n
                assert "".join(slices) == text


def test_settle_discards_a_snap_that_would_push_a_part_over_the_limit():
    """Built deliberately: the only clause mark in the snap window sits at
    the far edge of it, so the snapped cut lands at index 210 on a message
    with 2 slices of ~140 chars apiece, one 210 chars long. `_settle` must
    fall back to the plain ceil-division split rather than ship the
    oversized part."""
    filler = "\u4e2d"
    text = filler * 209 + "\u3002" + filler * 70
    assert len(text) == 280
    cfg = make_cfg(overflow="split", include_original=False)
    texts = [text]
    translated = [True]

    parts = _join(texts, translated, set(), 2, cfg)
    assert all(len(part) <= CHATBOX_LIMIT for part in parts)

    settled = _settle(texts, translated, set(), 2, cfg, parts)

    assert all(len(part) <= CHATBOX_LIMIT for part in settled)
    assert settled == parts


def test_snap_never_touches_the_word_path():
    # snap only changes the character ceil-division path; a Latin text with
    # every word under the limit never leaves the word path.
    text = " ".join(f"word{i}" for i in range(30))
    assert _balanced_slices(text, 3, CHATBOX_LIMIT, snap=True) == _balanced_slices(
        text, 3, CHATBOX_LIMIT, snap=False
    )


def test_fit_message_split_pins_a_latin_only_conversation_unchanged():
    # Pinned against today's output: wiring snap through _join/_settle must
    # not perturb a conversation that never reaches the character path.
    cfg = make_cfg(overflow="split")
    original = " ".join(f"word{i}" for i in range(50))
    translation = " ".join(f"mot{i}" for i in range(50))

    parts = fit_message(original, [("FR", translation)], cfg)

    assert parts == [
        "word0 word1 word2 word3 word4 word5 word6 word7 word8 word9 word10\n"
        "mot0 mot1 mot2 mot3 mot4 mot5 mot6 mot7 mot8 mot9 mot10",
        "word11 word12 word13 word14 word15 word16 word17 word18 word19 word20\n"
        "mot11 mot12 mot13 mot14 mot15 mot16 mot17 mot18 mot19 mot20",
        "word21 word22 word23 word24 word25 word26 word27 word28 word29 word30\n"
        "mot21 mot22 mot23 mot24 mot25 mot26 mot27 mot28 mot29 mot30",
        "word31 word32 word33 word34 word35 word36 word37 word38 word39 word40\n"
        "mot31 mot32 mot33 mot34 mot35 mot36 mot37 mot38 mot39 mot40",
        "word41 word42 word43 word44 word45 word46 word47 word48 word49\n"
        "mot41 mot42 mot43 mot44 mot45 mot46 mot47 mot48 mot49",
    ]
