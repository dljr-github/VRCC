"""How an over-limit message is carved into parts.

Split from ``test_chatbox_overflow`` for the file-length cap. These cover which
part each language's share lands in, which is a different question from whether
a message overflows at all.
"""

from __future__ import annotations

import random

from tests.test_chatbox_overflow import make_cfg
from vrcc.osc.chatbox import fit_message
from vrcc.osc.chatbox_slice import CHATBOX_LIMIT, _balanced_slices, _join, _settle

# The regression report's Japanese translation, split into its three expected
# slices. Escaped: a raw glyph can be recomposed by a normalizing write path
# where an escape cannot (see tests/test_linebreak.py). Pinned line by line
# from the report's own example.
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
    ceil-division. Each expected line is the report's own pinned example,
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
    # the plain boundaries are 23 and 46, unrelated to the clause marks.
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
    # character path snap touches. That routing threshold is a separate
    # question from snapping.
    for n in range(3, 6):
        slices = _balanced_slices(_JA_TRANSLATION, n, CHATBOX_LIMIT, snap=True)
        assert len(slices) == n
        assert all(slices), f"n={n} produced an empty slice: {slices!r}"
        assert "".join(slices) == _JA_TRANSLATION


def test_concatenation_reproduces_the_text_for_both_values_of_snap():
    """Wide enough to catch a regression in a script this suite would
    otherwise never exercise. `snap` only touches the character path
    (`_balanced_slices`' own routing decides which path a text takes), so
    the two script families below need two different reconstructions: the
    character path promises exact "".join() reconstruction, but the word
    path rejoins on whole words with a single space (see `_balanced_slices`'
    own docstring), since blank slices from a short translation collapse
    when concatenated raw -- `test_snap_never_touches_the_word_path`
    already pins that `snap` cannot change which one applies.
    """
    # Spaceless scripts, plus a plain unbreakable run (a single token
    # with no spaces at all, over `limit`, so it fails the word path's
    # own eligibility check regardless of script). The first two are
    # only on the character path from n=3 on: at n=2 the same routing
    # threshold above sends them down the word path instead, where raw
    # concatenation still holds because each lands whole in one slot.
    spaceless_texts = [
        "\u4e2d" * 55,  # CJK, no clause marks: the ceil-division floor
        _JA_TRANSLATION,  # CJK with clause marks: the regression text
        (  # Thai: the suite's one script with combining marks AND no spaces
            "\u0e2a\u0e27\u0e31\u0e2a\u0e14\u0e35\u0e04\u0e23\u0e31\u0e1a"
            "\u0e17\u0e38\u0e01\u0e04\u0e19\u0e27\u0e31\u0e19\u0e19\u0e35"
            "\u0e49\u0e2d\u0e32\u0e01\u0e32\u0e28\u0e14\u0e35\u0e21\u0e32"
            "\u0e01\u0e1c\u0e21\u0e01\u0e33\u0e25\u0e31\u0e07\u0e17\u0e14"
            "\u0e2a\u0e2d\u0e1a\u0e23\u0e30\u0e1a\u0e1a\u0e04\u0e33\u0e1a"
            "\u0e23\u0e23\u0e22\u0e32\u0e22\u0e20\u0e32\u0e29\u0e32\u0e44"
            "\u0e17\u0e22\u0e2d\u0e22\u0e39\u0e48\u0e04\u0e23\u0e31\u0e1a"
            "\u0e2b\u0e27\u0e31\u0e07\u0e27\u0e48\u0e32\u0e08\u0e30\u0e43"
            "\u0e0a\u0e49\u0e07\u0e32\u0e19\u0e44\u0e14\u0e49\u0e14\u0e35"
        ),
        "9" * 200,  # unbreakable run: not a spaceless script, one long token
    ]
    for text in spaceless_texts:
        for n in (2, 3, 4, 5):
            for snap in (False, True):
                slices = _balanced_slices(text, n, CHATBOX_LIMIT, snap=snap)
                assert len(slices) == n
                assert "".join(slices) == text

    # Word path: ordinary spaced text, and a script mix. snap has nothing to
    # do here (see `test_snap_never_touches_the_word_path`), but the
    # reconstruction is still worth pinning across scripts and n.
    word_path_texts = [
        " ".join(f"word{i}" for i in range(30)),  # Latin
        "hello " + "\u4e2d" * 20 + " world " + "\u65e5" * 20 + " test " + "\u4e2d" * 20,
    ]
    for text in word_path_texts:
        for n in (2, 3, 4, 5):
            for snap in (False, True):
                slices = _balanced_slices(text, n, CHATBOX_LIMIT, snap=snap)
                assert len(slices) == n
                assert " ".join(s for s in slices if s) == text


def test_snap_does_not_collapse_a_slice_when_the_window_repeats_the_prior_cut():
    """Minimal reproduction of a floor-inclusive collision: the only clause
    mark sits at index 26, so bounds[1] lands right after it, at 27, and
    bounds[2]'s window (27 to 45) has that same index 27 as its only
    candidate. Passing bounds[-1] as choose_cut's floor lets it hand that
    index straight back, since floor is an inclusive candidate there,
    collapsing the second slice to nothing; bounds[-1] + 1 is what
    `_balanced_slices` passes instead, forcing the search past it."""
    text = "\u4e2d" * 26 + "\u3002" + "\u4e2d" * 25
    assert len(text) == 52

    slices = _balanced_slices(text, 3, CHATBOX_LIMIT, snap=True)

    assert len(slices) == 3
    assert all(slices), f"a slice collapsed to empty: {slices!r}"
    assert "".join(slices) == text


def test_settle_rejects_a_candidate_arrangement_that_drops_a_part():
    """`_join` drops any slot whose pieces are all empty. A 2-word
    translation at n=4 has two blank slots regardless of which end
    `anchored` puts them on, and with no other text to cover for them (only
    one translation, `include_original=False`), all three of `_settle`'s
    attempts come back 2 parts long. `_settle` must not ship a 2-part
    result in place of the 4 parts it was handed, even though each of
    those 2 parts individually fits `CHATBOX_LIMIT` with room to spare:
    fitting the limit is not the same question as keeping every part."""
    cfg = make_cfg(overflow="split", include_original=False)
    texts = ["alpha beta"]
    translated = [True]
    # Stands in for whatever fit_message already committed to sending at
    # n=4; _settle only inspects its length and that each entry already
    # fits, not its provenance.
    parts = ["placeholder0", "placeholder1", "placeholder2", "placeholder3"]

    settled = _settle(texts, translated, set(), 4, cfg, parts)

    assert settled == parts


def test_settle_falls_back_to_parts_when_every_arrangement_overflows():
    """Built so all three of `_settle`'s attempts overflow `CHATBOX_LIMIT`,
    unlike test_settle_discards_a_snap_that_would_push_a_part_over_the_limit
    (which succeeds on the second attempt): this exercises the final
    `return parts` line instead. `original`'s word-path split puts most of
    its length in the last slot (a giant trailing word the greedy packer
    saves for the end); pairing that with the translation `A` (anchored
    there) or the CJK translation `B`'s snap-enlarged last slot (or both)
    all push that one part past the limit, while the plain, unanchored,
    unsnapped baseline does not."""
    cfg = make_cfg(overflow="split", include_original=True, translation_separator="\n")
    original = " ".join([f"w{i}" for i in range(20)] + ["G" * 60])
    a_translation = "X" * 30
    b_translation = "\u4e2d" * 176 + "\u3002" + "\u4e2d" * 103
    texts = [original, a_translation, b_translation]
    translated = [False, True, True]
    n = 4

    baseline = _join(texts, translated, set(), n, cfg)
    assert len(baseline) == n
    assert all(len(part) <= CHATBOX_LIMIT for part in baseline)
    for anchored, snap in ((True, True), (True, False), (False, True)):
        candidate = _join(texts, translated, set(), n, cfg, anchored=anchored, snap=snap)
        assert not (
            len(candidate) == len(baseline)
            and all(len(part) <= CHATBOX_LIMIT for part in candidate)
        ), (anchored, snap, [len(p) for p in candidate])

    settled = _settle(texts, translated, set(), n, cfg, baseline)

    assert settled == baseline


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


def _clause_marked_text(rng: random.Random, length: int) -> str:
    """CJK filler `length` characters long, with a clause mark planted every
    3 to 30 percent of it: the shape a real captioned sentence breaks into,
    for `test_snap_keeps_interior_slices_within_half_to_double_share`."""
    out: list[str] = []
    total = 0
    while total < length:
        run = min(max(1, round(rng.uniform(0.03, 0.30) * length)), length - total)
        out.append("中" * run)
        total += run
        if total < length:
            out.append("。")
            total += 1
    return "".join(out)


def test_snap_keeps_interior_slices_within_half_to_double_share():
    """`_balanced_slices`' own docstring claims that with `snap=True` no
    interior slice (excluding the first and the last, whose bounds are not
    both a choose_cut result) falls under half its share or over twice it.
    Generates CJK text 60 to 320 characters long with a clause mark every 3
    to 30 percent of it, at every part count 2 through 6, and checks every
    interior slice against `share = ceil(len(text) / n)`.

    The lower bound compares against `share // 2`, not the exact half: the
    window `choose_cut` is handed for the cut before an interior slice, and
    the one after it, is built from that same floored `size // 2`, so an odd
    share can legitimately land a slice one character under the exact half
    without either window having done anything wrong.
    """
    rng = random.Random(20260905)
    checked = 0
    for _ in range(3000):
        length = rng.randint(60, 320)
        text = _clause_marked_text(rng, length)
        n = rng.randint(2, 6)
        share = -(-len(text) // n)  # ceil division, same as _balanced_slices' size
        if max(len(w) for w in text.split()) <= CHATBOX_LIMIT // n:
            continue  # word-packed, not character-sliced: snap never touches it
        slices = _balanced_slices(text, n, CHATBOX_LIMIT, snap=True)
        assert len(slices) == n
        assert "".join(slices) == text
        for piece in slices[1:-1]:
            checked += 1
            assert share // 2 <= len(piece) <= 2 * share, (length, n, share, piece)
    assert checked > 0  # the loop must have actually exercised the character path


# -- routing: a lone spaceless word too long for its share goes to the
# character path even when the text as a whole contains an ASCII space ------


_STRAY_SPACE_JA = (
    "\u3059\u3054\u3044! "
    "\u99c5\u306e\u8fd1\u304f\u306e\u65b0\u3057\u3044\u30ab\u30d5\u30a7\u306f"
    "\u671d\u516b\u6642\u306b\u958b\u304f\u3089\u3057\u3044\u306e\u3067\u3001"
    "\u660e\u65e5\u4e00\u7dd2\u306b\u884c\u304d\u307e\u305b\u3093\u304b"
)


def test_balanced_slices_routes_a_lone_long_word_to_the_character_path():
    """The bug report this test exists for. `normalize` never converts `!`
    or `?`, so the ASCII space the model emits after one survives, and
    `is_spaceless` on the WHOLE text sees it and returns False. A guard
    keyed on that sends this to the word packer, which takes the
    4-character first word whole and has nothing left to balance the
    35-character second word against: 4, 35, empty. Checked per word, the
    second word alone is spaceless-script and longer than its 16-character
    share (48 // 3), so the guard reroutes to the character path."""
    slices = _balanced_slices(_STRAY_SPACE_JA, 3, 48)

    assert slices == [
        _STRAY_SPACE_JA[0:14],
        _STRAY_SPACE_JA[14:28],
        _STRAY_SPACE_JA[28:40],
    ]
    assert [len(s) for s in slices] == [14, 14, 12]
    assert all(slices), f"a slice collapsed to empty: {slices!r}"
    assert "".join(slices) == _STRAY_SPACE_JA


def test_balanced_slices_routes_and_snaps_the_lone_long_word_case():
    """Same text and share as above, with `snap=True`: per-word routing and
    snapping meet on this text, so they are checked together rather than
    the routing decision in isolation. The text's only clause mark, U+3001
    at index 28, falls inside
    the second cut's window (21 to 35) and is picked over the raw
    ceil-division index 28 itself, moving the boundary to 29 so the second
    slice ends right after the comma instead of splitting before it."""
    slices = _balanced_slices(_STRAY_SPACE_JA, 3, 48, snap=True)

    assert slices == [
        _STRAY_SPACE_JA[0:14],
        _STRAY_SPACE_JA[14:29],
        _STRAY_SPACE_JA[29:40],
    ]
    assert [len(s) for s in slices] == [14, 15, 11]
    assert all(slices), f"a slice collapsed to empty: {slices!r}"
    assert "".join(slices) == _STRAY_SPACE_JA


_KOREAN_SHORT_TOKENS = (
    "\uc548\ub155\ud558\uc138\uc694 \uc624\ub298 \ub0a0\uc528\uac00 "
    "\uc88b\ub124\uc694 \uac19\uc774 \uac78\uc744\uae4c\uc694"
)


def test_balanced_slices_still_word_packs_korean():
    """Korean tokens are space separated, and hangul is not an unspaced
    script, so the per-word guard never fires for Korean however long a
    token runs: the text keeps taking the word path, unlike the Japanese
    case above where a single token clears its share."""
    slices = _balanced_slices(_KOREAN_SHORT_TOKENS, 3, CHATBOX_LIMIT)

    assert slices == [
        "\uc548\ub155\ud558\uc138\uc694 \uc624\ub298",
        "\ub0a0\uc528\uac00 \uc88b\ub124\uc694",
        "\uac19\uc774 \uac78\uc744\uae4c\uc694",
    ]
    assert " ".join(s for s in slices if s) == _KOREAN_SHORT_TOKENS


def test_balanced_slices_latin_only_text_is_byte_for_byte_unchanged():
    """Latin words never satisfy `is_spaceless`, so the per-word guard is
    always False for this text and the word path runs unchanged."""
    text = " ".join(f"word{i}" for i in range(30))

    slices = _balanced_slices(text, 3, CHATBOX_LIMIT)

    assert slices == [
        "word0 word1 word2 word3 word4 word5 word6 word7 word8 word9 word10",
        "word11 word12 word13 word14 word15 word16 word17 word18 word19 word20",
        "word21 word22 word23 word24 word25 word26 word27 word28 word29",
    ]


def test_balanced_slices_a_long_latin_word_does_not_trigger_the_guard():
    """`internationalization` is 20 characters, past the 16-character share
    (48 // 3), which is exactly the length relationship that reroutes a
    spaceless-script word above. `is_spaceless` is still False for it,
    since it holds no CJK or Thai codepoints, so the guard does not misfire
    on a merely-long Latin word."""
    text = "hello there my friend this is an internationalization test right now"

    slices = _balanced_slices(text, 3, 48)

    assert slices == [
        "hello there my friend",
        "this is an internationalization",
        "test right now",
    ]
    assert " ".join(s for s in slices if s) == text


def test_fit_message_split_pins_a_latin_only_conversation_unchanged():
    # Wiring snap through _join/_settle must not perturb a conversation that
    # never reaches the character path.
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
