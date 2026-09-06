"""What `vrcc.osc.chatbox_slice` ships at the real CHATBOX_LIMIT of 144.

Split out of tests/test_chatbox_split.py for the file-length cap: that
file pins the arithmetic (limit=48, the toy share used to demonstrate
the stray-space routing guard in isolation); this one pins production
behavior, plus the character path's separate obligation not to sever a
Latin word once mixed Latin/CJK text is routed onto it.
"""

from __future__ import annotations

from tests.test_chatbox_budget import JA, ORIGINAL
from tests.test_chatbox_overflow import make_cfg
from vrcc.osc.chatbox import fit_message
from vrcc.osc.chatbox_slice import (
    CHATBOX_LIMIT,
    _balanced_slices,
    _covers_translations,
    _join,
    _lone_spaceless_run_fits_a_share,
    _settle,
)


_GUARD_ORIGINAL = (
    "I was walking around downtown earlier today after work and I found this "
    "really cute little shop that I had never noticed before even though I "
    "have walked past that same street so many times already this year"
)

# Same shape as _STRAY_SPACE_JA (a stray ASCII space, then one unbroken
# spaceless-script run), sized so the guard fires at the real CHATBOX_LIMIT
# rather than the toy limit=48 above.
_GUARD_TRANSLATION = (
    "\u3059\u3054\u3044? "
    "\u99c5\u306e\u8fd1\u304f\u306b\u65b0\u3057\u3044\u30ab\u30d5\u30a7\u304c\u3067\u304d\u305f\u3089\u3057\u304f\u3066\u3001"
    "\u4eca\u5ea6\u306e\u9031\u672b\u306b\u4e00\u7dd2\u306b\u884c\u3063\u3066\u307f\u305f\u3044\u3068\u601d\u3063\u3066\u3044\u3066\u3001"
    "\u30b1\u30fc\u30ad\u3082\u304a\u3044\u3057\u3044\u3089\u3057\u3044\u3068\u8074\u3044\u305f\u3053\u3068\u304c\u3042\u308b\u3057\u3001"
    "\u3082\u3057\u826f\u304b\u3063\u305f\u3089\u4e00\u7dd2\u306b\u884c\u304d\u307e\u305b\u3093\u304b\u3002"
)


def test_fit_message_reroutes_an_ordinary_message_at_the_real_chatbox_limit():
    """tests/test_chatbox_split.py's limit=48 tests pin the arithmetic; this
    one pins what a real VRChat message looks like at the production
    CHATBOX_LIMIT of 144, the only test that would catch a regression in
    shipped behavior.

    `_GUARD_TRANSLATION` is `_STRAY_SPACE_JA`'s shape at a size where the
    guard fires at production scale: a 5-character "sugoi? " opener plus
    an 81-character clause-marked body, one word far longer than any
    per-slice share the search tries here. Under a whole-text routing guard
    the word packer cannot split an 81-character indivisible token at all,
    so `_assemble` can only repeat it whole once the original's own share
    has shrunk enough to leave room, which does not happen until n=4 -- 4
    parts of [141, 138, 138, 135], the same 86-character translation
    duplicated in full, unchanged, in every one of them. The per-word guard
    reroutes that body to the character path at n=3 already, where it
    snaps onto its own clause marks: 3 parts of [97, 90, 107], each
    carrying a DIFFERENT slice of the translation, cut right after each of
    its three \u3001 marks.
    """
    cfg = make_cfg()  # live defaults: overflow "auto", include_original True, "\n"

    parts = fit_message(_GUARD_ORIGINAL, [("JA", _GUARD_TRANSLATION)], cfg)

    assert len(parts) == 3
    assert all(len(part) <= CHATBOX_LIMIT for part in parts)
    translation_lines = [part.split("\n")[-1] for part in parts]
    assert translation_lines == [
        _GUARD_TRANSLATION[0:25],
        _GUARD_TRANSLATION[25:47],
        _GUARD_TRANSLATION[47:86],
    ]
    assert [len(t) for t in translation_lines] == [25, 22, 39]
    assert "".join(translation_lines) == _GUARD_TRANSLATION


# -- mixed Latin and CJK: the character path must not sever a Latin word ----


def test_balanced_slices_does_not_sever_a_latin_word_in_mixed_text():
    """A code-switched translation: a 35-character spaceless CJK run, the
    ASCII word "without", then a 21-character spaceless CJK run. At n=5
    (144 // 5 = 28) the 35-character run clears its share and the whole
    text routes to the character path. The raw ceil-division boundary for
    slice 3 lands inside "without", severing it into "wit" (ending slice 2)
    and "hout" (starting slice 3); `linebreak._legal` refuses that position
    on the snapped path, and this test exercises it through the character
    path `_balanced_slices` actually takes.
    """
    cjk1 = (
        "\u56db\u65b9\u5c71\u8a71\u3067\u3059\u304c"
    )
    cjk2 = (
        "\u4f55\u3068\u304b\u304b\u3093\u3068\u304b"
    )
    text = cjk1 * 5 + " without " + cjk2 * 3

    slices = _balanced_slices(text, 5, CHATBOX_LIMIT, snap=True)

    assert slices == [
        "\u56db\u65b9\u5c71\u8a71\u3067\u3059\u304c\u56db\u65b9\u5c71\u8a71\u3067\u3059",
        "\u304c\u56db\u65b9\u5c71\u8a71\u3067\u3059\u304c\u56db\u65b9\u5c71\u8a71\u3067",
        "\u3059\u304c\u56db\u65b9\u5c71\u8a71\u3067\u3059\u304c ",
        "without \u4f55\u3068\u304b\u304b\u3093\u3068\u304b\u4f55",
        "\u3068\u304b\u304b\u3093\u3068\u304b\u4f55\u3068\u304b\u304b\u3093\u3068\u304b",
    ]
    assert [len(s) for s in slices] == [13, 13, 10, 16, 13]
    assert any("without" in s for s in slices), "without" + " was severed across a boundary"
    assert "".join(slices) == text


def test_balanced_slices_plain_path_does_not_sever_a_latin_word_in_mixed_text():
    # The plain (snap=False) path is what _assemble sizes every part count
    # with, and it ships whenever _settle finds no snapped candidate that
    # fits, so it has to keep a Latin island whole on its own.
    #
    # Routing is on the longest word against CHATBOX_LIMIT // n, so the
    # 35-character CJK run only clears its share from n=5 up: below that the
    # word packer runs and keeps "without" whole for free. The reconstruction
    # is asserted against the path each n actually takes, not as a disjunction
    # that either one satisfies.
    cjk1 = "四方山話ですが"
    cjk2 = "何とかかんとか"
    text = cjk1 * 5 + " without " + cjk2 * 3
    character_path = range(5, 7)

    for n in range(2, 7):
        slices = _balanced_slices(text, n, CHATBOX_LIMIT)
        if n in character_path:
            assert "".join(slices) == text, f"n={n} lost characters: {slices!r}"
        else:
            assert " ".join(s for s in slices if s) == text, f"n={n}: {slices!r}"
        assert any("without" in s for s in slices), f"n={n} severed the word: {slices!r}"


def test_a_latin_run_wider_than_a_slice_is_severed_rather_than_emptying_one():
    """An ASCII run longer than half a slice: a model code, a handle, a URL
    slug the checkpoint copied through. Keeping it whole would walk the
    boundary back to the run's start, leaving this slice empty and the next
    one carrying the rest of the text, which no part count in `fit_message`'s
    2..16 search can then fit. Both languages have to keep advancing
    together, so the run is cut instead.
    """
    ja = "これはテスト" + "a1b2c3d4e5" * 30 + "これはテスト"
    es = (
        "Hola, esto es una prueba de traduccion bastante larga para llenar "
        "la caja de texto del chat de VRChat sin problemas."
    )

    for n in (2, 3, 4):
        slices = _balanced_slices(ja, n, CHATBOX_LIMIT)
        assert "".join(slices) == ja
        assert all(slices), f"n={n} collapsed a slice: {[len(s) for s in slices]}"

    parts = fit_message("", [("JA", ja), ("ES", es)], make_cfg())
    assert all(len(part) <= CHATBOX_LIMIT for part in parts)
    # Every part carries a line of each language, not a run of one then the
    # other: the reader of either sees something in the first part.
    assert all(len(part.split("\n")) == 2 for part in parts), parts


def test_no_part_line_opens_or_closes_on_the_stray_space():
    """The per-word routing guard sends a translation carrying one stray
    ASCII space down the character path, where nothing forbids a boundary
    on or after that space. Stripping the assembled part only reaches its
    outermost ends, so a slice opening on the space would ship as an
    indented translation line under the original.
    """
    original = " ".join(f"word{i}" for i in range(40))
    translation = "あ" * 50 + "! " + "い" * 100

    parts = fit_message(original, [("JA", translation)], make_cfg())

    assert len(parts) > 1
    for part in parts:
        for line in part.split("\n"):
            assert line == line.strip(), f"padded line {line!r} in {part!r}"


# -- a caption with no translation: the character path still snaps ----------


_CAPTION_ONLY_JA = (
    "\u4ECA\u65E5\u306F\u4ED5\u4E8B\u306E\u3042\u3068\u306B\u8857\u3092\u6B69\u3044\u3066\u3044\u305F\u3089\u3001"
    "\u898B\u305F\u3053\u3068\u306E\u306A\u3044\u5C0F\u3055\u306A\u5E97\u3092\u898B\u3064\u3051\u307E\u3057\u305F\u3002"
    "\u99C5\u306E\u8FD1\u304F\u306B\u65B0\u3057\u3044\u30AB\u30D5\u30A7\u304C\u3067\u304D\u305F\u3089\u3057\u304F\u3066\u3001"
    "\u4ECA\u5EA6\u306E\u9031\u672B\u306B\u4E00\u7DD2\u306B\u884C\u3063\u3066\u307F\u305F\u3044\u3068\u601D\u3063\u3066\u3044\u307E\u3059\u3002"
    "\u30B1\u30FC\u30AD\u3082\u304A\u3044\u3057\u3044\u3089\u3057\u3044\u3068\u8074\u3044\u305F\u3053\u3068\u304C\u3042\u308B\u3057\u3001"
    "\u3082\u3057\u826F\u304B\u3063\u305F\u3089\u4E00\u7DD2\u306B\u884C\u304D\u307E\u305B\u3093\u304B\u3002"
    "\u6628\u65E5\u306F\u4ED5\u4E8B\u304C\u9045\u304F\u307E\u3067\u304B\u304B\u3063\u3066\u3057\u307E\u3044\u307E\u3057\u305F\u3002"
    "\u3053\u306E\u8FBA\u308A\u306F\u591C\u306B\u306A\u308B\u3068\u3068\u3066\u3082\u9759\u304B\u306B\u306A\u308A\u307E\u3059\u3002"
)


def test_fit_message_snaps_a_caption_sent_without_any_translation():
    """Captioning with translation off is an ordinary VRCC mode:
    `vrcc.core.pipeline_send.safe_submit` is reached with an empty
    translation list from five places, among them translation disabled
    (`pipeline_jobs.py`), the MT engine absent or raising, no target left
    after the source match, and typed text with translation off
    (`pipeline_typed.py`). `fit_message` builds `translated = [False]` for
    all of them, and `_settle` has to snap that arrangement like any other.

    This 161-character caption divides into two 81/80 slices, and the raw
    ceil-division boundary at 81 lands between \u30B1 (index 80) and its
    prolonged sound mark \u30FC (index 81), which may never begin a line.
    Snapping moves the cut one back, to 80, which is the position right
    after the \u3002 at index 79: a clause boundary, so `choose_cut` takes it
    from the clause scan rather than from the `_legal` walk.
    """
    cfg = make_cfg()

    parts = fit_message(_CAPTION_ONLY_JA, [], cfg)

    assert len(_CAPTION_ONLY_JA) == 161
    assert len(parts) == 2
    assert [len(part) for part in parts] == [80, 81]
    assert parts[0].endswith("\u3002")
    assert parts[1].startswith("\u30B1")
    assert "".join(parts) == _CAPTION_ONLY_JA
    assert all(len(part) <= CHATBOX_LIMIT for part in parts)


def test_settle_snaps_the_original_even_when_every_translation_repeats():
    # Anchoring has nothing to move when every translation is repeated
    # whole, but the ORIGINAL is never one of the repeated texts: its own
    # cuts still want snapping in that shape.
    cfg = make_cfg()
    original = "あ" * 35 + "。" + "い" * 44
    texts = [original, "OK"]
    translated = [False, True]

    plain = _join(texts, translated, {1}, 2, cfg)
    settled = _settle(texts, translated, {1}, 2, cfg, plain)

    assert settled != plain
    assert settled[0].split("\n")[0].endswith("。")
    assert not plain[0].split("\n")[0].endswith("。")


def test_snap_windows_keep_the_slice_between_two_cuts_at_half_a_share():
    # Measured from the grid alone, the two clause marks here pull the cuts
    # to 41 and 51 and leave a 10-character middle slice. The window opens
    # no nearer than half a slice (15) past the previous cut, so the second
    # mark is out of reach and the middle slice keeps its share.
    original = "あ" * 40 + "。" + "い" * 9 + "、" + "う" * 39
    size = -(-len(original) // 3)

    slices = _balanced_slices(original, 3, CHATBOX_LIMIT, snap=True)

    assert [len(s) for s in slices] == [41, 19, 30]
    assert slices[0].endswith("。")
    assert min(len(s) for s in slices) >= size // 2
    assert "".join(slices) == original


def test_snap_windows_keep_a_translation_slice_from_collapsing_to_a_mark():
    # Two marks two characters apart, one per window: measured from the grid
    # both would be taken and the middle slice would be the two characters
    # between them, shipped as a part whose translation line reads "中。".
    cfg = make_cfg()
    original = " ".join(f"word{i}" for i in range(36))
    translation = "中" * 60 + "。" + "中" + "。" + "中" * 60

    parts = fit_message(original, [("ZH", translation)], cfg)

    lines = [part.split("\n")[-1] for part in parts]
    assert len(parts) == 3
    assert min(len(line) for line in lines) >= 20
    assert "".join(lines) == translation


def test_settle_snaps_when_nothing_in_the_message_is_translated():
    """With no translations there is nothing to anchor, and the snapped
    arrangement alone is tried. Called at the shape `fit_message` builds for
    a caption-only message, `_settle` must return that arrangement rather
    than its argument.
    """
    cfg = make_cfg()
    texts = [_CAPTION_ONLY_JA]
    translated = [False]

    plain = _join(texts, translated, set(), 2, cfg)
    settled = _settle(texts, translated, set(), 2, cfg, plain)

    assert [len(part) for part in plain] == [81, 80]
    assert [len(part) for part in settled] == [80, 81]


# -- routing: a lone spaceless translation spreads across every slice -------


def test_balanced_slices_lone_spaceless_word_spreads_across_every_slice():
    """A single spaceless-script token short enough to sit inside a part's
    share (`CHATBOX_LIMIT // n`) still has to spread across every slice. The
    word path has nothing to distribute a ONE-word list across, so it piles
    the whole run into the first slice and leaves the rest empty:
    `_balanced_slices("中" * 46, 3, 144)` as `[46, 0, 0]`. `_join` calls here
    only for a text `_assemble` did NOT repeat, so nothing else is carrying
    that text across the parts."""
    text = "中" * 46

    start = _balanced_slices(text, 3, CHATBOX_LIMIT)
    end = _balanced_slices(text, 3, CHATBOX_LIMIT, anchor="end")

    assert all(start), f"a slice collapsed to empty: {start!r}"
    assert all(end), f"a slice collapsed to empty: {end!r}"
    assert "".join(start) == text
    assert "".join(end) == text


def test_lone_spaceless_run_fits_a_share_true_only_for_a_single_token_within_it():
    assert _lone_spaceless_run_fits_a_share("中" * 46, 3)  # 46 <= 144 // 3
    assert not _lone_spaceless_run_fits_a_share("中" * 50, 3)  # 50 > 48
    assert not _lone_spaceless_run_fits_a_share("word0 word1", 3)  # not lone
    assert not _lone_spaceless_run_fits_a_share("word0", 3)  # not spaceless


# -- search: growing the part count when nothing else covers every part -----


def test_covers_translations_flags_a_translation_too_short_to_fill_every_part():
    """A translation with fewer characters than parts cannot reach every
    slot by slicing, however it is distributed: three characters over four
    parts leaves one part with none of it. Spreading cannot answer this
    shape, since there is nothing left to give the empty part."""
    texts = ["word0 word1 word2 word3", "中中中"]
    translated = [False, True]

    assert not _covers_translations(texts, translated, set(), 4)
    assert _covers_translations(texts, translated, set(), 3)


def test_fit_message_split_grows_a_part_count_for_a_translation_too_short_to_slice():
    """Defect B: a translation too short to reach every part by slicing, and
    which does not fit repeated at the part count the raw message length
    estimates. The search must not settle for that count with the
    translation silent from most of it; it grows until repeating fits,
    which happens once the original's own share per part has shrunk enough
    to leave room."""
    cfg = make_cfg(overflow="split")
    original = " ".join(f"word{i}" for i in range(153))
    translation = "中"

    parts = fit_message(original, [("ZH", translation)], cfg)

    assert len(parts) == 9
    assert all(len(part) <= CHATBOX_LIMIT for part in parts)
    assert all(translation in part for part in parts)


def test_fit_message_split_keeps_its_part_count_when_a_share_sized_run_spreads():
    """A share-sized spaceless translation that spreads across every part at
    this count needs no extra one. Repeating it whole would fit here as well,
    so nothing forces the search further, and a count that already carries a
    piece of every translation in every part is the cheaper answer. `ORIGINAL`
    and `JA` are the fixtures `tools/bench_chatbox_budget.py` sweeps; wc=61,
    cc=47 is one of the messages that tool reports covered at this count
    rather than a larger one.
    """
    cfg = make_cfg(overflow="split")
    words = ORIGINAL.split()
    original = " ".join((words * (61 // len(words) + 1))[:61])
    translation = (JA * (47 // len(JA) + 1))[:47]

    parts = fit_message(original, [("JA", translation)], cfg)

    assert len(parts) == 3
    assert all(len(part) <= CHATBOX_LIMIT for part in parts)
    translation_lines = [part.split("\n")[-1] for part in parts]
    assert all(translation_lines), f"a part shipped silent: {parts!r}"
    assert "".join(translation_lines) == translation
