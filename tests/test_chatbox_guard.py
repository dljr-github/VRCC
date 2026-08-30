"""What `vrcc.osc.chatbox_slice` ships at the real CHATBOX_LIMIT of 144.

Split out of tests/test_chatbox_split.py for the file-length cap: that
file pins the arithmetic (limit=48, the toy share used to demonstrate
the stray-space routing guard in isolation); this one pins production
behavior, plus the character path's separate obligation not to sever a
Latin word once mixed Latin/CJK text is routed onto it.
"""

from __future__ import annotations

from tests.test_chatbox_overflow import make_cfg
from vrcc.osc.chatbox import fit_message
from vrcc.osc.chatbox_slice import (
    CHATBOX_LIMIT,
    _balanced_slices,
    _join,
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
    per-slice share the search tries here. Measured by loading the pre-fix
    `chatbox_slice.py` under its own module name and calling `fit_message`
    unchanged: the word packer cannot split an 81-character indivisible
    token at all, so `_assemble` can only repeat it whole once the
    original's own share has shrunk enough to leave room, which does not
    happen until n=4 -- 4 parts of [141, 138, 138, 135], the same
    86-character translation duplicated in full, unchanged, in every one
    of them. With the fix, the per-word guard reroutes that body to the
    character path at n=3 already, where it snaps onto its own clause
    marks: 3 parts of [97, 90, 107], each carrying a DIFFERENT slice of
    the translation, cut right after each of its three \u3001 marks.
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
    text routes to the character path. Measured against the pre-fix
    `linebreak.py` (loaded under its own module name, unchanged otherwise):
    the raw ceil-division boundary for slice 3 lands inside "without",
    severing it into "wit" (ending slice 2) and "hout" (starting slice 3).
    The fix is in `linebreak._legal`, not here: this test exercises it
    through the character path `_balanced_slices` actually takes, the
    only path where the severing could happen.
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

# -- a caption with no translation: the character path still snaps ----------


_CAPTION_ONLY_JA = (
    "今日は仕事のあとに街を歩いていたら、"
    "見たことのない小さな店を見つけました。"
    "駅の近くに新しいカフェができたらしくて、"
    "今度の週末に一緒に行ってみたいと思っています。"
    "ケーキもおいしいらしいと聴いたことがあるし、"
    "もし良かったら一緒に行きませんか。"
    "昨日は仕事が遅くまでかかってしまいました。"
    "この辺りは夜になるととても静かになります。"
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
    ceil-division boundary at 81 lands between \u30B1 and its prolonged sound
    mark \u30FC, which may never begin a line. Snapping moves it one back,
    onto the \u3002 at index 80.
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


def test_settle_snaps_when_nothing_in_the_message_is_translated():
    """The skip is keyed on every translation already being repeated whole,
    which is vacuously satisfied when there are no translations at all.
    Called at the shape `fit_message` builds for a caption-only message,
    `_settle` must return a snapped arrangement rather than its argument.
    """
    cfg = make_cfg()
    texts = [_CAPTION_ONLY_JA]
    translated = [False]

    plain = _join(texts, translated, set(), 2, cfg)
    settled = _settle(texts, translated, set(), 2, cfg, plain)

    assert [len(part) for part in plain] == [81, 80]
    assert [len(part) for part in settled] == [80, 81]
