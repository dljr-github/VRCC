"""Post-decode CJK punctuation normalization.

NLLB writes Japanese and Chinese with the ASCII period and comma whatever the
target script is, and m2m100 does the same for Chinese. Observed 2026-08-30 on
CPU int8 with the shipped decoding defaults, translating "I bought apples,
oranges and pears. They were cheap." from English:
`nllb-600M-int8` returned 我买了果,果和梨,它们很便宜. and
`m2m100-418M-int8` returned 我买了苹果,橙子和珍珠,它们很便宜。 Neither is
tokenizer damage: the same decode path returns 彼は「はい」と言い、その後、
彼は去った。 from m2m100 into Japanese, ideographic throughout.

`normalize` is the repair, keyed on the target's FLORES script subtag, which
the caller already holds (see `vrcc.translate.engine.TranslateEngine.translate`,
the only caller). Applying it to every family is deliberate: the defect is not
confined to one checkpoint, and the guards below leave text that already
carries the ideographic mark untouched.
"""

from __future__ import annotations

from vrcc.core.languages import Language

# Per-script mark mapping for the ASCII '.' and ',' the checkpoints emit,
# keyed on the FLORES script subtag rather than the GUI display name, the same
# accessor vrcc.stt.engine._script_seed uses. A CJK language added to the
# registry later is then covered without editing this table. Chinese splits the
# comma: the enumeration comma U+3001 belongs only between list items, while
# U+FF0C is the ordinary clause separator, so only the clause separator is safe
# to emit without knowing sentence structure. Japanese uses U+3001 for both.
# Korean (Hang) is absent on purpose: modern Korean writes ASCII marks.
_MARKS: dict[str, dict[str, str]] = {
    "Jpan": {".": "。", ",": "、"},
    "Hans": {".": "。", ",": "，"},
    "Hant": {".": "。", ",": "，"},
}

# A mark converts only when the character it attaches to falls in one of these
# ranges. Written out explicitly rather than via unicodedata.east_asian_width,
# which also matches fullwidth Latin. U+3000 IDEOGRAPHIC SPACE is left out of
# the range starting at U+3001: it is whitespace, not something a mark attaches
# to.
_CJK_RANGES: tuple[tuple[int, int], ...] = (
    (0x3001, 0x303F),    # CJK symbols and punctuation
    (0x3040, 0x309F),    # Hiragana
    (0x30A0, 0x30FF),    # Katakana
    (0x3400, 0x4DBF),    # CJK unified ideographs extension A
    (0x4E00, 0x9FFF),    # CJK unified ideographs
    (0xF900, 0xFAFF),    # CJK compatibility ideographs
    (0xFF66, 0xFF9F),    # Halfwidth katakana, dakuten and handakuten included
    (0x20000, 0x2FA1F),  # CJK extension B and later, plus compatibility supplement
)

# Closing marks carry no script of their own, so a terminator after one is
# judged against whatever the bracket or quote closes on. Both checkpoints
# quote with ASCII " and with the fullwidth and curly forms, and without this
# 他说"是的",然后离开了. keeps its ASCII comma while the same sentence without
# the quotes converts. Escaped rather than written out: a run of
# near-identical bracket glyphs is unreadable as literals.
_CLOSERS = "\"'\u2019\u201D)]}\uFF09\uFF3D\uFF5D\u300D\u300F\u3011\u3009\u300B\u3015\uFF63"

# Terminators the target script already uses. A mark next to one of these is
# left alone rather than converted, which would double the glyph: the input is
# a checkpoint that mixes conventions, so です。. is a shape to expect.
_ALREADY = "\u3002\u3001\uFF0C\uFF0E\uFF61\uFF64"


def _is_cjk(ch: str) -> bool:
    """True if ``ch`` falls in one of the ranges above."""
    cp = ord(ch)
    return any(lo <= cp <= hi for lo, hi in _CJK_RANGES)


def _anchor(text: str, i: int) -> int:
    """Index of the character the mark at ``text[i]`` attaches to, looking
    through any run of closing brackets and quotes, or -1 if there is none."""
    j = i - 1
    while j >= 0 and text[j] in _CLOSERS:
        j -= 1
    return j


def _should_convert(text: str, i: int) -> bool:
    """True if the mark at ``text[i]`` converts, judged only against the
    ORIGINAL ``text`` so the walk never reads its own output.

    A period directly followed by ASCII letters or digits is a file extension
    or a domain label, not a sentence end: without that test テスト.txt becomes
    テスト。txt. A run of the same mark is left alone, since an ASCII ellipsis
    reads better than three ideographic full stops. The neighbour test looks
    past one space, because :func:`normalize` absorbs that space: judging
    です. ... on the space alone converts, and the ideographic stop then lands
    against the ellipsis it was supposed to keep clear of. What remains is the
    test that leaves decimals, thousands separators, ``e.g.`` and URLs alone:
    the character the mark attaches to has to be CJK.
    """
    mark = text[i]
    if i + 1 < len(text):
        if mark == "." and text[i + 1].isascii() and text[i + 1].isalnum():
            return False
        k = i + 2 if text[i + 1] == " " else i + 1
        if k < len(text) and (text[k] == mark or text[k] in _ALREADY):
            return False
    j = _anchor(text, i)
    if j < 0 or text[j] in _ALREADY:
        return False
    return _is_cjk(text[j])


def normalize(text: str, target: Language) -> str:
    """Rewrite ASCII ``.`` and ``,`` into the ideographic marks a CJK target
    expects, wherever one attaches to a CJK character.

    Returns ``text`` unchanged for every script outside `_MARKS`, Korean
    included. ``!`` and ``?`` are never touched, since half-width marks are
    ordinary in casual chat and converting them reads stiffly, and neither are
    colons, semicolons, parentheses or quotes.

    A converted mark absorbs one following ASCII space: the ideographic glyph
    carries its own advance width, so the space is redundant.
    """
    marks = _MARKS.get(target.nllb.rpartition("_")[2])
    if marks is None:
        return text

    out: list[str] = []
    n = len(text)
    i = 0
    while i < n:
        ch = text[i]
        if ch in marks and _should_convert(text, i):
            out.append(marks[ch])
            if i + 1 < n and text[i + 1] == " ":
                i += 2
                continue
        else:
            out.append(ch)
        i += 1
    return "".join(out)
