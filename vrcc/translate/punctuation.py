"""Post-decode CJK punctuation normalization.

`MtTokenizer.decode` was measured byte-identical to
`tokenizers.Tokenizer.decode(ids, skip_special_tokens=True)`, and the decoding
parameters were cleared by ablation -- neither is the source of the defect.
`nllb-1.3B-int8` writes Japanese and Chinese with ASCII period and comma
regardless of target script; `m2m100-418M` on the same decode path does not,
which is what points at the checkpoint rather than VRCC. `normalize` is the
repair: a post-decode pass keyed on the target language that the caller
already has in hand (see `vrcc.translate.engine.TranslateEngine.translate`,
the only caller).
"""

from __future__ import annotations

from vrcc.core.languages import Language

# Per-language mark mapping for the ASCII '.' and ',' NLLB emits regardless of
# script. Chinese splits the comma: the enumeration comma U+3001 belongs only
# between list items, while U+FF0C is the ordinary clause separator, so only
# the clause separator is safe to emit without knowing sentence structure.
# Japanese uses U+3001 for both, so one entry covers it. Korean is excluded
# entirely: modern Korean writes ASCII period and comma.
_MARKS: dict[str, dict[str, str]] = {
    "Japanese": {".": "。", ",": "、"},
    "Chinese Simplified": {".": "。", ",": "，"},
    "Chinese Traditional": {".": "。", ",": "，"},
}

# A preceding character must fall in one of these ranges for a mark to
# convert. Written out explicitly rather than via unicodedata.east_asian_width,
# which also matches fullwidth Latin. U+3000 IDEOGRAPHIC SPACE is left out of
# the range starting at U+3001: it is whitespace, not something a mark
# attaches to.
_CJK_RANGES: tuple[tuple[int, int], ...] = (
    (0x3001, 0x303F),    # CJK symbols and punctuation: 」 、 。 included
    (0x3040, 0x309F),    # Hiragana
    (0x30A0, 0x30FF),    # Katakana, including the prolonged sound mark and ・
    (0x3400, 0x4DBF),    # CJK unified ideographs extension A
    (0x4E00, 0x9FFF),    # CJK unified ideographs
    (0xF900, 0xFAFF),    # CJK compatibility ideographs
    (0xFF66, 0xFF9D),    # Halfwidth katakana
    (0x20000, 0x2FA1F),  # CJK extension B and later, plus compatibility supplement
)


def _is_cjk(ch: str) -> bool:
    """True if ``ch`` falls in one of the ranges above."""
    cp = ord(ch)
    return any(lo <= cp <= hi for lo, hi in _CJK_RANGES)


def _should_convert(text: str, i: int) -> bool:
    """True if the mark at ``text[i]`` converts, judged only against the
    ORIGINAL ``text`` so the walk never cascades off its own output.

    Index 0 has no preceding character, so it never converts. A run of the
    same mark on either side is left alone: without that check
    ``です...`` would convert left to right, since the second period sees a
    freshly minted ``。`` before it, and land on ``です。。。`` -- an ASCII
    ellipsis reads better than three ideographic full stops. What remains is
    the single test that leaves decimals, thousands separators, ``e.g.`` and
    URLs alone: the preceding character has to be CJK.
    """
    if i == 0:
        return False
    mark = text[i]
    if text[i - 1] == mark:
        return False
    if i + 1 < len(text) and text[i + 1] == mark:
        return False
    return _is_cjk(text[i - 1])


def normalize(text: str, target: Language) -> str:
    """Rewrite ASCII ``.`` and ``,`` into the ideographic marks a CJK target
    expects, wherever one follows a CJK character.

    Returns ``text`` unchanged for any ``target`` outside `_MARKS`: every
    script but Japanese and the two Chinese scripts, Korean included, since
    modern Korean keeps ASCII marks. ``!`` and ``?`` are never touched --
    half-width marks are ordinary in casual chat and converting them reads
    stiffly -- and neither are colons, semicolons, parentheses or quotes.

    A converted mark absorbs exactly one following ASCII space, never a run:
    the ideographic glyph carries its own advance width, so one following
    space is always redundant.
    """
    marks = _MARKS.get(target.display)
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
