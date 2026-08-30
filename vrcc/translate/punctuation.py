"""Post-decode CJK punctuation normalization.

Both shipped MT families write CJK with the ASCII period and comma. Decodes
`normalize` had to repair, measured 2026-08-31 on CPU int8 at the live
settings (beam 4, repetition penalty 1.1, no_repeat_ngram_size 3), six
English sentences per target:

    nllb-600M-int8, nllb-1.3B-int8      Jpan 6/6  Hans 6/6  Hant 6/6
    m2m100-418M-int8, m2m100-1.2B-int8  Jpan 0/6  Hans 5/6  Hant 5/6

nllb-600M-int8 turned "I bought apples, oranges and pears. They were cheap."
into "\u6211\u4E70\u4E86\u679C,\u5B50\u548C\u68A8.\u5B83\u4EEC\u5F88\u4FBF\u5B9C."
The m2m100 Jpan cell rules out tokenizer damage. Applying the repair to
every family is deliberate: the table finds the defect in both. It keys on
the target's FLORES script subtag, which its only caller
(`vrcc.translate.engine.TranslateEngine.translate`) already holds.
"""

from __future__ import annotations

from vrcc.core.languages import Language

# Per-script mapping for the ASCII '.' and ',' the checkpoints emit, keyed on
# the FLORES script subtag so a CJK language added later needs no edit here.
# Chinese splits the comma: U+3001 belongs only between list items and U+FF0C
# is the ordinary clause separator, so only the separator is safe to emit
# without knowing sentence structure. Japanese uses U+3001 for both. Korean
# (Hang) is absent on purpose: modern Korean writes ASCII marks.
_MARKS: dict[str, dict[str, str]] = {
    "Jpan": {".": "\u3002", ",": "\u3001"},
    "Hans": {".": "\u3002", ",": "\uFF0C"},
    "Hant": {".": "\u3002", ",": "\uFF0C"},
}

# A mark converts only when what it attaches to falls in one of these ranges,
# written out rather than via unicodedata.east_asian_width, which also matches
# fullwidth Latin. U+3000 IDEOGRAPHIC SPACE is out: nothing attaches to it.
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
# judged on what the bracket or quote closes over: without this a quoted
# clause keeps its ASCII comma where the same clause unquoted converts. Both
# checkpoints quote with ASCII " as well as the fullwidth and curly forms.
_CLOSERS = "\"'\u2019\u201D)]}\uFF09\uFF3D\uFF5D\u300D\u300F\u3011\u3009\u300B\u3015\uFF63"

# Terminators the target script already uses. A mark beside one is left alone
# rather than converted, which would double the glyph: a checkpoint mixing
# conventions puts an ASCII stop straight after an ideographic one.
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

    A period followed by an ASCII letter or digit is a file extension or a
    domain label, not a sentence end. A repeated mark is left alone, an ASCII
    ellipsis reading better than three ideographic stops; that test looks past
    one space, since :func:`normalize` absorbs the space and the converted
    stop would then land against the ellipsis. The last test, the anchor being
    CJK, is what leaves decimals, ``e.g.`` and URLs alone."""
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
    expects, wherever one attaches to a CJK character. Returns ``text``
    unchanged for every script outside `_MARKS`, Korean included.

    ``!`` and ``?`` are never touched: halfwidth marks are ordinary in casual
    chat and converting them reads stiffly. Nor are colons, semicolons,
    parentheses or quotes. A converted mark absorbs one following ASCII space,
    the ideographic glyph carrying its own advance width."""
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
