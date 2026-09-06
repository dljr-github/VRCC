"""Post-decode CJK punctuation normalization.

A decode can carry the ASCII period and comma into CJK output instead of
the ideographic marks the script expects (see the "observed" cases in
`tests/test_translate_punctuation.py`, recorded from real checkpoint
output). `normalize` repairs that after the fact rather than depending on
either MT family to emit the right glyph on its own. It keys on the
target's FLORES script subtag, not on which model produced the text: a
checkpoint that already writes the ideographic mark has nothing left to
convert, so running the same repair regardless of family costs nothing
where the defect is absent and fixes it where it is not. That subtag is
what its only caller (`vrcc.translate.engine.TranslateEngine.translate`)
already holds.
"""

from __future__ import annotations

from vrcc.core.charclass import anchor, is_cjk, is_opener
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

# Terminators the target script already uses. A mark beside one is left alone
# rather than converted, which would double the glyph: a checkpoint mixing
# conventions puts an ASCII stop straight after an ideographic one.
_ALREADY = "\u3002\u3001\uFF0C\uFF0E\uFF61\uFF64"


def _should_convert(text: str, i: int) -> bool:
    """True if the mark at ``text[i]`` converts, judged only against the
    ORIGINAL ``text`` so the walk never reads its own output.

    A period followed by an ASCII letter or digit is a file extension or a
    domain label, not a sentence end. A mark standing next to another one is
    left alone: an ASCII ellipsis reads better than three ideographic stops,
    and a converted glyph beside an unconverted neighbour reads as a mix of
    conventions. That test looks past one space, since :func:`normalize`
    absorbs the space and the converted mark would then land against its
    neighbour. An opening bracket anchors nothing: it has opened, not closed.
    The last test, the anchor being CJK, is what leaves decimals, ``e.g.``
    and URLs alone."""
    mark = text[i]
    if i + 1 < len(text):
        if mark == "." and text[i + 1].isascii() and text[i + 1].isalnum():
            return False
        k = i + 2 if text[i + 1] == " " else i + 1
        if k < len(text) and (text[k] in ".," or text[k] in _ALREADY):
            return False
    j = anchor(text, i)
    if j < 0 or text[j] in _ALREADY or is_opener(text[j]):
        return False
    return is_cjk(text[j])


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
    # Two C-level scans against a per-character Python loop over the whole
    # hypothesis. One family writes Japanese with the ideographic marks
    # already, so this is the common case there, not a rare one.
    if not any(mark in text for mark in marks):
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
