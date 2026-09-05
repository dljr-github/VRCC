"""Character classes shared by the MT punctuation normalizer and the chatbox
line breaker: which characters belong to a CJK script, and which are the
brackets and quotes that carry no script of their own.
"""

from __future__ import annotations

import unicodedata

# The straight ASCII quotes open and close alike, so both predicates admit
# them; every other bracket or quote declares its side through its category.
_ASCII_QUOTES = "\"'"


def is_cjk(ch: str) -> bool:
    """Whether ``ch`` is written in a CJK script: kana, ideographs, or the
    punctuation and symbols that travel with them. Fullwidth Latin and digits
    are not, which is why the ranges are written out rather than derived from
    ``unicodedata.east_asian_width``. Korean is not: modern Korean writes
    ASCII punctuation and is a spaced script (see
    ``vrcc.osc.linebreak.is_spaceless``), so no Hangul range belongs here,
    Hangul Compatibility Jamo included even though it sits between two
    ranges that do.

    U+3000 IDEOGRAPHIC SPACE is out: nothing attaches to it, and as
    whitespace it separates rather than joins. The blocks, in order:
      0x3001-0x312F  CJK Symbols and Punctuation (minus U+3000), Hiragana,
                      Katakana, Bopomofo
      0x3190-0x31FF  Kanbun, Bopomofo Extended, CJK Strokes, Katakana
                      Phonetic Extensions (Hangul Compatibility Jamo,
                      U+3130-U+318F, sits in the gap before this and is
                      excluded)
      0x3300-0x9FFF  CJK Compatibility, CJK Unified Ideographs Extension A,
                      Yijing Hexagram Symbols, CJK Unified Ideographs
                      (Enclosed CJK Letters and Months, U+3200-U+32FF, sits
                      in the gap before this and is excluded: it interleaves
                      parenthesized and circled Hangul, e.g. U+3200 and
                      U+3260, with circled ideographs and squared era names
                      in the same block, so keeping any of it would
                      reintroduce a Hangul range under a different name)
      0xF900-0xFAFF  CJK Compatibility Ideographs
      0xFF65-0xFF9F  Halfwidth katakana with its middle dot, dakuten and
                      handakuten
      0x20000-0x2FA1F  Extension B onward through the compatibility
                        supplement
    """
    cp = ord(ch)
    return (
        0x3001 <= cp <= 0x312F
        or 0x3190 <= cp <= 0x31FF
        or 0x3300 <= cp <= 0x9FFF
        or 0xF900 <= cp <= 0xFAFF
        or 0xFF65 <= cp <= 0xFF9F
        or 0x20000 <= cp <= 0x2FA1F
    )


def is_opener(ch: str) -> bool:
    """An opening bracket or quote, in any script."""
    return ch in _ASCII_QUOTES or unicodedata.category(ch) in ("Ps", "Pi")


def is_closer(ch: str) -> bool:
    """A closing bracket or quote, in any script."""
    return ch in _ASCII_QUOTES or unicodedata.category(ch) in ("Pe", "Pf")
