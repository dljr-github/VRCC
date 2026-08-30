"""Script-level line-breaking knowledge for chatbox text.

Answers "what does this script do at a cut": which characters a slice
boundary may land next to, and where a cut reads best. Knows nothing about
the 144-character chatbox limit or how many parts a message needs; that
policy stays in :mod:`vrcc.osc.chatbox_format`, which calls in here.

Every non-ASCII codepoint below is a \\uXXXX escape, never a raw glyph: a
glyph can be silently recomposed by a write path where an escape cannot, and
an escape reads back as the exact codepoint it names.
"""

from __future__ import annotations

import unicodedata

# Terminators, their halfwidth forms, and the ellipsis. A cut is welcome
# right after one, so none may lead the line it would otherwise end.
_BREAK_AFTER = "\u3002\u3001\uFF0C\uFF0E\uFF01\uFF1F\uFF1A\uFF1B\uFF61\uFF64\u2026"

# ASCII "!" and "?" stay out. The checkpoints emit them inside Japanese and
# normalize leaves them alone, but over 2500 generated Japanese messages
# (150-400 chars, ideographic marks alongside ASCII terminators, n = 2, 3,
# 4, 15,000 cuts) adding them took cuts after a sentence end from 98.1% to
# 100% while taking slices opening on a space from 7 to 3254 and slices cut
# through a URL from 57 to 1295.

# Closing brackets and quotes: a closer at the start of a line strands the
# reader looking for what it closed, so `_NO_START` includes this set. It
# is also the run `_ends_clause` walks back through, a closer carrying no
# script of its own. A test keeps it identical to `punctuation._CLOSERS`.
_CLOSING = "\"'\u2019\u201D)]}\uFF09\uFF3D\uFF5D\u300D\u300F\u3011\u3009\u300B\u3015\uFF63"

# Opening brackets and quotes: the mirror set. An opener at the end of a
# line strands the reader before its content, so it belongs in `_NO_END`.
_OPENING = "\"'\u2018\u201C([{\uFF08\uFF3B\uFF5B\u300C\u300E\u3010\u3008\u300A\u3014\uFF62"

# Kinsoku line-start prohibition: each of these reads as attached to what
# came before it, never as the start of something new.
_NO_START = (
    _BREAK_AFTER
    + "\u30FB"  # middle dot: separates the words on either side of it
    + "\u3005\u309D\u309E\u30FD\u30FE"  # iteration marks: repeat the character before them
    + "\u30FC"  # prolonged sound mark: extends the vowel before it
    + "\u3041\u3043\u3045\u3047\u3049\u3063\u3083\u3085\u3087\u308E"  # hiragana small kana: never stand alone
    + "\u30A1\u30A3\u30A5\u30A7\u30A9\u30C3\u30E3\u30E5\u30E7\u30EE"  # katakana small kana: never stand alone
    + "\u309B\u309C"  # voicing marks: modify the kana before them
    + _CLOSING
    + "\uFF67\uFF68\uFF69\uFF6A\uFF6B\uFF6C\uFF6D\uFF6E\uFF6F\uFF70\uFF9E\uFF9F"  # halfwidth small kana, prolonged mark, voicing marks
    + "\u0E33\u0E30\u0E46\u0E2F"  # Thai SARA AM, SARA A, MAIYAMOK, PAIYANNOI: attach backward
)

# Kinsoku line-end prohibition: each of these is written leading into what
# follows it.
_NO_END = (
    _OPENING
    + "\u0E40\u0E41\u0E42\u0E43\u0E44"  # Thai preposed vowels, written before their consonant
)


def safe_cut(text: str, index: int) -> int:
    """Back a prospective slice boundary at `index` up past any combining
    mark (category "Mn" or "Mc", Thai vowel and tone marks among them) so a
    cut never separates one from its base. `unicodedata.combining(ch) != 0`
    is not enough: it returns 0 for Thai MAI HAN-AKAT and SARA II, which
    still need their base. Falls back to `index` when nudging would reach 0,
    so callers always make forward progress."""
    cut = index
    while 0 < cut < len(text) and unicodedata.category(text[cut]) in ("Mn", "Mc"):
        cut -= 1
    return cut if cut > 0 else index


def is_spaceless(text: str) -> bool:
    """Whether `text` is written in a script that does not separate words.

    `text.split()` returns ONE token for a Japanese, Chinese or Thai
    sentence, so the word packer takes it whole into one slice and leaves the
    other parts blank of it; cutting by character is the only way every part
    carries a share. Those three are the only unspaced scripts
    vrcc.core.languages offers, so the ranges stop there.
    """
    if any(ch.isspace() for ch in text):
        return False
    return any(
        "\u3000" <= ch <= "\u9FFF"  # CJK punctuation, kana, unified ideographs
        or "\uAC00" <= ch <= "\uD7AF"  # hangul syllables
        or "\uF900" <= ch <= "\uFAFF"  # CJK compatibility ideographs
        or "\uFF65" <= ch <= "\uFF9F"  # halfwidth katakana
        or "\u0E01" <= ch <= "\u0E5B"  # Thai
        for ch in text
    )


def _is_ascii_alnum(ch: str) -> bool:
    """Whether `ch` is an ASCII letter or digit. ASCII-only on purpose: a
    severed Latin word is the concern, not `str.isalnum()`'s broader notion,
    which also accepts Devanagari and CJK digits."""
    return ch.isascii() and ch.isalnum()


def _legal(text: str, i: int) -> bool:
    """Whether a slice may start at `text[i]`: not a combining mark, not
    forbidden to lead a line, not preceded by something forbidden to end one,
    and not splitting an ASCII word in half.

    Every condition reads only `text[i]` and `text[i - 1]`, so this stays one
    pointwise test with no memory of a caller's backward walk: nothing here
    can make that walk oscillate."""
    if unicodedata.category(text[i]) in ("Mn", "Mc"):
        return False
    if text[i] in _NO_START:
        return False
    if i > 0 and _is_ascii_alnum(text[i - 1]) and _is_ascii_alnum(text[i]):
        return False
    return i == 0 or text[i - 1] not in _NO_END


def _ends_clause(text: str, i: int) -> bool:
    """Whether cutting before `i` lands right after a clause mark, looking
    back through any run of closers the mark might sit behind."""
    j = i - 1
    while j >= 0 and text[j] in _CLOSING:
        j -= 1
    return j >= 0 and text[j] in _BREAK_AFTER


def choose_cut(text: str, index: int, floor: int, lo: int, hi: int) -> int:
    """Pick a slice boundary near `index`.

    Scans `[lo, hi]`, clamped to `[floor, len(text) - 2]`, for the nearest
    clause boundary a line may also start on; failing that walks backward
    from `index` to the nearest `_legal` position; failing that returns the
    clamped `index`. The window excludes `len(text) - 1` because every
    normalized CJK translation ends in a terminator, and leaving it a
    candidate lets one interior cut swallow a whole slice and leave the true
    last slice empty.

    Never below `floor`, and never at or past `len(text) - 1` unless `floor`
    is already there: the caller's monotonic bounds outrank the terminal
    clamp."""
    ceiling = len(text) - 2
    lo = max(lo, floor)
    hi = min(hi, ceiling)
    if lo <= hi:
        clause = [
            i for i in range(lo, hi + 1) if _ends_clause(text, i) and _legal(text, i)
        ]
        if clause:
            return min(clause, key=lambda i: abs(i - index))
    start = min(index, ceiling)
    for i in range(start, floor - 1, -1):
        if _legal(text, i):
            return i
    return max(start, floor)
