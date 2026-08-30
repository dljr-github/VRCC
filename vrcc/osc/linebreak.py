"""Script-level line-breaking knowledge for chatbox text.

Answers "what does this script do at a cut": which characters a slice
boundary may land next to, and where a cut reads best. Knows nothing about
the 144-character chatbox limit or how many parts a message needs; that
policy stays in :mod:`vrcc.osc.chatbox_format`, which calls in here.

Every non-ASCII codepoint below is written as a \\uXXXX escape rather than a
raw glyph, so a normalizing editor or write path cannot silently substitute a
canonically-equivalent codepoint (as happened once to a raw glyph here): an
escape reads back as the exact codepoint it names, and a reviewer can verify
it by eye against a codepoint table without relying on font rendering.
"""

from __future__ import annotations

import unicodedata

# Terminators, their halfwidth forms, and the ellipsis. A cut is welcome
# right after one of these, so they also can never lead the line they would
# otherwise end (folded into `_NO_START` below).
_BREAK_AFTER = "\u3002\u3001\uFF0C\uFF0E\uFF01\uFF1F\uFF1A\uFF1B\uFF61\uFF64\u2026"

# ASCII "!" and "?" stay out, though the checkpoints emit them inside
# Japanese and punctuation.normalize leaves them alone. Measured over 2500
# generated Japanese messages (150-400 chars, ideographic marks alongside
# ASCII terminators, n = 2, 3, 4, 15,000 cuts): adding them takes cuts
# landing after a sentence end from 98.1% to 100%, and costs slices opening
# on a space 7 -> 3254 (a snap after "! " lands on the space) and slices cut
# through a URL 57 -> 1295 (a query string's own "?").

# Closing brackets and quotes. A closer at the start of a line strands the
# reader looking for what it closed, so `_NO_START` includes this set. The
# same set is also the run `_ends_clause` walks back through: a closer
# carries no script of its own, so the mark it closes over decides, which
# is the same reasoning `vrcc.translate.punctuation._anchor` uses for the
# same walk. Kept identical to `punctuation._CLOSERS`, character for
# character, by a test that checks the two against each other.
_CLOSING = "\"'\u2019\u201D)]}\uFF09\uFF3D\uFF5D\u300D\u300F\u3011\u3009\u300B\u3015\uFF63"

# Opening brackets and quotes: the mirror set. An opener at the end of a
# line strands the reader before its content, so it belongs in `_NO_END`.
_OPENING = "\"'\u2018\u201C([{\uFF08\uFF3B\uFF5B\u300C\u300E\u3010\u3008\u300A\u3014\uFF62"

# Kinsoku line-start prohibition: a line may never begin with one of these,
# since each reads as attached to what came before it rather than as the
# start of something new.
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

# Kinsoku line-end prohibition: a line may never end on one of these, since
# each is written leading into what follows it.
_NO_END = (
    _OPENING
    + "\u0E40\u0E41\u0E42\u0E43\u0E44"  # Thai preposed vowels, written before their consonant
)


def safe_cut(text: str, index: int) -> int:
    """Back a prospective slice boundary at `index` up past any combining
    mark (`unicodedata.category(ch) in ("Mn", "Mc")`, e.g. Thai vowel/tone
    marks) so a cut never separates one from its base character.
    `unicodedata.combining(ch) != 0` is not enough: it returns 0 for some
    marks, such as Thai MAI HAN-AKAT and SARA II, that still need their base.
    Falls back to the original `index` if nudging would collapse to 0 (an
    adversarial run of nothing but combining marks), so callers always make
    forward progress.
    """
    cut = index
    while 0 < cut < len(text) and unicodedata.category(text[cut]) in ("Mn", "Mc"):
        cut -= 1
    return cut if cut > 0 else index


def is_spaceless(text: str) -> bool:
    """Whether `text` is written in a script that does not separate words.

    `text.split()` returns ONE token for a Japanese or Chinese sentence, and
    that token is usually under the 144-char limit, so the word packer accepts
    it and drops the whole translation into ONE slice while leaving the other
    parts blank of it. A translation short enough to travel whole is repeated
    by :func:`_assemble` and never arrives here; one too long has to be cut,
    and cutting it by character is the only way every part carries a share.

    Whitespace anywhere means the script separates words (Korean does, and packs
    correctly), so only a run of unseparated text reaches the character branch
    the docstring above already promises it.

    Thai is here for the same reason as CJK: written without spaces between
    words, so `.split()` returns one token. Those are the only scripts among
    the languages vrcc.core.languages offers that do this, so the ranges stop
    there rather than guessing at ones nothing can produce.
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
    """Whether `ch` is an ASCII letter or digit, the two characters a cut
    between them would sever a Latin word rather than land on a script or
    kinsoku boundary. ASCII-only: a Latin loanword digraph is the concern,
    not `str.isalnum()`'s broader notion that also accepts, say, Devanagari
    or CJK digits, which is not this rule's job."""
    return ch.isascii() and ch.isalnum()


def _legal(text: str, i: int) -> bool:
    """Whether a slice may start at `text[i]`: not a combining mark, not
    forbidden to lead a line, not preceded by something forbidden to end
    one, and not splitting an ASCII word in half.

    A single test rather than a chain, folding the combining-mark rule and
    the ASCII-word rule into the kinsoku rule. Each condition here reads
    only `text[i]` and `text[i - 1]`, the same two positions the kinsoku
    checks already read, so this is still one pointwise test with no memory
    of where a caller's backward walk has already been: nothing here can
    make that walk oscillate.
    """
    if unicodedata.category(text[i]) in ("Mn", "Mc"):
        return False
    if text[i] in _NO_START:
        return False
    if i > 0 and _is_ascii_alnum(text[i - 1]) and _is_ascii_alnum(text[i]):
        return False
    return i == 0 or text[i - 1] not in _NO_END


def _ends_clause(text: str, i: int) -> bool:
    """Whether cutting before `i` lands right after a clause mark, looking
    back through any run of closing brackets and quotes the mark might sit
    behind."""
    j = i - 1
    while j >= 0 and text[j] in _CLOSING:
        j -= 1
    return j >= 0 and text[j] in _BREAK_AFTER


def choose_cut(text: str, index: int, floor: int, lo: int, hi: int) -> int:
    """Pick a slice boundary near `index`.

    Scans `[lo, hi]`, clamped at or above `floor` and at or below
    `len(text) - 2`, for the clause boundary nearest `index`: a cut there
    reads as a sentence break rather than a severed word or mark. Every
    normalized CJK translation ends in a terminator, so the window excludes
    `len(text) - 1`; leaving it a candidate lets one interior cut swallow a
    whole slice and leave the true last slice empty.

    Falls back to walking backward from `index` (also clamped to
    `len(text) - 2`) to the nearest `_legal` position, never below `floor`.
    If nothing in that range is legal either, returns the clamped `index`
    unchanged, which is what the caller would have used anyway: not a new
    failure mode, just today's.

    Never returns below `floor`. Also never at or past `len(text) - 1`,
    unless `floor` itself is already at or past that point, in which case
    `floor` wins: the monotonic-bounds guarantee the caller depends on takes
    priority over the terminal clamp.
    """
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
