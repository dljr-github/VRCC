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

from vrcc.core.charclass import anchor, is_cjk, is_closer, is_opener

# Terminators, their halfwidth forms, and the ellipsis. A cut is welcome
# right after one, so none may lead the line it would otherwise end.
_BREAK_AFTER = "\u3002\u3001\uFF0C\uFF0E\uFF01\uFF1F\uFF1A\uFF1B\uFF61\uFF64\u2026"

# ASCII "!" and "?" stay out. The checkpoints emit them inside Japanese and
# normalize leaves them alone, usually followed by an ASCII space or sitting
# inside a URL, so welcoming a cut after them opened far more slices on that
# space or inside the URL than it closed on a sentence end.

# Thai signs written after their consonant: SARA A, MAI HAN-AKAT, SARA AA and
# SARA AM through the above and below vowels (U+0E30 to U+0E3A), then
# LAKKHANGYAO, MAIYAMOK and the tone and cancellation marks (U+0E45 to
# U+0E4E), plus PAIYANNOI. Each attaches to what precedes it, so a cut in
# front of one strands it. SARA AM also arrives in its decomposed spelling
# U+0E4D U+0E32, which NFC leaves alone (the mapping is a compatibility one),
# so the tail U+0E32 has to be refused on its own.
_THAI_AFTER = (
    "\u0E2F"
    + "".join(map(chr, range(0x0E30, 0x0E3B)))
    + "".join(map(chr, range(0x0E45, 0x0E4F)))
)

# Kinsoku line-start prohibition: each of these reads as attached to what
# came before it, never as the start of something new. Closing brackets and
# quotes belong here too and are recognised by category.
_NO_START = (
    _BREAK_AFTER
    + "\u30FB\uFF65"  # middle dot: separates the words on either side of it
    + "\u3005\u309D\u309E\u30FD\u30FE"  # iteration marks: repeat the character before them
    + "\u30FC"  # prolonged sound mark: extends the vowel before it
    + "\u3041\u3043\u3045\u3047\u3049\u3063\u3083\u3085\u3087\u308E\u3095\u3096"  # hiragana small kana: never stand alone
    + "\u30A1\u30A3\u30A5\u30A7\u30A9\u30C3\u30E3\u30E5\u30E7\u30EE\u30F5\u30F6"  # katakana small kana: never stand alone
    + "\u309B\u309C"  # voicing marks: modify the kana before them
    + "\uFF67\uFF68\uFF69\uFF6A\uFF6B\uFF6C\uFF6D\uFF6E\uFF6F\uFF70\uFF9E\uFF9F"  # halfwidth small kana, prolonged mark, voicing marks
    + _THAI_AFTER
)

# Kinsoku line-end prohibition: each of these is written leading into what
# follows it. Opening brackets and quotes belong here too, by category.
# Thai preposed vowels are written before the consonant they voice.
_NO_END = "".join(map(chr, range(0x0E40, 0x0E45)))


def _attached(ch: str) -> bool:
    """Whether `ch` renders as part of the character before it: a combining
    mark (category "Mn" or "Mc"), or a Thai sign written after its consonant
    that Unicode files as a letter. `unicodedata.combining(ch) != 0` is not
    enough for either: it returns 0 for MAI HAN-AKAT and SARA II."""
    return unicodedata.category(ch) in ("Mn", "Mc") or ch in _THAI_AFTER


def safe_cut(text: str, index: int) -> int:
    """Back a prospective cut at `index` up past anything attached to the
    character before it, so a cut never separates a mark or a Thai vowel sign
    from its base. Falls back to `index` when nudging would reach 0, so
    callers always make forward progress."""
    cut = index
    while 0 < cut < len(text) and _attached(text[cut]):
        cut -= 1
    return cut if cut > 0 else index


def slice_cut(text: str, index: int, floor: int = 0) -> int:
    """`safe_cut`, then backed up to the start of an ASCII word the cut would
    otherwise sever: the character path exists for spaceless scripts, and a
    Latin island inside one is still a word.

    The walk stops at `floor` and hands back the `safe_cut` boundary
    instead. A run of ASCII wider than the window has no boundary its caller
    can afford: walking to its start empties this slice and leaves the next
    one carrying the rest of the text, which costs the message a whole part.
    Callers always make forward progress either way."""
    cut = safe_cut(text, index)
    word_start = cut
    while (
        floor < word_start < len(text)
        and _is_ascii_alnum(text[word_start - 1])
        and _is_ascii_alnum(text[word_start])
    ):
        word_start -= 1
    return word_start if word_start > floor else cut


def is_spaceless(text: str) -> bool:
    """Whether `text` is written in a script that does not separate words.

    `text.split()` returns ONE token for a Japanese, Chinese or Thai
    sentence, so the word packer takes it whole into one slice and leaves the
    other parts blank of it; cutting by character is the only way every part
    carries a share. Those three are the only unspaced scripts
    vrcc.core.languages offers. Korean is spaced, hangul or not, so it stays
    on the word path however long a token runs. Whitespace anywhere makes
    this False whatever the script.
    """
    if any(ch.isspace() for ch in text):
        return False
    return any(is_cjk(ch) or "\u0E01" <= ch <= "\u0E5B" for ch in text)


def _is_ascii_alnum(ch: str) -> bool:
    """Whether `ch` is an ASCII letter or digit. ASCII-only on purpose: a
    severed Latin word is the concern, not `str.isalnum()`'s broader notion,
    which also accepts Devanagari and CJK digits."""
    return ch.isascii() and ch.isalnum()


def _legal(text: str, i: int) -> bool:
    """Whether a slice may start at `text[i]`: not attached to the character
    before it, not forbidden to lead a line, not preceded by something
    forbidden to end one, and not splitting an ASCII word in half.

    Every condition reads only `text[i]` and `text[i - 1]`, so this stays one
    pointwise test with no memory of a caller's backward walk: nothing here
    can make that walk oscillate."""
    ch = text[i]
    if _attached(ch) or ch in _NO_START or is_closer(ch):
        return False
    if i == 0:
        return True
    prev = text[i - 1]
    if _is_ascii_alnum(prev) and _is_ascii_alnum(ch):
        return False
    return prev not in _NO_END and not is_opener(prev)


def _ends_clause(text: str, i: int) -> bool:
    """Whether cutting before `i` lands right after a clause mark, looking
    back through any run of closers the mark might sit behind."""
    j = anchor(text, i)
    return j >= 0 and text[j] in _BREAK_AFTER


def choose_cut(text: str, index: int, floor: int, lo: int, hi: int) -> int:
    """Pick a slice boundary near `index`.

    Scans `[lo, hi]`, clamped to `[floor, len(text) - 2]`, for the nearest
    clause boundary a line may also start on; failing that walks backward
    from `index` to the nearest `_legal` position at or above `lo`, then
    forward from there to the nearest one still inside `[lo, hi]`; failing
    both returns the clamped `index`, backed off an attached character where
    the window allows. Every answer stays inside the window, so the caller's
    size bound holds whichever one it is. The window stops at
    `len(text) - 2` so the last slice always keeps two characters: on a
    normalized CJK translation the last one is a terminator, and a slice of
    nothing but a terminator is a part spent on one glyph.

    Never below `floor`, and never at or past `len(text) - 1` unless `floor`
    is already there: the caller's monotonic bounds outrank the terminal
    clamp."""
    ceiling = len(text) - 2
    lo = max(lo, floor)
    hi = min(hi, ceiling)
    if lo > hi:
        # The window closed above the terminal clamp, so no position satisfies
        # both. `floor` is the only bound left that the caller depends on, and
        # answering above it would put the boundary past the text.
        return floor
    # `_legal` first: it is pointwise, and it already refuses every closer,
    # so it short-circuits the unbounded backward walk `_ends_clause` would
    # otherwise run at every position of a long run of quotes or brackets.
    clause = [
        i for i in range(lo, hi + 1) if _legal(text, i) and _ends_clause(text, i)
    ]
    if clause:
        return min(clause, key=lambda i: abs(i - index))
    # Clamped to `hi`, not just to `ceiling`: an `index` above the window
    # would start the backward walk outside it and hand back a boundary the
    # caller's size bound never allowed for.
    start = min(index, hi)
    # Bounded by `lo`, not `floor`: a long ASCII run makes every interior
    # position illegal, and a walk that ran to `floor` would escape the
    # caller's window entirely and hand back a boundary half a text away.
    for i in range(start, lo - 1, -1):
        if _legal(text, i):
            return i
    # Forward only once backward found nothing: a slice under its share beats
    # one over it, but a Latin word filling the window below `start` is still
    # a word, and its end usually sits inside the window above. Bounded below
    # by `lo` same as the backward walk: `start` can itself sit below `lo`
    # when `index` does, and scanning from `start + 1` unguarded would answer
    # from below the caller's own window.
    for i in range(max(start + 1, lo), hi + 1):
        if _legal(text, i):
            return i
    # The one return no rule has vetted. The window bound outranks the
    # attachment rule: inside a run of nothing but marks no position satisfies
    # both, and a slice a whole window short is the worse cut.
    return max(safe_cut(text, start), lo)
