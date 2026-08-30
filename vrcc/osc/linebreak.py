"""Script-level line-breaking knowledge for chatbox text.

Answers "what does this script do at a cut": which characters a slice
boundary may land next to, and where a cut reads best. Knows nothing about
the 144-character chatbox limit or how many parts a message needs; that
policy stays in :mod:`vrcc.osc.chatbox_format`, which calls in here.
"""

from __future__ import annotations

import unicodedata

# Terminators, their halfwidth forms, and the ellipsis. A cut is welcome
# right after one of these, so they also can never lead the line they would
# otherwise end (folded into `_NO_START` below).
_BREAK_AFTER = "。、，．！？：；｡､…"

# Closing brackets and quotes. A closer at the start of a line strands the
# reader looking for what it closed, so `_NO_START` includes this set. The
# same set is also the run `_ends_clause` walks back through: a closer
# carries no script of its own, so the mark it closes over decides, which
# is the same reasoning `vrcc.translate.punctuation._anchor` uses for the
# same walk.
_CLOSING = "\"'’”)]}）］｝」』】〉》〕｣"

# Opening brackets and quotes: the mirror set. An opener at the end of a
# line strands the reader before its content, so it belongs in `_NO_END`.
_OPENING = "\"'‘“([{（［｛「『【〈《〔｢"

# Kinsoku line-start prohibition: a line may never begin with one of these,
# since each reads as attached to what came before it rather than as the
# start of something new.
_NO_START = (
    _BREAK_AFTER
    + "・"  # middle dot: separates the words on either side of it
    + "々ゝゞヽヾ"  # iteration marks: repeat the character before them
    + "ー"  # prolonged sound mark: extends the vowel before it
    + "ぁぃぅぇぉっゃゅょゎ"  # hiragana small kana: never stand alone
    + "ァィゥェォッャュョヮ"  # katakana small kana: never stand alone
    + "゛゜"  # voicing marks: modify the kana before them
    + _CLOSING
    + "ｧｨｩｪｫｬｭｮｯｰﾞﾟ"  # halfwidth small kana, prolonged mark, voicing marks
    + "ำะๆฯ"  # Thai SARA AM, SARA A, MAIYAMOK, PAIYANNOI: attach backward
)

# Kinsoku line-end prohibition: a line may never end on one of these, since
# each is written leading into what follows it.
_NO_END = (
    _OPENING
    + "เแโใไ"  # Thai preposed vowels, written before their consonant
)


def safe_cut(text: str, index: int) -> int:
    """Back a prospective slice boundary at `index` up past any combining
    mark (`unicodedata.category(ch) in ("Mn", "Mc")`, e.g. Thai vowel/tone
    marks) so a cut never separates one from its base character.
    `unicodedata.combining(ch) != 0` is not enough: it returns 0 for some
    marks, such as Thai SARA AM and SARA II, that still need their base.
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
        "　" <= ch <= "鿿"      # CJK punctuation, kana, unified ideographs
        or "가" <= ch <= "힯"   # hangul syllables
        or "豈" <= ch <= "﫿"   # CJK compatibility ideographs
        or "･" <= ch <= "ﾟ"   # halfwidth katakana
        or "ก" <= ch <= "๛"      # Thai
        for ch in text
    )


def _legal(text: str, i: int) -> bool:
    """Whether a slice may start at `text[i]`: not a combining mark, not
    forbidden to lead a line, and not preceded by something forbidden to
    end one.

    A single test rather than a chain, folding the combining-mark rule into
    the kinsoku rule. That leaves one monotonic backward walk for a caller
    searching for a legal position, with no ordering question between the
    two rules and nothing that can oscillate.
    """
    if unicodedata.category(text[i]) in ("Mn", "Mc"):
        return False
    if text[i] in _NO_START:
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

    Never returns below `floor` and never at or past `len(text) - 1`.
    """
    ceiling = len(text) - 2
    lo = max(lo, floor)
    hi = min(hi, ceiling)
    if lo <= hi:
        clause = [i for i in range(lo, hi + 1) if _ends_clause(text, i)]
        if clause:
            return min(clause, key=lambda i: abs(i - index))
    start = min(index, ceiling)
    for i in range(start, floor - 1, -1):
        if _legal(text, i):
            return i
    return max(start, floor)
