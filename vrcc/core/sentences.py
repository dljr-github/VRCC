"""Detect a completed sentence in a (partial) transcription. Qt-free.

Used to inject a finished sentence into the chatbox at the speculative pass,
before the speaker fully stops. A minimum word count guards against committing
on an abbreviation ("Mr.") or a bare number ("3.14"). The word-count guard is
waived for space-less scripts (CJK, kana, Thai), where a whole sentence is one
whitespace token but is still complete.
"""

from __future__ import annotations

# Sentence-final marks: ASCII, ellipsis, CJK fullwidth, Devanagari danda,
# Arabic question mark. The Greek question mark is ASCII ';' and is handled
# separately (only when the text actually contains Greek) to avoid treating a
# Latin-script semicolon as a sentence end.
_TERMINALS = ".!?…。！？।؟"
# Closing marks allowed to trail the terminal: quotes, brackets, guillemets,
# CJK corner/angle brackets.
_TRAILING = "\"')”’」』】》《«»‹›]}"


def _is_spaceless_script(text: str) -> bool:
    """True if the text contains characters from a script that does not use
    inter-word spaces (CJK ideographs, kana, Thai), where a whole sentence is
    one whitespace token so the word-count fragment guard cannot apply."""
    for ch in text:
        o = ord(ch)
        if (
            0x4E00 <= o <= 0x9FFF  # CJK unified ideographs
            or 0x3040 <= o <= 0x30FF  # hiragana + katakana
            or 0x0E00 <= o <= 0x0E7F  # Thai
            or 0x3400 <= o <= 0x4DBF  # CJK extension A
        ):
            return True
    return False


def _has_greek(text: str) -> bool:
    return any(0x0370 <= ord(ch) <= 0x03FF for ch in text)


def ends_sentence(text: str, min_words: int) -> bool:
    """Whether ``text`` ends a sentence: a terminal punctuation mark (optionally
    followed by a closing quote/bracket) and enough content to not be a bare
    abbreviation. The content check is whitespace words for spaced scripts and
    is waived for space-less scripts (an unbroken CJK/Thai sentence is one token
    but complete)."""
    stripped = text.rstrip()
    if not stripped:
        return False
    core = stripped.rstrip(_TRAILING)
    if not core:
        return False
    last = core[-1]
    is_terminal = last in _TERMINALS or (last == ";" and _has_greek(core))
    if not is_terminal:
        return False
    body = core[:-1].strip()
    if not body:
        return False
    if _is_spaceless_script(body):
        return True
    return len(body.split()) >= min_words
