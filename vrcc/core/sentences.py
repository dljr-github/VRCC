"""Detect a completed sentence in a (partial) transcription. Qt-free.

Used to inject a finished sentence into the chatbox at the speculative pass,
before the speaker fully stops. A minimum word count guards against committing
on an abbreviation ("Mr.") or a bare number ("3.14").
"""

from __future__ import annotations

# Sentence-final marks: ASCII, ellipsis, and CJK fullwidth forms.
_TERMINALS = ".!?…。！？"
# Closing marks allowed to trail the terminal (quotes, brackets).
_TRAILING = '"' + "'" + ')' + '"' + '’' + '」' + '』' + ']' + '}' + '】'


def ends_sentence(text: str, min_words: int) -> bool:
    """Whether ``text`` ends a sentence: at least ``min_words`` words and a
    terminal punctuation mark (optionally followed by a closing quote/bracket)."""
    stripped = text.rstrip()
    if not stripped:
        return False
    if len(stripped.split()) < min_words:
        return False
    last = stripped.rstrip(_TRAILING)
    if not last:
        return False
    return last[-1] in _TERMINALS
