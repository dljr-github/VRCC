"""Pinyin reading lines for Chinese translations (a study aid). Zero Qt.

:func:`annotate` appends a tone-marked pinyin line under the text of every
Chinese entry in a ``[(display_name, text), ...]`` translation list, so the
reading reaches the chatbox and the caption log through the existing
formatting paths unchanged. ``pypinyin`` is a required dependency, imported
lazily so a broken install degrades to unannotated text (logged once) instead
of taking down the translation pipeline.
"""

from __future__ import annotations

import logging
import re

from vrcc.core import languages

logger = logging.getLogger("vrcc.translate")

_missing_logged = False

# CJK unified ideographs (+ ext A and compatibility block): without at least
# one, a "reading" would just duplicate the text, so such entries are skipped.
_HAN = re.compile(r"[㐀-䶿一-鿿豈-﫿]")


# pypinyin emits punctuation as standalone tokens; joining with spaces then
# leaves "hao , tian" -- reattach closers/openers to their neighbor.
_SPACE_BEFORE = re.compile(r" (?=[、。，！？；：,.!?;:%\)）】」』…])")
_SPACE_AFTER = re.compile(r"(?<=[\(（【「『]) ")


def _is_chinese(display_name: str) -> bool:
    lang = languages.LANGUAGES.get(display_name)
    return lang is not None and lang.whisper == "zh"


def _reading(text: str) -> str | None:
    """Tone-marked pinyin for `text` ("nǐ hǎo"), or None without pypinyin.
    Non-hanzi runs (latin, punctuation) pass through as their own tokens."""
    global _missing_logged
    try:
        from pypinyin import pinyin
    except ImportError:
        if not _missing_logged:
            _missing_logged = True
            logger.warning(
                "pypinyin is not installed; Chinese translations stay unannotated"
            )
        return None
    reading = " ".join(token[0] for token in pinyin(text))
    return _SPACE_AFTER.sub("", _SPACE_BEFORE.sub("", reading))


def annotate(
    translations: list[tuple[str, str]]
) -> list[tuple[str, str]]:
    """Return `translations` with a pinyin line appended to Chinese entries.

    Never raises: a pypinyin failure keeps the original text (captions must
    not drop over a study annotation).
    """
    out: list[tuple[str, str]] = []
    for name, text in translations:
        if _is_chinese(name) and _HAN.search(text):
            try:
                reading = _reading(text)
            except Exception:  # noqa: BLE001 -- annotation is best-effort
                logger.exception("pinyin annotation failed; keeping plain text")
                reading = None
            if reading:
                text = f"{text}\n{reading}"
        out.append((name, text))
    return out
