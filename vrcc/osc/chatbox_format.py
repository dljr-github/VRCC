"""Fit captions/translations to VRChat's 144-char chatbox display limit.

Split out of :mod:`vrcc.osc.chatbox` (which re-exports these names) so that
module stays under the line cap: this half is pure text shaping, no OSC, no
threads.
"""

from __future__ import annotations

import unicodedata

from vrcc.core.config import OscConfig

CHATBOX_LIMIT = 144

# Cap on how many parallel slices fit_message tries before falling back to
# greedy word packing: past this a message is degenerate (separator overhead
# dominates) and unreadable at 2 s a part anyway.
_MAX_MESSAGE_SLICES = 16

# Parts drain one every max(split_delay_s, min_interval_s) and a new utterance
# clears the queue, so measured over a 12-turn conversation nothing past the
# third part is seen at conversational pace. fit_message will spend one extra
# part to carry a translation whole, but never past this.
_MAX_REPEAT_PARTS = 3


def format_message(
    original: str, translations: list[tuple[str, str]], cfg: OscConfig
) -> str:
    """Build the chatbox text from a phrase and its ``[(name, text), ...]``
    translations (name unused yet). No translations -> just ``original``; else
    ``cfg.include_original`` decides whether ``original`` is prepended and the
    pieces are joined by ``cfg.translation_separator``. Overall-stripped.

    Always the RAW join, regardless of length: callers that need the text
    fit to `CHATBOX_LIMIT` go through `fit_message`, which budgets the
    original out of the way of the translation when this overflows in
    "truncate"/"send" mode (see `_budget_original`). Other callers (the
    caption log's overflow badge, `scripts/smoke_e2e.py`) rely on this
    staying the untouched length to detect that a message was too long.
    """
    if not translations:
        return original.strip()

    texts = [text for _, text in translations]
    parts = [original, *texts] if cfg.include_original else texts
    return cfg.translation_separator.join(parts).strip()


def _budget_original(original: str, texts: list[str], separator: str) -> str:
    """Shorten `original` to fit alongside `texts` (the translations,
    already-joined order) within `CHATBOX_LIMIT`, reserving the full
    translation text first. If the translations alone already fill or
    exceed the budget, `original` is dropped entirely rather than the
    translation losing any of its share.
    """
    translations_joined = separator.join(texts)
    budget = CHATBOX_LIMIT - len(separator) - len(translations_joined)
    if budget <= 0:
        return translations_joined.strip()
    if len(original) <= budget:
        shortened = original
    elif budget == 1:
        shortened = "…"
    else:
        shortened = original[: budget - 1] + "…"
    return separator.join([shortened, *texts]).strip()


def fit_chatbox(text: str, mode: str) -> list[str]:
    """Fit `text` to VRChat's 144-char display limit per ``mode``: ``truncate``
    clips over-limit text to ``text[:143] + "…"``; ``split`` greedily packs
    whole words into <=144-char chunks (hard-splitting a lone over-long word);
    ``send`` passes through unchanged. Empty text -> ``[]``.
    """
    if not text:
        return []
    if mode == "send":
        return [text]
    if mode == "truncate":
        if len(text) <= CHATBOX_LIMIT:
            return [text]
        return [text[: CHATBOX_LIMIT - 1] + "…"]
    if mode == "split":
        return _split_words(text, CHATBOX_LIMIT)
    raise ValueError(f"Unknown overflow mode: {mode!r}")


def _safe_cut(text: str, index: int) -> int:
    """Back a prospective slice boundary at `index` up past any combining
    marks (`unicodedata.combining(ch) != 0`, e.g. Thai vowel/tone marks) so
    a cut never separates one from its base character. Falls back to the
    original `index` if nudging would collapse to 0 (an adversarial run of
    nothing but combining marks), so callers always make forward progress.
    """
    cut = index
    while 0 < cut < len(text) and unicodedata.combining(text[cut]) != 0:
        cut -= 1
    return cut if cut > 0 else index


def _is_spaceless(text: str) -> bool:
    """Whether `text` is written in a script that does not separate words.

    `text.split()` returns ONE token for a Japanese or Chinese sentence, and
    that token is usually under the 144-char limit, so the word packer accepts
    it and drops the whole translation into ONE slice while leaving the other
    parts blank of it. A translation short enough to travel whole is repeated
    by :func:`_assemble` and never arrives here; one too long has to be cut,
    and cutting it by character is the only way every part carries a share.

    Whitespace anywhere means the script separates words (Korean does, and packs
    correctly), so only a run of unseparated CJK reaches the character branch
    the docstring above already promises it.
    """
    if any(ch.isspace() for ch in text):
        return False
    return any(
        "　" <= ch <= "鿿"      # CJK punctuation, kana, unified ideographs
        or "가" <= ch <= "힯"   # hangul syllables
        or "豈" <= ch <= "﫿"   # CJK compatibility ideographs
        or "･" <= ch <= "ﾟ"   # halfwidth katakana
        for ch in text
    )


def _balanced_slices(text: str, n: int, limit: int) -> list[str]:
    """Split `text` into exactly `n` ordered slices of near-equal length.

    Word-based whenever the text splits into words that each fit `limit`:
    each slice takes whole words greedily toward a running
    remaining-length/remaining-slices target (last slice takes the rest), so
    joining the slices with spaces preserves every word in order.
    Character-based ceil-division runs only for spaceless scripts or a
    pathological over-long word, where the concatenation reproduces `text`
    exactly (boundaries are nudged off combining marks). Callers drop empty
    slices.

    Fewer words than slices simply leaves the trailing slices blank, so a text
    that runs out early fades rather than repeating.
    """
    words = text.split()
    # A spaceless run within its per-part share is better carried whole, which
    # _assemble already does where it fits.
    spaceless = _is_spaceless(text) and len(text) > limit // n
    if words and not spaceless and all(len(word) <= limit for word in words):
        slices: list[str] = []
        idx = 0
        for k in range(n - 1):
            target = len(" ".join(words[idx:])) / (n - k)
            piece = words[idx]
            idx += 1
            while idx < len(words):
                grown = len(piece) + 1 + len(words[idx])
                # Take the next word only while it moves the slice at least
                # as close to the target -- overshoot stays within one word.
                if abs(grown - target) > abs(len(piece) - target):
                    break
                piece = f"{piece} {words[idx]}"
                idx += 1
            slices.append(piece)
            if idx >= len(words):
                break
        slices.extend([""] * (n - 1 - len(slices)))
        slices.append(" ".join(words[idx:]))
        return slices
    size = -(-len(text) // n)  # ceil division
    bounds = [0]
    for i in range(1, n):
        cut = _safe_cut(text, min(i * size, len(text)))
        bounds.append(max(cut, bounds[-1]))
    bounds.append(len(text))
    return [text[bounds[i] : bounds[i + 1]] for i in range(n)]


def fit_message(
    original: str, translations: list[tuple[str, str]], cfg: OscConfig
) -> list[str]:
    """Fit a caption and its translations into send-ready chatbox parts.

    Non-"split" modes defer to `fit_chatbox`, but on a translation-aware
    text: if the plain `format_message` join overflows `CHATBOX_LIMIT` with
    `cfg.include_original` on, the original -- not the translation, the
    line a non-speaker actually reads -- is the one shortened to make room
    (see `_budget_original`). In "split" mode an over-limit message is NOT
    greedy-packed as one joined string (that carves each language
    arbitrarily across part boundaries): instead every text is cut into the
    same number of balanced slices via `_balanced_slices` and part i joins
    slice i of each text with ``cfg.translation_separator``, so all
    languages advance together. Empty slices are omitted; a message that
    already fits comes back as one part.
    """
    joined = format_message(original, translations, cfg)
    if cfg.overflow != "split":
        if cfg.include_original and translations and len(joined) > CHATBOX_LIMIT:
            texts = [text for _, text in translations]
            joined = _budget_original(original, texts, cfg.translation_separator)
        return fit_chatbox(joined, cfg.overflow)
    if not joined:
        return []
    if len(joined) <= CHATBOX_LIMIT:
        return [joined]

    # The same texts format_message joins, in the same order.
    if translations:
        texts = [text for _, text in translations]
        translated = [True] * len(texts)
        if cfg.include_original:
            texts.insert(0, original)
            translated.insert(0, False)
    else:
        texts = [original]
        translated = [False]
    stripped = [(text.strip(), is_tr) for text, is_tr in zip(texts, translated)]
    texts = [text for text, _ in stripped if text]
    translated = [is_tr for text, is_tr in stripped if text]

    start = max(2, -(-len(joined) // CHATBOX_LIMIT))
    for n in range(start, _MAX_MESSAGE_SLICES + 1):
        parts, repeated = _assemble(texts, translated, n, cfg)
        if not (parts and all(len(part) <= CHATBOX_LIMIT for part in parts)):
            continue
        # One more part, never past the third, when the extra room lets a
        # translation travel whole that could not at this count. Measured, that
        # trade wins at one target (delivery 75% to 100%) and is what the
        # ceiling protects: unbounded, the same growth cost three targets 75%
        # to 71.7%, because a part nobody reads is worse than a shorter one.
        if n + 1 <= _MAX_REPEAT_PARTS:
            grown, grown_repeated = _assemble(texts, translated, n + 1, cfg)
            fits = grown and all(len(part) <= CHATBOX_LIMIT for part in grown)
            if fits and len(grown_repeated) > len(repeated):
                return grown
        return parts
    # Degenerate input that no slice count could balance: split EACH
    # language's own text independently rather than the flat joined string
    # (whose ``.split()`` would treat the "\n" separator as just more
    # whitespace, losing it and interleaving languages mid-chunk). Every
    # language still delivers in full, just as its own run of parts.
    parts = []
    for text in texts:
        parts.extend(_split_words(text, CHATBOX_LIMIT))
    return parts


def _assemble(
    texts: list[str], translated: list[bool], n: int, cfg: OscConfig
) -> tuple[list[str], set[int]]:
    """Build `n` parts, repeating each translation that fits rather than
    slicing it. Returns the parts and which texts were repeated.

    Parts drain one every `max(split_delay_s, min_interval_s)`, and a new
    utterance clears the queue, so anything past roughly the third part is
    rarely seen at conversational pace. Slicing a translation therefore spread
    it across parts the reader would never receive; putting a short one in
    EVERY part means it arrives in the first and is still on screen when the
    user stops talking. Measured over a 12-turn conversation at four paces,
    that beat slicing on delivery and on what survives afterwards, in every
    combination tried.

    Repeated shortest first and only while every part still fits, so a long
    translation falls back to being sliced across the parts in order. The
    original is always sliced: it is the one text the reader can afford to lose
    the tail of, since it is not what a non-speaker is reading.
    """
    # Shortest first so a long translation cannot crowd out two short ones it
    # could not fit beside anyway. Swept 2875 realistic messages through
    # fit_message with the order reversed and the output was identical every
    # time, so this is insurance rather than a measured win: keep it because it
    # is the order that can only help, not because it is currently doing work.
    order = sorted(
        (i for i, is_tr in enumerate(translated) if is_tr), key=lambda i: len(texts[i])
    )
    repeated: set[int] = set()
    for i in order:
        candidate = repeated | {i}
        if all(
            len(part) <= CHATBOX_LIMIT
            for part in _join(texts, candidate, n, cfg)
        ):
            repeated = candidate
    return _join(texts, repeated, n, cfg), repeated


def _join(
    texts: list[str], repeated: set[int], n: int, cfg: OscConfig
) -> list[str]:
    """`n` parts, with the `repeated` texts whole in each and the rest sliced."""
    sliced = {
        i: _balanced_slices(text, n, CHATBOX_LIMIT)
        for i, text in enumerate(texts)
        if i not in repeated
    }
    parts = []
    for slot in range(n):
        pieces = [
            texts[i] if i in repeated else sliced[i][slot]
            for i in range(len(texts))
        ]
        part = cfg.translation_separator.join(p for p in pieces if p).strip()
        if part:
            parts.append(part)
    return parts


def _split_words(text: str, limit: int) -> list[str]:
    chunks: list[str] = []
    current = ""
    for word in text.split():
        while len(word) > limit:
            if current:
                chunks.append(current)
                current = ""
            cut = _safe_cut(word, limit)
            chunks.append(word[:cut])
            word = word[cut:]

        candidate = f"{current} {word}" if current else word
        if len(candidate) <= limit:
            current = candidate
        else:
            chunks.append(current)
            current = word
    if current:
        chunks.append(current)
    return chunks
