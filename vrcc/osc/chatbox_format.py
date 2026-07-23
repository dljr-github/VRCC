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


def _balanced_slices(text: str, n: int, limit: int, anchor: str = "start") -> list[str]:
    """Split `text` into exactly `n` ordered slices of near-equal length.

    Word-based whenever the text splits into words that each fit `limit`
    (fewer words than slices just leaves blank slices, positioned per
    `anchor`): each slice takes whole words greedily toward a running
    remaining-length/remaining-slices target (last slice takes the rest), so
    joining the slices with spaces preserves every word in order.
    Character-based ceil-division runs only for spaceless scripts or a
    pathological over-long word, where the concatenation reproduces `text`
    exactly (boundaries are nudged off combining marks). Callers drop empty
    slices.

    `anchor="start"` (default) leaves any unused trailing slices blank --
    right for the original, which should fade out once exhausted.
    `anchor="end"` moves that content to the LAST slices instead, blank ones
    first: a short translation that runs out early still lands in the
    final, persisting part rather than only the first one.
    """
    words = text.split()
    if words and all(len(word) <= limit for word in words):
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
        if anchor == "end":
            content = [s for s in slices if s]
            slices = [""] * (n - len(content)) + content
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

    # The same texts format_message joins, in the same order. The original
    # anchors "start" (fades out once exhausted); every translation anchors
    # "end" (a short one still lands in the final, persisting part) -- see
    # `_balanced_slices`.
    if translations:
        texts = [text for _, text in translations]
        anchors = ["end"] * len(texts)
        if cfg.include_original:
            texts.insert(0, original)
            anchors.insert(0, "start")
    else:
        texts = [original]
        anchors = ["start"]
    stripped = [(text.strip(), anchor) for text, anchor in zip(texts, anchors)]
    texts = [text for text, _ in stripped if text]
    anchors = [anchor for text, anchor in stripped if text]

    start = max(2, -(-len(joined) // CHATBOX_LIMIT))
    for n in range(start, _MAX_MESSAGE_SLICES + 1):
        sliced = [
            _balanced_slices(text, n, CHATBOX_LIMIT, anchor=anchor)
            for text, anchor in zip(texts, anchors)
        ]
        parts = []
        for i in range(n):
            part = cfg.translation_separator.join(
                s[i] for s in sliced if s[i]
            ).strip()
            if part:
                parts.append(part)
        if parts and all(len(part) <= CHATBOX_LIMIT for part in parts):
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
