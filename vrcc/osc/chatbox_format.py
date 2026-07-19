"""Fit captions/translations to VRChat's 144-char chatbox display limit.

Split out of :mod:`vrcc.osc.chatbox` (which re-exports these names) so that
module stays under the line cap: this half is pure text shaping, no OSC, no
threads.
"""

from __future__ import annotations

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
    """
    if not translations:
        return original.strip()

    texts = [text for _, text in translations]
    parts = [original, *texts] if cfg.include_original else texts
    return cfg.translation_separator.join(parts).strip()


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


def _balanced_slices(text: str, n: int, limit: int) -> list[str]:
    """Split `text` into exactly `n` ordered slices of near-equal length.

    Word-based whenever the text splits into words that each fit `limit`
    (fewer words than slices just leaves trailing slices empty): each slice
    takes whole words greedily toward a running remaining-length/remaining-
    slices target (last slice takes the rest), so joining the slices with
    spaces preserves every word in order. Character-based ceil-division runs
    only for spaceless scripts or a pathological over-long word, where the
    concatenation reproduces `text` exactly. Callers drop empty slices.
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
        return slices
    size = -(-len(text) // n)  # ceil division
    return [text[i * size : (i + 1) * size] for i in range(n)]


def fit_message(
    original: str, translations: list[tuple[str, str]], cfg: OscConfig
) -> list[str]:
    """Fit a caption and its translations into send-ready chatbox parts.

    Non-"split" modes defer to `fit_chatbox` on the `format_message` result.
    In "split" mode an over-limit message is NOT greedy-packed as one joined
    string (that carves each language arbitrarily across part boundaries):
    instead every text is cut into the same number of balanced slices via
    `_balanced_slices` and part i joins slice i of each text with
    ``cfg.translation_separator``, so all languages advance together. Empty
    slices are omitted; a message that already fits comes back as one part.
    """
    joined = format_message(original, translations, cfg)
    if cfg.overflow != "split":
        return fit_chatbox(joined, cfg.overflow)
    if not joined:
        return []
    if len(joined) <= CHATBOX_LIMIT:
        return [joined]

    # The same texts format_message joins, in the same order.
    if translations:
        texts = [text for _, text in translations]
        if cfg.include_original:
            texts.insert(0, original)
    else:
        texts = [original]
    texts = [text.strip() for text in texts]
    texts = [text for text in texts if text]

    start = max(2, -(-len(joined) // CHATBOX_LIMIT))
    for n in range(start, _MAX_MESSAGE_SLICES + 1):
        sliced = [_balanced_slices(text, n, CHATBOX_LIMIT) for text in texts]
        parts = []
        for i in range(n):
            part = cfg.translation_separator.join(
                s[i] for s in sliced if s[i]
            ).strip()
            if part:
                parts.append(part)
        if parts and all(len(part) <= CHATBOX_LIMIT for part in parts):
            return parts
    # Degenerate input that no slice count could balance: greedy word packing
    # still delivers everything, just without per-language alignment.
    return _split_words(joined, CHATBOX_LIMIT)


def _split_words(text: str, limit: int) -> list[str]:
    chunks: list[str] = []
    current = ""
    for word in text.split():
        while len(word) > limit:
            if current:
                chunks.append(current)
                current = ""
            chunks.append(word[:limit])
            word = word[limit:]

        candidate = f"{current} {word}" if current else word
        if len(candidate) <= limit:
            current = candidate
        else:
            chunks.append(current)
            current = word
    if current:
        chunks.append(current)
    return chunks
