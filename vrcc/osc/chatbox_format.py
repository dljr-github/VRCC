"""Fit captions/translations to VRChat's 144-char chatbox display limit.

Split out of :mod:`vrcc.osc.chatbox` (which re-exports these names) so that
module stays under the line cap: this half is pure text shaping, no OSC, no
threads.
"""

from __future__ import annotations

from vrcc.core.config import OscConfig
from vrcc.osc.chatbox_slice import CHATBOX_LIMIT, _assemble, _settle
from vrcc.osc.linebreak import safe_cut

# CHATBOX_LIMIT lives in chatbox_slice.py (arranging n parts across
# languages); chatbox.py re-exports it from here so existing
# `from vrcc.osc.chatbox import CHATBOX_LIMIT` call sites keep working.

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
    "truncate" mode (see `_budget_original`). Other callers (the
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
        # safe_cut, not a raw index: a Thai or Devanagari boundary landing
        # between a base and its mark drops the mark and renders a different
        # syllable.
        shortened = original[: safe_cut(original, budget - 1)] + "…"
    return separator.join([shortened, *texts]).strip()


def _share_limit(texts: list[str], separator: str, limit: int) -> list[str]:
    """Shorten `texts` so their `separator` join fits `limit`, each text taking
    an equal share and the ones already inside it releasing the surplus.

    Cutting the joined string instead spends the limit front to back, so the
    last target arrives empty however short it is: order decides who is lost,
    not length.

    An equal share of the characters is not fairness, since the same sentence
    costs a different number of them in every language. What it buys is the
    largest number of targets that arrive WHOLE, which a proportional split
    gives up: on the ja/es/de trio in
    `tools/bench_chatbox_budget.py --policies`, proportional leaves all three
    at 52% where this leaves ja complete.

    Shortening stops on a word boundary (`_split_words`). Spaceless scripts
    have none to find and fall back to its character cut.
    """
    room = limit - len(separator) * (len(texts) - 1)
    if room < len(texts):
        return texts
    shares = [0] * len(texts)
    pending = list(range(len(texts)))
    while pending:
        per = room // len(pending)
        fits = [i for i in pending if len(texts[i]) <= per]
        if not fits:
            for i in pending:
                shares[i] = per
            break
        for i in fits:
            shares[i] = len(texts[i])
            room -= len(texts[i])
            pending.remove(i)
    shortened = []
    for text, share in zip(texts, shares):
        if len(text) <= share:
            shortened.append(text)
        elif share > 1:
            packed = _split_words(text, share - 1)
            shortened.append((packed[0] if packed else "") + "…")
        else:
            shortened.append("…")
    return shortened


def resolve_overflow(text: str, mode: str) -> str:
    """The mode `text` is really shaped by. The three explicit modes are
    returned untouched; "auto" reads the message and answers for it.

    Parts drain one at a time and a new utterance clears the queue, so past
    `_MAX_REPEAT_PARTS` the tail is not read at conversational pace: sending
    it costs the reader the wait and delivers a fragment anyway. Up to that
    many parts, splitting loses nothing and wins outright. Past it, one
    shortened message that arrives whole beats a queue that will not.

    The part count is the floor `fit_message` starts its search from, so a
    message needing more parts than this estimate still resolves to "split"
    only when the estimate says it is cheap, never the other way.
    """
    if mode != "auto":
        return mode
    if len(text) <= CHATBOX_LIMIT:
        return "split"
    estimated_parts = -(-len(text) // CHATBOX_LIMIT)
    return "split" if estimated_parts <= _MAX_REPEAT_PARTS else "truncate"


def fit_chatbox(text: str, mode: str) -> list[str]:
    """Fit `text` to VRChat's 144-char display limit per ``mode``: ``truncate``
    clips over-limit text to 143 characters plus an ellipsis, the cut backed
    off an attached character (`safe_cut`); ``split`` greedily packs whole
    words into <=144-char chunks (hard-splitting a lone over-long word);
    ``send`` passes through unchanged. Empty text -> ``[]``.
    """
    if not text:
        return []
    mode = resolve_overflow(text, mode)
    if mode == "send":
        return [text]
    if mode == "truncate":
        if len(text) <= CHATBOX_LIMIT:
            return [text]
        return [text[: safe_cut(text, CHATBOX_LIMIT - 1)] + "…"]
    if mode == "split":
        return _split_words(text, CHATBOX_LIMIT)
    raise ValueError(f"Unknown overflow mode: {mode!r}")


def fit_message(
    original: str, translations: list[tuple[str, str]], cfg: OscConfig
) -> list[str]:
    """Fit a caption and its translations into send-ready chatbox parts.

    "truncate" defers to `fit_chatbox`, but on a translation-aware text: if
    the plain `format_message` join overflows `CHATBOX_LIMIT` with
    `cfg.include_original` on, the original -- not the translation, the
    line a non-speaker actually reads -- is the one shortened to make room
    (see `_budget_original`), and whatever room is left is divided between
    the targets so that none of them is cut away outright (see
    `_share_limit`). "send" reaches `fit_chatbox` untouched, since its label
    promises the full text and names VRChat as what cuts it. In "split"
    mode an over-limit message is NOT
    greedy-packed as one joined string (that carves each language
    arbitrarily across part boundaries): instead every text is cut into the
    same number of balanced slices via `_balanced_slices` and part i joins
    slice i of each text with ``cfg.translation_separator``, so all
    languages advance together. Empty slices are omitted; a message that
    already fits comes back as one part.
    """
    joined = format_message(original, translations, cfg)
    mode = resolve_overflow(joined, cfg.overflow)
    if mode != "split":
        # "send" is labelled as handing VRChat the whole string and letting it
        # do the cutting, so neither reshaping step below applies to it.
        if mode == "truncate" and translations and len(joined) > CHATBOX_LIMIT:
            texts = [text for _, text in translations]
            if cfg.include_original:
                joined = _budget_original(original, texts, cfg.translation_separator)
            if len(texts) > 1 and len(joined) > CHATBOX_LIMIT:
                joined = cfg.translation_separator.join(
                    _share_limit(texts, cfg.translation_separator, CHATBOX_LIMIT)
                ).strip()
        return fit_chatbox(joined, mode)
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
        # No candidate left to gain: grown_repeated cannot outgrow a repeated
        # set that already holds every translation, so the extra _assemble
        # would be thrown away.
        if n + 1 <= _MAX_REPEAT_PARTS and len(repeated) < sum(translated):
            grown, grown_repeated = _assemble(texts, translated, n + 1, cfg)
            fits = grown and all(len(part) <= CHATBOX_LIMIT for part in grown)
            if fits and len(grown_repeated) > len(repeated):
                return _settle(
                    texts, translated, grown_repeated, n + 1, cfg, grown
                )
        return _settle(texts, translated, repeated, n, cfg, parts)
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
            cut = safe_cut(word, limit)
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
