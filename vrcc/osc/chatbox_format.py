"""Fit captions/translations to VRChat's 144-char chatbox display limit.

Split out of :mod:`vrcc.osc.chatbox` (which re-exports these names) so that
module stays under the line cap: this half is pure text shaping, no OSC, no
threads.
"""

from __future__ import annotations

from vrcc.core.config import OscConfig
from vrcc.osc.linebreak import choose_cut, is_spaceless, safe_cut

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
        shortened = original[: budget - 1] + "…"
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
    clips over-limit text to ``text[:143] + "…"``; ``split`` greedily packs
    whole words into <=144-char chunks (hard-splitting a lone over-long word);
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
        return [text[: CHATBOX_LIMIT - 1] + "…"]
    if mode == "split":
        return _split_words(text, CHATBOX_LIMIT)
    raise ValueError(f"Unknown overflow mode: {mode!r}")


def _balanced_slices(
    text: str, n: int, limit: int, anchor: str = "start", snap: bool = False
) -> list[str]:
    """Split `text` into exactly `n` ordered slices of near-equal length.

    Word-based whenever the text splits into words that each fit `limit`:
    each slice takes whole words greedily toward a running
    remaining-length/remaining-slices target (last slice takes the rest), so
    joining the slices with spaces preserves every word in order.
    Character-based ceil-division runs only for spaceless scripts or a
    pathological over-long word, where the concatenation reproduces `text`
    exactly (boundaries are nudged off combining marks). Callers drop empty
    slices.

    Fewer words than slices leaves blank slices, positioned per `anchor`.
    ``"start"`` leaves them at the end, so the original fades out once
    exhausted. ``"end"`` puts them first, so a text that runs out early still
    lands in the final part: parts drain one at a time and a new utterance
    clears the queue, so the last part is the one still on screen when the user
    stops talking, and a translation absent from it is one the reader never
    ends up with.

    `snap` only changes the character path: each ceil-division boundary is
    nudged with `linebreak.choose_cut` to the nearest clause mark within half
    a slice either way. A slice sits between two independently chosen cuts,
    so its length ranges from 0 to `size + 2 * (size // 2)`, roughly twice
    its target share. Measured on 98,924 slices from a synthetic Japanese
    corpus at n=2 and n=3 (60-220 characters, clause marks every 6-28
    characters): 241 exceeded 1.5x their target share and the worst observed
    was 1.89x. 0 needs two adjacent boundaries landing on the same index,
    which is only reachable when far more slices are requested than the text
    has room for (see `_settle`'s part-count check, which rejects a snapped
    arrangement that collapsed a slice this way before it reaches a caller).
    `snap=False` is byte for byte what this function has always done, on
    both paths; word-based slicing never reads `snap` at all, since a word
    boundary already reads right and the clause vocabulary has nothing to
    add there.
    """
    words = text.split()
    # A spaceless run within its per-part share is better carried whole, which
    # _assemble already does where it fits.
    spaceless = is_spaceless(text) and len(text) > limit // n
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
        if anchor == "end":
            content = [piece for piece in slices if piece]
            slices = [""] * (n - len(content)) + content
        return slices
    size = -(-len(text) // n)  # ceil division
    bounds = [0]
    for i in range(1, n):
        ideal = min(i * size, len(text))
        if snap:
            # bounds[-1] + 1, not bounds[-1]: choose_cut's floor is the
            # lowest PERMISSIBLE return value and treats it as an inclusive
            # candidate (linebreak.py's lo = max(lo, floor)), so passing
            # bounds[-1] itself lets it hand back the previous boundary
            # unchanged when that is the only clause mark in the window,
            # collapsing this slice to empty. This call needs the stronger,
            # narrower guarantee of strictly after the previous boundary, so
            # it is expressed here rather than by narrowing choose_cut's
            # contract, which Task 1 pinned as inclusive with its own tests.
            cut = choose_cut(
                text, ideal, bounds[-1] + 1, ideal - size // 2, ideal + size // 2
            )
        else:
            cut = safe_cut(text, ideal)
        # Belt and braces: choose_cut already honors floor, but this is the
        # same guard the unsnapped path has always needed to stay monotonic.
        bounds.append(max(cut, bounds[-1]))
    bounds.append(len(text))
    return [text[bounds[i] : bounds[i + 1]] for i in range(n)]


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
        if n + 1 <= _MAX_REPEAT_PARTS:
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


def _settle(
    texts: list[str],
    translated: list[bool],
    repeated: set[int],
    n: int,
    cfg: OscConfig,
    parts: list[str],
) -> list[str]:
    """`parts`, replaced by the first better arrangement that still fits.

    Tries anchoring a sliced translation into the LAST parts and snapping
    character-based boundaries onto clause marks, in preference order:
    anchored and snapped, anchored alone, snapped alone. Anchoring outranks
    snapping because anchoring decides whether the reader receives the
    translation at all (a translation with fewer words than parts leaves the
    trailing ones blank, and the last part is the one still on screen once
    the queue drains), while snapping only decides where the text looks
    broken. Applied only after the part count is settled, and only when every
    part still fits: the part-count search must keep seeing the plain
    ceil-division sizes it has always searched over, or the delivery numbers
    in `fit_message`'s comment stop describing what ships.

    Skips straight to returning `parts` when every translation is already
    repeated whole rather than sliced. Only a sliced translation needs
    relocating, so anchoring is a guaranteed no-op here; snapping is skipped
    too, even though the original can still be spaceless text long enough to
    want it, because every translation already arrives whole in every part,
    which is the outcome the part-count search was optimizing for.
    """
    if all(i in repeated for i, is_tr in enumerate(translated) if is_tr):
        return parts
    # Preference order: anchored and snapped, anchored alone, snapped alone.
    for anchored, snap in ((True, True), (True, False), (False, True)):
        candidate = _join(
            texts, translated, repeated, n, cfg, anchored=anchored, snap=snap
        )
        # len(candidate) == len(parts): _join drops any slot whose pieces are
        # all empty, so an arrangement that collapses one slice to nothing can
        # come back one part short of what fit_message already committed to
        # sending. Every remaining part can still individually fit
        # CHATBOX_LIMIT while the arrangement as a whole has silently dropped
        # one, so the count has to be checked on its own, not folded into the
        # per-part length test.
        if (
            candidate
            and len(candidate) == len(parts)
            and all(len(part) <= CHATBOX_LIMIT for part in candidate)
        ):
            return candidate
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
            for part in _join(texts, translated, candidate, n, cfg)
        ):
            repeated = candidate
    return _join(texts, translated, repeated, n, cfg), repeated


def _join(
    texts: list[str],
    translated: list[bool],
    repeated: set[int],
    n: int,
    cfg: OscConfig,
    anchored: bool = False,
    snap: bool = False,
) -> list[str]:
    """`n` parts, with the `repeated` texts whole in each and the rest sliced.

    ``anchored`` moves a sliced translation's content to the LAST parts.
    ``snap`` nudges each character-based boundary onto the nearest clause
    mark (see `_balanced_slices`). Both default off while a part count is
    being judged: anchoring changes which original slice a translation shares
    a part with, and snapping changes slice lengths, and a count that only
    fits because of either would be chosen over a larger one that the search
    would otherwise have picked, which measured better for anchoring and is
    the whole reason snapping runs after the search rather than inside it.
    """
    sliced = {
        i: _balanced_slices(
            text,
            n,
            CHATBOX_LIMIT,
            anchor="end" if anchored and translated[i] else "start",
            snap=snap,
        )
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
