"""Arrange a caption and its translations into balanced, size-limited
chatbox parts.

Given several texts that must ship together (an original plus its
translations), this module cuts each into the same number of ordered
slices and assembles those slices into parts that fit `CHATBOX_LIMIT`,
preferring an arrangement where a translation survives whole over one
that reads more evenly. It has no opinion on how many parts a message
needs: :mod:`vrcc.osc.chatbox_format` searches for that count and calls
in here once it has a candidate.
"""

from __future__ import annotations

from vrcc.core.config import OscConfig
from vrcc.osc.linebreak import choose_cut, is_spaceless, safe_cut

CHATBOX_LIMIT = 144


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
    nudged with `linebreak.choose_cut` to the nearest clause mark within
    half a slice either way. A slice spans two independently chosen cuts,
    so it can reach `size + 2 * (size // 2)` (just under twice its share),
    a structural bound, not an empirical one. One 98,924-slice corpus (n=2,
    n=3, 60-220 chars, clause marks every 6-28) measured roughly 300 slices
    over 1.5x their share; 0 needs two boundaries on the same index, only
    when n grossly exceeds what the text can support. `snap=False` is byte
    for byte what this function has always done, on both paths; the word
    path never reads `snap`, since a word boundary already reads right.
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
            # bounds[-1] + 1, not bounds[-1]: floor is an inclusive candidate
            # (lo = max(lo, floor) in linebreak.py), so the window's only
            # clause mark could be bounds[-1] itself, handed back unchanged
            # and collapsing this slice. Strictly past the previous boundary
            # belongs here, a stronger guarantee than the shared contract in
            # linebreak.choose_cut, pinned inclusive by its own tests.
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
        # len(candidate) == len(parts): _join drops any slot left empty by a
        # collapsed slice, so a shorter candidate could still pass the
        # per-part CHATBOX_LIMIT test while missing a whole part.
        if (
            candidate
            and len(candidate) == len(parts)
            and all(len(part) <= CHATBOX_LIMIT for part in candidate)
        ):
            return candidate
    return parts
