"""Arrange a caption and its translations into balanced, size-limited
chatbox parts.

Cuts every text that ships together into the same number of ordered slices
and assembles those into parts fitting `CHATBOX_LIMIT`, preferring an
arrangement where a translation survives whole. How many parts a message
needs is decided in :mod:`vrcc.osc.chatbox_format`, which calls in here.
"""

from __future__ import annotations

from vrcc.core.config import OscConfig
from vrcc.osc.linebreak import choose_cut, is_spaceless, slice_cut

CHATBOX_LIMIT = 144


def _balanced_slices(
    text: str, n: int, limit: int, anchor: str = "start", snap: bool = False
) -> list[str]:
    """Split `text` into exactly `n` ordered slices of near-equal length.

    Word-based whenever every word fits `limit` AND no word is both
    spaceless-script and over its per-slice share (`limit // n`), each slice
    taking whole words greedily toward a remaining-length/remaining-slices
    target. Otherwise character-based ceil-division, each boundary backed off
    an attached character and out of any ASCII word it would sever
    (`linebreak.slice_cut`), whose concatenation reproduces `text` exactly.
    Callers drop empty slices.

    Fewer words than slices leaves blanks, per `anchor`: ``"start"`` puts
    them last, ``"end"`` first so a short text still reaches the final part.

    `snap` only changes the character path: each ceil-division boundary is
    nudged with `linebreak.choose_cut` to the nearest clause mark within
    half a slice of its grid position, and never nearer than half a slice to
    the cut before it, so no interior slice falls under half its share or
    over twice it, property-checked over generated text by
    `tests.test_chatbox_split.test_snap_keeps_interior_slices_within_half_to_double_share`.
    The word path ignores `snap`: a word boundary reads right.
    """
    words = text.split()
    # A spaceless run inside its per-part share travels better whole, which
    # _assemble already does. Checked per WORD: a translation carrying one
    # stray ASCII space (normalize leaves `!` and `?` alone) still holds a
    # spaceless word over its share. Korean stays on the word path however
    # long a token runs: hangul is not an unspaced script, so is_spaceless
    # is false of it (see linebreak.is_spaceless).
    # Length first: it is a C-level int compare where is_spaceless is a
    # Python scan of every character, and this runs on every word of every
    # text on every part count the search tries.
    share = limit // n
    spaceless = any(len(word) > share and is_spaceless(word) for word in words)
    if words and not spaceless and all(len(word) <= limit for word in words):
        slices: list[str] = []
        idx = 0
        for k in range(n - 1):
            target = len(" ".join(words[idx:])) / (n - k)
            piece = words[idx]
            idx += 1
            while idx < len(words):
                grown = len(piece) + 1 + len(words[idx])
                # Take the next word only while it moves the slice closer to
                # the target; overshoot stays within one word.
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
            # The floor is the previous cut plus one (an inclusive candidate
            # handed back unchanged would collapse this slice), and the window
            # opens no nearer than half a slice past that cut: measured from
            # the grid alone, two cuts drawn toward the same marks leave a few
            # characters between them.
            cut = choose_cut(
                text,
                ideal,
                bounds[-1] + 1,
                max(ideal - size // 2, bounds[-1] + size // 2),
                ideal + size // 2,
            )
        else:
            # Same lower bound the snapped window opens at: a Latin run
            # wider than half a slice is severed rather than kept whole,
            # since walking to its start would empty this slice and hand the
            # rest of the text to the next one.
            cut = slice_cut(text, ideal, max(bounds[-1], ideal - size // 2))
        # Neither path is asked to honor the previous bound below its own
        # floor, and both need bounds monotonic.
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

    ``anchored`` moves a sliced translation's content to the LAST parts;
    ``snap`` nudges each character-based boundary onto the nearest clause
    mark. Both default off while a part count is being judged, or a count
    that only fits because of one of them would win over the larger one the
    search would otherwise have picked."""
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
        # Stripped per piece, not once over the join: a character-path slice
        # can open or close on the stray ASCII space a translation carries,
        # and inside a separator that space ships as an indented line.
        pieces = [piece.strip() for piece in pieces]
        part = cfg.translation_separator.join(p for p in pieces if p)
        if part:
            parts.append(part)
    return parts


def _assemble(
    texts: list[str], translated: list[bool], n: int, cfg: OscConfig
) -> tuple[list[str], set[int]]:
    """Build `n` parts, repeating each translation that fits rather than
    slicing it. Returns the parts and which texts were repeated.

    Parts drain one every `max(split_delay_s, min_interval_s)` and a new
    utterance clears the queue, so anything past roughly the third part is
    rarely seen at conversational pace. A short translation repeated in
    EVERY part arrives in the first and is still on screen when the user
    stops talking: measured over a 12-turn conversation at four paces, that
    beat slicing on delivery and on what survives, in every combination
    tried. Repeating goes shortest first and only while every part fits, so
    a long translation falls back to slicing; the original is always sliced,
    being the one text a non-speaker is not reading.
    """
    # Shortest first so a long translation cannot crowd out two short ones it
    # could not fit beside anyway. Swept 2875 realistic messages with the
    # order reversed and the output was identical every time: insurance, not
    # a measured win, kept because it can only help.
    order = sorted(
        (i for i, is_tr in enumerate(translated) if is_tr), key=lambda i: len(texts[i])
    )
    repeated: set[int] = set()
    parts: list[str] | None = None
    for i in order:
        candidate = repeated | {i}
        grown = _join(texts, translated, candidate, n, cfg)
        if all(len(part) <= CHATBOX_LIMIT for part in grown):
            repeated, parts = candidate, grown
    if parts is None:
        parts = _join(texts, translated, repeated, n, cfg)
    return parts, repeated


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
    character-based boundaries onto clause marks, in that preference order:
    anchoring decides whether the reader receives the translation at all,
    snapping only where the text looks broken. Runs after the part count is
    settled and only when every part still fits, so the search keeps seeing
    the plain ceil-division sizes `fit_message`'s delivery numbers describe.

    Anchoring has nothing to move when every translation is repeated whole,
    or when there are none (a caption with translation off is an ordinary
    shape: `pipeline_send.safe_submit` takes an empty translation list from
    five call sites). The ORIGINAL is never repeated, so its cuts still want
    snapping in both, and the snapped arrangement alone is tried there."""
    movable = any(is_tr and i not in repeated for i, is_tr in enumerate(translated))
    anchoring = [(True, True), (True, False)] if movable else []
    for anchored, snap in [*anchoring, (False, True)]:
        candidate = _join(
            texts, translated, repeated, n, cfg, anchored=anchored, snap=snap
        )
        # len(candidate) == len(parts): _join drops any slot left empty by a
        # collapsed slice, so a shorter candidate can pass the per-part limit
        # while missing a whole part.
        if (
            candidate
            and len(candidate) == len(parts)
            and all(len(part) <= CHATBOX_LIMIT for part in candidate)
        ):
            return candidate
    return parts
