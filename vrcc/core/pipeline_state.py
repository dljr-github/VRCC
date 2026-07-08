"""Thread-shared speculative-reuse and typing state for the pipeline.

Qt-free. Each method is exactly one acquisition of the class's single lock,
mirroring the pipeline's original inline lock blocks: callers never need two
calls where one lock block did two things.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vrcc.stt.engine import SttResult

# Distinguishes "no cache entry" from "cached a legitimate None result" (a
# speculative whose transcription was quality-gated to None still caches).
_MISSING = object()


class SpecCache:
    """Speculative-reuse state tying an utterance's early transcription to
    its final one. All state is guarded by one internal lock."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cache: dict[tuple[int, int], "SttResult | None"] = {}
        self._stale: set[tuple[int, int]] = set()
        self._pending: dict[int, int] = {}
        self._last_finalized = 0

    def reset(self) -> None:
        """Fresh run: never inherit a prior run's half-resolved utterances."""
        with self._lock:
            self._cache.clear()
            self._stale.clear()
            self._pending.clear()
            self._last_finalized = 0

    def note_speculative(self, utterance_id: int, samples_id: int) -> None:
        """Record a newly enqueued speculative and un-stale its key."""
        with self._lock:
            self._pending[utterance_id] = samples_id
            self._stale.discard((utterance_id, samples_id))

    def drop_discarded(self, utterance_id: int) -> None:
        """Discard: drop the pending speculative's cached result and mark its
        key stale so an in-flight transcription throws its result away."""
        with self._lock:
            samples_id = self._pending.pop(utterance_id, None)
            if samples_id is not None:
                key = (utterance_id, samples_id)
                self._cache.pop(key, None)
                self._stale.add(key)

    def store_result(self, key: tuple[int, int], result: "SttResult | None") -> bool:
        """Cache a finished speculative result. Returns False (clearing the
        stale mark) if the key was discarded while transcribing -- the caller
        throws the result away."""
        with self._lock:
            if key in self._stale:
                self._stale.discard(key)
                return False
            self._cache[key] = result
            return True

    def pop_result(self, key: tuple[int, int]) -> "SttResult | None | object":
        """Pop the cached result for a final, or ``_MISSING`` on a miss.
        Safe without extra locking: the single STT worker drains its queue
        FIFO and the speculative is always enqueued before its final, so the
        cache is populated before this lookup; a miss just re-transcribes."""
        with self._lock:
            return self._cache.pop(key, _MISSING)

    def mark_finalized(self, utterance_id: int) -> int:
        """Bound the caches: drop everything for utterances at or below the
        newest finalized one (their speculatives can never be reused now).
        Returns the cutoff so the caller can prune typing orphans."""
        with self._lock:
            if utterance_id > self._last_finalized:
                self._last_finalized = utterance_id
            cutoff = self._last_finalized
            self._cache = {k: v for k, v in self._cache.items() if k[0] > cutoff}
            self._stale = {k for k in self._stale if k[0] > cutoff}
            self._pending = {u: s for u, s in self._pending.items() if u > cutoff}
            return cutoff


class TypingTracker:
    """Typing-indicator bookkeeping: in-flight = typing on (speculative)
    until resolved (submit / gated-None / discard). Guarded by one lock."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._in_flight: set[int] = set()
        # Utterances whose typing-off is owned by a pending MT job: exempt
        # from the defensive prune (still in flight though finalized).
        self._owned_by_mt: set[int] = set()

    def reset(self) -> None:
        with self._lock:
            self._in_flight.clear()
            self._owned_by_mt.clear()

    def begin(self, utterance_id: int) -> None:
        with self._lock:
            self._in_flight.add(utterance_id)

    def resolve(self, utterance_id: int) -> bool:
        """Drop the utterance from both sets; True when nothing is left in
        flight (the caller turns the typing indicator off)."""
        with self._lock:
            self._in_flight.discard(utterance_id)
            self._owned_by_mt.discard(utterance_id)
            return not self._in_flight

    def own_by_mt(self, utterance_id: int) -> None:
        """Register MT ownership of typing-off (exempt from orphan pruning)."""
        with self._lock:
            self._owned_by_mt.add(utterance_id)

    def prune_orphans(self, cutoff: int) -> tuple[set[int], bool]:
        """Defense in depth: the segmenter invariant resolves every in-flight
        entry explicitly, so this should find nothing -- but prune orphans at
        or below ``cutoff`` anyway so a stuck speculative can't wedge typing
        on forever. MT-owned entries are exempt (still in flight though
        finalized). Returns (orphaned_ids, emptied)."""
        with self._lock:
            orphaned = {
                u
                for u in self._in_flight
                if u <= cutoff and u not in self._owned_by_mt
            }
            if orphaned:
                self._in_flight.difference_update(orphaned)
            return orphaned, bool(orphaned) and not self._in_flight
