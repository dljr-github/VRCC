"""Translate a speculative transcription ahead of its final.

A final that reuses its speculative's text would otherwise pay the MT call
after finalizing. The speculative's text is offered to the MT worker as a
low-priority prefetch whose result is cached; the real job takes it from the
cache when utterance, text, source, targets and engine all match.

A prefetch is invisible to everything but this cache: it publishes no events,
submits nothing and never touches the typing indicator. It also never costs a
real job anything: it is only queued with spare room, dropped at dequeue if
real work is waiting, and ignored by the MT worker's shed check.
"""

from __future__ import annotations

import logging
import queue
import threading
import weakref
from collections import OrderedDict
from dataclasses import dataclass
from typing import TYPE_CHECKING

from vrcc.core import languages

if TYPE_CHECKING:
    from vrcc.core.languages import Language
    from vrcc.core.pipeline import Pipeline
    from vrcc.stt.engine import SttResult

logger = logging.getLogger("vrcc.core.pipeline")

# Off switch for benchmarks that need the plain path.
ENABLED = True

# Entries are one per utterance and consumed by the real job; the bound only
# matters when finals never arrive (discards, a stalled MT worker).
CACHE_MAX = 4

_Key = tuple  # (utterance_id, text, src, tuple(targets))


@dataclass
class _PrefetchJob:
    utterance_id: int
    text: str
    src: "Language"
    manage_typing: bool = False


class PretranslateCache:
    """Cached prefetch translations and the counters for the session summary.
    One lock guards both; counters survive reset(), the cache does not."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._entries: "OrderedDict[int, tuple[_Key, weakref.ref, tuple]]" = OrderedDict()
        self._queued = 0
        self.offered = 0
        self.hit = 0
        self.miss = 0
        self.wasted = 0
        self.dropped = 0

    def reset(self) -> None:
        with self._lock:
            self.wasted += len(self._entries)
            self._entries.clear()
            self._queued = 0

    def queued(self) -> int:
        with self._lock:
            return self._queued

    def note_queued(self) -> None:
        with self._lock:
            self._queued += 1
            self.offered += 1

    def note_dequeued(self, *, put_failed: bool = False) -> None:
        with self._lock:
            self._queued = max(0, self._queued - 1)
            if put_failed:
                self.offered -= 1
                self.dropped += 1

    def note_dropped(self) -> None:
        with self._lock:
            self.dropped += 1

    def store(self, key: _Key, engine, translations) -> None:
        try:
            ref = weakref.ref(engine)
        except TypeError:
            self.note_dropped()
            return
        with self._lock:
            if key[0] in self._entries:
                self.wasted += 1
            self._entries[key[0]] = (key, ref, tuple(translations))
            self._entries.move_to_end(key[0])
            while len(self._entries) > CACHE_MAX:
                self._entries.popitem(last=False)
                self.wasted += 1

    def take(self, key: _Key, engine) -> "list | None":
        """Pop the utterance's entry. Usable only on an exact key and engine
        match; anything else counts as wasted work and a miss."""
        with self._lock:
            entry = self._entries.pop(key[0], None)
            if entry is not None and entry[0] == key and entry[1]() is engine:
                self.hit += 1
                return list(entry[2])
            if entry is not None:
                self.wasted += 1
            self.miss += 1
            return None

    def forget(self, utterance_id: int) -> None:
        with self._lock:
            if self._entries.pop(utterance_id, None) is not None:
                self.wasted += 1


def _targets_for(p: "Pipeline", src: "Language") -> list:
    all_targets = [languages.get(name) for name in p._config.translate.targets]
    return [lang for lang in all_targets if lang != src]


def offer(
    p: "Pipeline", utterance_id: int, result: "SttResult | None", stored: bool
) -> None:
    """Queue a prefetch for a freshly stored speculative result. Never
    blocks the STT worker and never takes the last free MT queue slot."""
    if not (ENABLED and stored) or result is None or not result.text:
        return
    if not p._config.translate.enabled or p.mt_slot.current is None:
        return
    # Late import: pipeline_jobs imports this module.
    from vrcc.core.pipeline_jobs import resolve_source_language

    src = resolve_source_language(p, result.language)
    if src is None:
        return
    q = p._mt_queue
    if q.qsize() >= q.maxsize - 1:
        return
    p._pretranslate.note_queued()
    try:
        q.put_nowait(_PrefetchJob(utterance_id, result.text, src))
    except queue.Full:
        p._pretranslate.note_dequeued(put_failed=True)


def real_queue_full(p: "Pipeline") -> bool:
    """MT queue fullness with queued prefetches not counted, so one can
    never push a real job into the shed path."""
    q = p._mt_queue
    return q.qsize() - p._pretranslate.queued() >= q.maxsize


def run(p: "Pipeline", job: _PrefetchJob, stop: "threading.Event") -> None:
    cache = p._pretranslate
    cache.note_dequeued()
    if stop.is_set() or p._mt_queue.qsize() > 0:
        cache.note_dropped()
        return
    try:
        targets = _targets_for(p, job.src)
        if not targets:
            cache.note_dropped()
            return
        with p.mt_slot.borrow() as engine:
            if engine is None:
                cache.note_dropped()
                return
            translations = engine.translate(job.text, job.src, targets)
    except Exception:  # noqa: BLE001 -- a prefetch failure is never user visible
        logger.debug("pretranslate failed", exc_info=True)
        cache.note_dropped()
        return
    if stop.is_set() or translations is None:
        cache.note_dropped()
        return
    key = (job.utterance_id, job.text, job.src, tuple(targets))
    cache.store(key, engine, translations)


def take(p: "Pipeline", job, targets: list, engine) -> "list | None":
    """Cached translations for the real job, or None to translate now."""
    key = (job.utterance_id, job.text, job.src, tuple(targets))
    return p._pretranslate.take(key, engine)


def log_summary(p: "Pipeline") -> None:
    c = p._pretranslate
    if not c.offered:
        return
    logger.info(
        "Pretranslate: %d offered, %d used, %d missed, %d wasted, %d dropped.",
        c.offered, c.hit, c.miss, c.wasted, c.dropped,
    )
