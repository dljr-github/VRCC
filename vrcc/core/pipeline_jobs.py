"""Job creation and processing for the pipeline's STT/MT workers.

Module functions take the Pipeline instance ``p``: locks, engines, queues and
config stay Pipeline attributes -- only the per-job code lives here. Import
direction: pipeline imports this module (never the reverse at runtime).
"""

from __future__ import annotations

import logging
import queue
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from vrcc.audio.frames import SAMPLE_RATE
from vrcc.core import languages
from vrcc.core.events import (
    AppError,
    PhraseRecognized,
    PhraseTranslated,
)
from vrcc.core.pipeline_send import safe_submit
from vrcc.core.pipeline_state import _MISSING
from vrcc.translate import pinyin

if TYPE_CHECKING:
    import threading

    import numpy as np

    from vrcc.audio.segmenter import SegDiscard, SegFinal, SegSpeculative
    from vrcc.core.languages import Language
    from vrcc.core.pipeline import Pipeline
    from vrcc.stt.engine import SttResult

# Same logger as the orchestrator: one operational stream for the pipeline.
logger = logging.getLogger("vrcc.core.pipeline")

# Distinguishes "engine is being swapped out (None)" from a legitimate None
# transcription result (quality-gated): a job that sees _NO_ENGINE is dropped.
_NO_ENGINE = object()

# Blocked-enqueue poll: re-check the stop flag so stop() can't deadlock.
_PUT_POLL_S = 0.1


@dataclass
class _SttJob:
    utterance_id: int
    samples: "np.ndarray"
    speculative: bool
    samples_id: int


@dataclass
class _MtJob:
    utterance_id: int
    text: str
    src: "Language"
    manage_typing: bool


# -- shared-state helpers (queues/caches live on Pipeline; only the logic
# that touches them from job code lives here) --------------------------------


def _mark_finalized(p: "Pipeline", utterance_id: int) -> None:
    """Bound the speculative caches and prune typing orphans below the cutoff
    (see TypingTracker.prune_orphans)."""
    cutoff = p._spec.mark_finalized(utterance_id)
    orphaned, emptied = p._typing.prune_orphans(cutoff)
    if orphaned:
        logger.warning(
            "pruned orphaned typing entries %s (segmenter invariant "
            "violated?)",
            sorted(orphaned),
        )
    if emptied:
        p._set_typing(False)


def _finalize_dropped(p: "Pipeline", utterance_id: int) -> None:
    """Resolve typing and finalize a final dropped before forward_final
    (ids monotonic across runs, so a late/zombie drop is safe)."""
    p._resolve_typing(utterance_id)
    _mark_finalized(p, utterance_id)


def _enqueue(p: "Pipeline", q: "queue.Queue", job) -> None:
    """Put a job, applying backpressure (blocking) but waking to drop it if
    stop() is requested, so a full downstream queue never deadlocks stop."""
    while not p._stop_flag.is_set():
        try:
            q.put(job, timeout=_PUT_POLL_S)
            return
        except queue.Full:
            continue


def _submit(p: "Pipeline", original: str, translations: list, utterance_id: int) -> None:
    """safe_submit, then close the finalize-to-submission latency window
    handle_final opened for this id (see LatencyTracker); every send path
    below goes through here rather than calling safe_submit directly, so
    none of them can forget to close it."""
    safe_submit(p, original, translations, utterance_id)
    elapsed = p._latency.pop_elapsed(utterance_id)
    if elapsed is not None:
        p._stats.record_latency(elapsed)


# -- segmenter-event handlers (job creation) --------------------------------


def handle_speculative(p: "Pipeline", event: "SegSpeculative") -> None:
    """Enqueue a speculative non-blocking: a full queue means the engine is
    behind real time, and blocking here would stall the segmenter thread that
    is the only consumer draining the frame queue upstream. The final
    re-transcribes the same audio regardless, so a skip costs an early
    caption, never a caption -- unlike the blocking backpressure below, which
    real audio depends on."""
    if not p._should_caption():
        return
    samples_id = id(event.samples)
    # Noted before the put: the worker can dequeue the job the moment it
    # lands, and consume_stale has to find the key un-staled by then.
    p._spec.note_speculative(event.utterance_id, samples_id)
    try:
        p._stt_queue.put_nowait(
            _SttJob(event.utterance_id, event.samples, True, samples_id)
        )
    except queue.Full:
        p._spec.forget_speculative(event.utterance_id, samples_id)
        p._skipped_speculatives += 1
        logger.debug(
            "stt queue full; skipped speculative for utterance %s",
            event.utterance_id,
        )
        return
    p._begin_typing(event.utterance_id)


def handle_final(p: "Pipeline", event: "SegFinal") -> None:
    if not p._should_caption():
        # Gated at finalize time: no transcription. Still resolve any
        # typing indicator and bound the caches for this utterance.
        p._resolve_typing(event.utterance_id)
        _mark_finalized(p, event.utterance_id)
        return
    # Opens the finalize-to-chatbox-submission latency window (see
    # LatencyTracker); only once a job is actually going to be enqueued, so
    # a gate closed at this same check above never leaves a start unclosed
    # for longer than this run.
    p._latency.note_finalized(event.utterance_id)
    _enqueue(
        p,
        p._stt_queue,
        _SttJob(event.utterance_id, event.samples, False, id(event.samples)),
    )


def handle_discard(p: "Pipeline", event: "SegDiscard") -> None:
    p._spec.drop_discarded(event.utterance_id)
    p._resolve_typing(event.utterance_id)


# -- STT job processing ------------------------------------------------------


def _call_engine(
    p: "Pipeline", samples: "np.ndarray", stop: "threading.Event", *, speculative: bool
) -> "SttResult | None | object":
    """Transcribe via the STT slot, timing the lock wait and the engine call
    as two separate spans.

    Timing starts here, after the job was dequeued, so queue wait is never
    counted. borrow() holds the slot's lock for the whole block below, so a
    concurrent model swap or the 'heard' stream stalls ``wait_s`` (timed
    before ``engine`` is used) rather than the engine call itself
    (``call_s``, timed only around ``engine.transcribe``).

    Nothing is recorded for ``_NO_ENGINE``, which means ``engine.transcribe``
    was never invoked (swapped out mid-flight), nor once ``stop`` is set: a
    call that outlasted stop()'s join returns into the next run, whose
    counters it must not touch."""
    wait_start = time.monotonic()
    with p.stt_slot.borrow() as engine:
        wait_s = time.monotonic() - wait_start
        if engine is None:
            result, call_s = _NO_ENGINE, 0.0
        else:
            call_start = time.monotonic()
            result = engine.transcribe(samples)
            call_s = time.monotonic() - call_start
    if result is _NO_ENGINE or stop.is_set():
        return result
    audio_s = len(samples) / SAMPLE_RATE
    p._stats.record_call(speculative, audio_s, wait_s, call_s)
    logger.debug(
        "stt call (%s): %.2fs audio, %.3fs wait + %.3fs call",
        "speculative" if speculative else "final",
        audio_s,
        wait_s,
        call_s,
    )
    return result


def process_stt_job(p: "Pipeline", job: _SttJob, stop: "threading.Event") -> None:
    key = (job.utterance_id, job.samples_id)

    if job.speculative:
        if p._spec.consume_stale(key):
            # Discarded while queued (SegDiscard landed before this job
            # reached the engine): the result would only be thrown away, and
            # on CPU that wasted inference is what fills the queue that
            # blocks the segmenter. store_result's stale check still covers
            # the discard-while-transcribing case below.
            if not stop.is_set():
                p._stale_speculatives += 1
            return
        result = _call_engine(p, job.samples, stop, speculative=True)
        if result is _NO_ENGINE:
            return  # engine swapped out mid-flight: drop the job
        if stop.is_set():
            # Stopped (maybe restarted) mid-transcribe: this result belongs
            # to an abandoned run, must not touch a new run's shared state.
            return
        p._spec.store_result(key, result)
        return

    # Reuse the speculative's cached result on identical samples, else
    # transcribe fresh. The spec lock is released before _call_engine (never
    # nested inside the STT slot's lock), preserving lock ordering.
    result = p._spec.pop_result(key)
    if result is _MISSING:
        result = _call_engine(p, job.samples, stop, speculative=False)
        if result is _NO_ENGINE:
            _finalize_dropped(p, job.utterance_id)  # engine swapped out mid-flight
            return
    elif not stop.is_set():
        # Same guard _call_engine applies: a job that outlasted stop()'s join
        # returns into the next run, whose counters it must not touch.
        p._stats.record_reuse()
    if stop.is_set():
        _finalize_dropped(p, job.utterance_id)  # abandoned mid-call
        return
    forward_final(p, job.utterance_id, result)


def _send_caption(
    p, send_id, text, src, *, language="", avg_logprob=0.0, no_speech_prob=0.0
):
    """Publish the recognized caption for send_id and route its text to
    translation (owning typing-off via own_by_mt) or, MT disabled, straight to
    the chatbox with typing resolved. Does NOT finalize: the caller owns
    _mark_finalized. `src` is a resolved Language, never None."""
    p._bus.publish(
        PhraseRecognized(
            utterance_id=send_id,
            text=text,
            language=language,
            avg_logprob=avg_logprob,
            no_speech_prob=no_speech_prob,
        )
    )
    if p.mt_slot.current is not None and p._config.translate.enabled:
        # Register MT ownership of typing-off BEFORE enqueueing, so the MT
        # worker can't resolve it before the exemption (_mark_finalized)
        # is visible.
        p._typing.own_by_mt(send_id)
        _enqueue(p, p._mt_queue, _MtJob(send_id, text, src, manage_typing=True))
    else:
        # Translation disabled: original phrase goes straight to chatbox.
        _submit(p, text, [], send_id)
        p._resolve_typing(send_id)


def forward_final(p: "Pipeline", utterance_id: int, result: "SttResult | None") -> None:
    if not p._should_caption():
        # Re-check the gate at send time: enqueue-time gating (handle_final)
        # can't see a mute/captioning-off that landed while the STT job was in
        # flight. _mark_finalized resolves the caches and bounds them for this utterance.
        p._resolve_typing(utterance_id)
        _mark_finalized(p, utterance_id)
        return

    if result is None:
        # Quality-gated: nothing downstream, just resolve typing. Counted
        # here since engine.transcribe() itself only returns None, with no
        # further signal of why to a caller above it.
        p._input.record_gate_suppressed()
        p._resolve_typing(utterance_id)
        _mark_finalized(p, utterance_id)
        return

    src = resolve_source_language(p, result.language)

    if src is None:
        # "auto" detected a Whisper code with no registered Language: the MT
        # engine must never be told the wrong source (garbage translation
        # with no warning). Send the original untranslated instead.
        p._bus.publish(
            PhraseRecognized(
                utterance_id=utterance_id,
                text=result.text,
                language=result.language,
                avg_logprob=result.avg_logprob,
                no_speech_prob=result.no_speech_prob,
            )
        )
        logger.warning(
            "auto-detected language %r has no registered match; sending "
            "original text without translation",
            result.language,
        )
        p._bus.publish(
            AppError(
                "SOURCE_LANG_UNSUPPORTED",
                f"Detected language '{result.language}' is not supported "
                "for translation; sent untranslated.",
            )
        )
        # Same reason as send_untranslated: with sending off there is no
        # ChatboxSent either, and the caption row needs one of the two.
        p._bus.publish(
            PhraseTranslated(
                utterance_id=utterance_id,
                original=result.text,
                source_lang=result.language,
                translations=(),
            )
        )
        _submit(p, result.text, [], utterance_id)
        p._resolve_typing(utterance_id)
        _mark_finalized(p, utterance_id)
        return

    _send_caption(
        p, utterance_id, result.text, src,
        language=result.language, avg_logprob=result.avg_logprob,
        no_speech_prob=result.no_speech_prob,
    )
    _mark_finalized(p, utterance_id)


def resolve_source_language(p: "Pipeline", detected_whisper: str) -> "Language | None":
    """Resolve the MT source language, or ``None`` when "auto" detected a
    Whisper code the registry has no entry for (translation must be skipped,
    never mislabeled as English)."""
    src_cfg = p._config.stt.source_language
    if src_cfg != "auto":
        return languages.get(src_cfg)
    return languages.from_whisper(detected_whisper)


# -- MT job processing -------------------------------------------------------


def send_untranslated(p: "Pipeline", job: _MtJob) -> None:
    """Resolve an utterance that will not be translated after all: say so on
    the bus, then put the original text in the chatbox's slot.

    The empty PhraseTranslated is not cosmetic. With "send to VRChat" off,
    safe_submit returns without sending, so no ChatboxSent follows, and a
    caption row that hears neither event has nothing left that could ever
    resolve it: it sits on "translating…" for the rest of the session.
    """
    p._bus.publish(
        PhraseTranslated(
            utterance_id=job.utterance_id,
            original=job.text,
            source_lang=job.src.display,
            translations=(),
        )
    )
    _submit(p, job.text, [], job.utterance_id)
    if job.manage_typing:
        p._resolve_typing(job.utterance_id)


def process_mt_job(p: "Pipeline", job: _MtJob, stop: "threading.Event") -> None:
    if p._mt_queue.full():
        # Shed at the consumer, oldest job first: a queue still full after a
        # dequeue means the engine is falling behind real time. Sending this
        # job untranslated keeps the STT worker's _enqueue from blocking on
        # this queue; shedding at the producer instead would let a late
        # translation preempt an already-sent original (osc/chatbox.py's
        # coalesce-latest-wins clears the pending queue on every submit).
        if stop.is_set():
            return  # abandoned mid-call: discard, publish nothing
        send_untranslated(p, job)
        return
    try:
        # A target matching the source would only echo the transcription, so
        # the engine is never asked for it. Reachable when source_language is
        # "auto" (the GUI excludes an explicit source from the target
        # combos); "auto" resolves whisper "zh" to Chinese Simplified, so a
        # Chinese Traditional target keeps translating (script conversion).
        # Built inside the try: languages.get raises on a name the registry
        # doesn't know, and that must still fall through to send_untranslated
        # rather than reach _mt_loop's handler, which never submits.
        all_targets = [languages.get(name) for name in p._config.translate.targets]
        targets = [lang for lang in all_targets if lang != job.src]
        # Call the engine with the slot's lock held so a concurrent detach_mt
        # waits before unloading; a None engine (disabled/swapped-out) ->
        # send original. Only that lock held here (no lock-order cycle).
        with p.mt_slot.borrow() as engine:
            if engine is None:
                translations = None
            elif not targets:
                # Nothing left to translate: the empty result flows through the
                # normal publish path so the caption row resolves and the
                # original still reaches the chatbox.
                translations = []
            else:
                translations = engine.translate(job.text, job.src, targets)
    except Exception as exc:  # noqa: BLE001 -- translation must not drop the caption
        if stop.is_set():
            return  # abandoned mid-call: discard, publish nothing
        logger.exception("translation failed; sending original text")
        p._bus.publish(AppError("MT_JOB_FAILED", str(exc)))
        # Captions must not silently vanish: keep the ORIGINAL text.
        send_untranslated(p, job)
        return

    if translations is None:
        # Engine swapped out mid-flight (or absent): keep the ORIGINAL
        # text, the same graceful path the exception branch uses.
        if stop.is_set():
            return  # abandoned mid-call: discard, publish nothing
        send_untranslated(p, job)
        return

    if stop.is_set():
        return  # abandoned mid-call: discard, publish nothing
    if translations and p._config.translate.pinyin:
        # Annotate before publish so the caption log and the chatbox show the
        # same reading line.
        translations = pinyin.annotate(translations)
    p._bus.publish(
        PhraseTranslated(
            utterance_id=job.utterance_id,
            original=job.text,
            source_lang=job.src.display,
            translations=tuple(translations),
        )
    )
    submitted = translations
    if len(targets) < len(all_targets) and not p._config.osc.include_original:
        # Hiding the original presumes every configured target carries a
        # translation. A skipped source-matching target is served by the
        # original text itself, so that text re-enters the message in the
        # target's slot; without it, readers of the source language get
        # nothing at all.
        submitted = list(translations)
        for i, lang in enumerate(all_targets):
            if lang == job.src:
                submitted.insert(i, (lang.display, job.text))
    _submit(p, job.text, submitted, job.utterance_id)
    if job.manage_typing:
        p._resolve_typing(job.utterance_id)
