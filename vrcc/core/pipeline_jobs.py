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
    """Transcribe via ``p._transcribe`` and record the call's wall-clock cost.

    Timing starts here, after the job was dequeued, so queue wait is never
    counted. It does cover ``_stt_lock``, which ``p._transcribe`` takes and
    which a model swap or the 'heard' stream can hold for seconds: a run
    that swapped models reports those seconds as engine time.

    Nothing is recorded for ``_NO_ENGINE``, which means ``engine.transcribe``
    was never invoked (swapped out mid-flight), nor once ``stop`` is set: a
    call that outlasted stop()'s join returns into the next run, whose
    counters it must not touch."""
    start = time.monotonic()
    result = p._transcribe(samples)
    wall_s = time.monotonic() - start
    if result is _NO_ENGINE or stop.is_set():
        return result
    audio_s = len(samples) / SAMPLE_RATE
    p._stats.record_call(speculative, audio_s, wall_s)
    logger.debug(
        "stt call (%s): %.2fs audio, %.3fs wall",
        "speculative" if speculative else "final",
        audio_s,
        wall_s,
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
    # transcribe fresh. The spec lock is released before _transcribe (never
    # nested inside _stt_lock), preserving lock ordering.
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
    if p._mt is not None and p._config.translate.enabled:
        # Register MT ownership of typing-off BEFORE enqueueing, so the MT
        # worker can't resolve it before the exemption (_mark_finalized)
        # is visible.
        p._typing.own_by_mt(send_id)
        _enqueue(p, p._mt_queue, _MtJob(send_id, text, src, manage_typing=True))
    else:
        # Translation disabled: original phrase goes straight to chatbox.
        safe_submit(p, text, [], send_id)
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
        # Quality-gated: nothing downstream, just resolve typing.
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
        safe_submit(p, result.text, [], utterance_id)
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
    safe_submit(p, job.text, [], job.utterance_id)
    if job.manage_typing:
        p._resolve_typing(job.utterance_id)


def process_mt_job(p: "Pipeline", job: _MtJob, stop: "threading.Event") -> None:
    try:
        # Call the engine under _mt_lock so a concurrent detach_mt waits
        # before unloading; a None engine (disabled/swapped-out) -> send
        # original. Only _mt_lock held here (no lock-order cycle).
        with p._mt_lock:
            engine = p._mt
            # A target matching the source would only echo the transcription,
            # so the engine is never asked for it. Reachable when
            # source_language is "auto" (the GUI excludes an explicit source
            # from the target combos); "auto" resolves whisper "zh" to Chinese
            # Simplified, so a Chinese Traditional target keeps translating
            # (script conversion).
            all_targets = [languages.get(name) for name in p._config.translate.targets]
            targets = [lang for lang in all_targets if lang != job.src]
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
    safe_submit(p, job.text, submitted, job.utterance_id)
    if job.manage_typing:
        p._resolve_typing(job.utterance_id)
