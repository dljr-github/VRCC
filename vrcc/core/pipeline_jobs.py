"""Job creation and processing for the pipeline's STT/MT workers.

Module functions take the Pipeline instance ``p``: locks, engines, queues and
config stay Pipeline attributes -- only the per-job code lives here. Import
direction: pipeline imports this module (never the reverse at runtime).
"""

from __future__ import annotations

import logging
import queue
from dataclasses import dataclass
from typing import TYPE_CHECKING

from vrcc.core import languages
from vrcc.core.events import (
    AppError,
    PhraseRecognized,
    PhraseTranslated,
)
from vrcc.core.pipeline_send import safe_submit
from vrcc.core.pipeline_state import _MISSING
from vrcc.core.sentences import (
    ends_sentence,
    followed_complete_sentences,
    split_sentences,
)

if TYPE_CHECKING:
    import threading

    import numpy as np

    from vrcc.audio.segmenter import SegDiscard, SegFinal, SegPartial, SegSpeculative
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

# Early-commit confidence floor. A committed sentence is already sent, so a
# partial must be confident to commit one early: below these a noisy partial
# can commit a sentence the full-context final would get right (measured on
# loud babble). Stricter than the STT's own hallucination gate, on purpose.
_COMMIT_MAX_NO_SPEECH = 0.3
_COMMIT_MIN_LOGPROB = -0.35


@dataclass
class _SttJob:
    utterance_id: int
    samples: "np.ndarray"
    speculative: bool
    samples_id: int
    # A live partial: transcribe-and-publish only, never stored/forwarded/
    # finalized (see the top-of-function branch in process_stt_job).
    partial: bool = False


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
    p._commits.clear(utterance_id)


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
    if not p._should_caption():
        return
    samples_id = id(event.samples)
    p._spec.note_speculative(event.utterance_id, samples_id)
    p._begin_typing(event.utterance_id)
    _enqueue(
        p,
        p._stt_queue,
        _SttJob(event.utterance_id, event.samples, True, samples_id),
    )


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
    p._commits.clear(event.utterance_id)
    p._resolve_typing(event.utterance_id)


def handle_partial(p: "Pipeline", event: "SegPartial") -> None:
    """Queue at most one in-flight partial transcription. Additive to the
    speculative/final flow: never touches SpecCache or typing, never begins
    typing (the speculative pass already owns that indicator)."""
    if not p._config.vad.sentence_inject:
        return
    if not p._should_caption():
        return
    with p._partial_lock:
        if p._partial_pending:
            return  # one already in flight: this snapshot is coalesced away
        p._partial_pending = True
    _enqueue(
        p,
        p._stt_queue,
        _SttJob(
            event.utterance_id,
            event.samples,
            speculative=False,
            samples_id=id(event.samples),
            partial=True,
        ),
    )


# -- STT job processing ------------------------------------------------------


def process_stt_job(p: "Pipeline", job: _SttJob, stop: "threading.Event") -> None:
    if job.partial:
        _process_partial_job(p, job, stop)
        return

    key = (job.utterance_id, job.samples_id)

    if job.speculative:
        result = p._transcribe(job.samples)
        if result is _NO_ENGINE:
            return  # engine swapped out mid-flight: drop the job
        if stop.is_set():
            # Stopped (maybe restarted) mid-transcribe: this result belongs
            # to an abandoned run, must not touch a new run's shared state.
            return
        stored = p._spec.store_result(key, result)
        if stored and _should_inject_sentence(p, result):
            # A finished sentence: cut the utterance FIRST, then send it.
            # forward_final enqueues the MT job through the blocking _enqueue;
            # request_commit only sets the segmenter's commit flag (read on its
            # next frame), so cutting first stops the segmenter appending the
            # next sentence's onset to the still-open buffer during that block.
            # mark_emitted_early stays AFTER forward_final: its _mark_finalized
            # prunes _emitted_early at cutoff==uid, so marking earlier would
            # erase the guard that dedupes a natural final racing the commit.
            # The single FIFO STT worker sets that guard before it could dequeue
            # the racing final, so pop_emitted_early still holds.
            p.segmenter.request_commit(job.utterance_id)
            forward_final(p, job.utterance_id, result)
            p._spec.mark_emitted_early(job.utterance_id)
        return

    # Final: a sentence already emitted early from the speculative pass must
    # not send twice. This only fires in the race where the natural final was
    # queued before request_commit cut the utterance; the common commit path
    # emits no final at all.
    if p._spec.pop_emitted_early(job.utterance_id):
        # The early send's forward_final may have handed typing-off to a
        # still-running MT job (own_by_mt): resolving it here as well would
        # turn the indicator off while that job is still translating.
        if not p._typing.is_owned_by_mt(job.utterance_id):
            p._resolve_typing(job.utterance_id)
        _mark_finalized(p, job.utterance_id)
        return

    # Reuse the speculative's cached result on identical samples, else
    # transcribe fresh. The spec lock is released before _transcribe (never
    # nested inside _stt_lock), preserving lock ordering.
    result = p._spec.pop_result(key)
    if result is _MISSING:
        result = p._transcribe(job.samples)
        if result is _NO_ENGINE:
            _finalize_dropped(p, job.utterance_id)  # engine swapped out: resolve typing and bound the caches
            return
    if stop.is_set():
        _finalize_dropped(p, job.utterance_id)  # abandoned: resolve typing and bound the caches
        return
    forward_final(p, job.utterance_id, result)


def _process_partial_job(p: "Pipeline", job: _SttJob, stop: "threading.Event") -> None:
    """Transcribe the growing utterance buffer and commit any newly-stable
    complete sentence to the chatbox (its own message, its own id), never
    finalizing the utterance and never resetting the segmenter buffer (that
    would clip the next sentence's onset). The pending flag is cleared right
    after transcribe on every path, so a coalesced partial can always fire
    again."""
    try:
        result = p._transcribe(job.samples)
    finally:
        with p._partial_lock:
            p._partial_pending = False
    if stop.is_set():
        return
    if result is _NO_ENGINE or result is None:
        return
    if stop.is_set() or not p._should_caption():
        return
    if job.utterance_id <= p._spec.last_finalized():
        return
    _commit_stable_sentences(p, job.utterance_id, result)


def _commit_stable_sentences(p: "Pipeline", utterance_id: int, result: "SttResult") -> None:
    """Commit each complete sentence that is followed by more text AND stable
    across two consecutive partials (CommitTracker.stable_new). Each committed
    sentence gets its own distinct id so the chatbox does not coalesce
    successive sentences and the log shows each as its own row. No
    request_commit (the buffer must keep growing to preserve the next
    sentence's onset); no finalize (the utterance is still active)."""
    cfg = p._config.vad
    if not cfg.sentence_inject:
        return
    if (result.no_speech_prob >= _COMMIT_MAX_NO_SPEECH
            or result.avg_logprob <= _COMMIT_MIN_LOGPROB):
        return  # low-confidence partial: let the whole-utterance final handle it
    followed = followed_complete_sentences(result.text, cfg.sentence_min_words)
    new = p._commits.stable_new(utterance_id, followed)
    if not new:
        return
    src = resolve_source_language(p, result.language)
    if src is None:
        return  # "auto" hit an unregistered language: let the final send it untranslated
    for sentence in new:
        _send_caption(p, p._next_message_id(), sentence, src, language=result.language)


def _should_inject_sentence(p: "Pipeline", result: "SttResult | None") -> bool:
    """Whether a speculative result is a complete sentence worth sending now
    (feature enabled, non-empty result, confident partial, terminal
    punctuation past the minimum word count). A low-confidence pause
    snapshot is left for the whole-utterance final, which is more accurate
    in noise."""
    cfg = p._config.vad
    return (
        cfg.sentence_inject
        and result is not None
        and result.no_speech_prob < _COMMIT_MAX_NO_SPEECH
        and result.avg_logprob > _COMMIT_MIN_LOGPROB
        and ends_sentence(result.text, cfg.sentence_min_words)
    )


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
        safe_submit(p, result.text, [], utterance_id)
        p._resolve_typing(utterance_id)
        _mark_finalized(p, utterance_id)
        return

    sentences = split_sentences(result.text)
    remaining = p._commits.uncommitted(utterance_id, sentences)
    if sentences and not remaining:
        # Every sentence already streamed out by the partials or the pause,
        # so there is nothing new to send.
        p._resolve_typing(utterance_id)
        _mark_finalized(p, utterance_id)
        return
    if len(remaining) == len(sentences):
        # No sentence streamed early (or the text has no sentence boundary):
        # the whole transcription goes out as one caption under the utterance id.
        _send_caption(
            p,
            utterance_id,
            result.text,
            src,
            language=result.language,
            avg_logprob=result.avg_logprob,
            no_speech_prob=result.no_speech_prob,
        )
        _mark_finalized(p, utterance_id)
        return
    # Some sentences streamed early. Send the rest one caption each, in text
    # order, so a short sentence the partials skipped is never merged with a
    # non-adjacent tail. Each gets its own id, like the partial commits; the
    # per-caption sends own their typing-off, and the utterance's own typing
    # (begun only at a pause) is resolved here so _mark_finalized does not
    # prune it as an orphan.
    for sentence in remaining:
        _send_caption(p, p._next_message_id(), sentence, src, language=result.language)
    p._resolve_typing(utterance_id)
    _mark_finalized(p, utterance_id)


def resolve_source_language(p: "Pipeline", detected_whisper: str) -> "Language | None":
    """Resolve the MT source language, or ``None`` when "auto" detected a
    Whisper code the registry has no entry for (translation must be skipped,
    never mislabeled as English)."""
    src_cfg = p._config.stt.source_language
    if src_cfg != "auto":
        return languages.get(src_cfg)
    # "auto": map the detected Whisper code to the first matching Language.
    for lang in languages.LANGUAGES.values():
        if lang.whisper == detected_whisper:
            return lang
    return None


# -- MT job processing -------------------------------------------------------


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
        # Captions must not silently vanish: send the ORIGINAL text.
        safe_submit(p, job.text, [], job.utterance_id)
        if job.manage_typing:
            p._resolve_typing(job.utterance_id)
        return

    if translations is None:
        # Engine swapped out mid-flight (or absent): send the ORIGINAL
        # text, the same graceful path the exception branch uses.
        if stop.is_set():
            return  # abandoned mid-call: discard, publish nothing
        safe_submit(p, job.text, [], job.utterance_id)
        if job.manage_typing:
            p._resolve_typing(job.utterance_id)
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
