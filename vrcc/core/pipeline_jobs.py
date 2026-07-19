"""Job creation and processing for the pipeline's STT/MT workers.

Module functions take the Pipeline instance ``p``: locks, engines, queues and
config stay Pipeline attributes -- only the per-job code lives here. Import
direction: pipeline imports this module (never the reverse at runtime).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from vrcc.core import languages
from vrcc.core.events import AppError, PhrasePartial, PhraseRecognized, PhraseTranslated
from vrcc.core.pipeline_state import _MISSING
from vrcc.core.sentences import ends_sentence

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


# -- segmenter-event handlers (job creation) --------------------------------


def handle_speculative(p: "Pipeline", event: "SegSpeculative") -> None:
    if not p._should_caption():
        return
    samples_id = id(event.samples)
    p._spec.note_speculative(event.utterance_id, samples_id)
    p._begin_typing(event.utterance_id)
    p._enqueue(
        p._stt_queue,
        _SttJob(event.utterance_id, event.samples, True, samples_id),
    )


def handle_final(p: "Pipeline", event: "SegFinal") -> None:
    if not p._should_caption():
        # Gated at finalize time: no transcription. Still resolve any
        # typing indicator and bound the caches for this utterance.
        p._resolve_typing(event.utterance_id)
        p._mark_finalized(event.utterance_id)
        return
    p._enqueue(
        p._stt_queue,
        _SttJob(event.utterance_id, event.samples, False, id(event.samples)),
    )


def handle_discard(p: "Pipeline", event: "SegDiscard") -> None:
    p._spec.drop_discarded(event.utterance_id)
    p._resolve_typing(event.utterance_id)


def handle_partial(p: "Pipeline", event: "SegPartial") -> None:
    """Queue at most one in-flight partial transcription. Additive to the
    speculative/final flow: never touches SpecCache or typing, never begins
    typing (the speculative pass already owns that indicator)."""
    if not p._config.vad.live_partials:
        return
    if not p._should_caption():
        return
    with p._partial_lock:
        if p._partial_pending:
            return  # one already in flight: this snapshot is coalesced away
        p._partial_pending = True
    p._enqueue(
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
            # A finished sentence: send it now and cut the utterance so the
            # next words become a fresh one. forward_final finalizes this id
            # (which prunes the caches for it), so mark the emitted-early guard
            # AFTER, where it survives to dedupe a natural final racing the
            # commit; a later utterance's finalize is what eventually clears it.
            # A second inject for this id cannot form: a resume racing this
            # speculative would have gone through SegDiscard -> drop_discarded
            # and made store_result return False above; since it did not, the
            # next process() frame consumes this request_commit and starts a
            # fresh id before another speculative for this one can form.
            forward_final(p, job.utterance_id, result)
            p._spec.mark_emitted_early(job.utterance_id)
            p.segmenter.request_commit(job.utterance_id)
        return

    # Final: a sentence already emitted early from the speculative pass must
    # not send twice. This only fires in the race where the natural final was
    # queued before request_commit cut the utterance; the common commit path
    # emits no final at all.
    if p._spec.pop_emitted_early(job.utterance_id):
        p._resolve_typing(job.utterance_id)
        p._mark_finalized(job.utterance_id)
        return

    # Reuse the speculative's cached result on identical samples, else
    # transcribe fresh. The spec lock is released before _transcribe (never
    # nested inside _stt_lock), preserving lock ordering.
    result = p._spec.pop_result(key)
    if result is _MISSING:
        result = p._transcribe(job.samples)
        if result is _NO_ENGINE:
            return  # engine swapped out mid-flight: drop the job
    if stop.is_set():
        return  # abandoned mid-call: discard, publish nothing
    forward_final(p, job.utterance_id, result)


def _process_partial_job(p: "Pipeline", job: _SttJob, stop: "threading.Event") -> None:
    """Transcribe-and-publish only: never touches SpecCache, forward_final,
    mark_finalized, or typing. The pending flag is cleared right after
    transcribe, on every path (stop/no-engine/gated-None/exception included),
    so a coalesced partial is always free to fire again."""
    try:
        result = p._transcribe(job.samples)
    finally:
        with p._partial_lock:
            p._partial_pending = False
    if stop.is_set():
        return  # abandoned mid-call: discard, publish nothing
    if result is _NO_ENGINE or result is None:
        return  # engine swapped out, or quality-gated: nothing to show
    p._bus.publish(PhrasePartial(job.utterance_id, result.text))
    if p._config.osc.send_to_vrchat:
        safe_submit_partial(p, result.text)


def _should_inject_sentence(p: "Pipeline", result: "SttResult | None") -> bool:
    """Whether a speculative result is a complete sentence worth sending now
    (feature enabled, non-empty result, terminal punctuation past the
    minimum word count)."""
    cfg = p._config.vad
    return (
        cfg.sentence_inject
        and result is not None
        and ends_sentence(result.text, cfg.sentence_min_words)
    )


def forward_final(p: "Pipeline", utterance_id: int, result: "SttResult | None") -> None:
    if result is None:
        # Quality-gated: nothing downstream, just resolve typing.
        p._resolve_typing(utterance_id)
        p._mark_finalized(utterance_id)
        return

    p._bus.publish(
        PhraseRecognized(
            utterance_id=utterance_id,
            text=result.text,
            language=result.language,
            avg_logprob=result.avg_logprob,
            no_speech_prob=result.no_speech_prob,
        )
    )
    src = resolve_source_language(p, result.language)

    if p._mt is not None and p._config.translate.enabled:
        # Register MT ownership of typing-off BEFORE enqueueing, so the MT
        # worker can't resolve it before the exemption (_mark_finalized)
        # is visible.
        p._typing.own_by_mt(utterance_id)
        p._enqueue(
            p._mt_queue,
            _MtJob(utterance_id, result.text, src, manage_typing=True),
        )
    else:
        # Translation disabled: original phrase goes straight to chatbox.
        safe_submit(p, result.text, [], utterance_id)
        p._resolve_typing(utterance_id)

    p._mark_finalized(utterance_id)


def resolve_source_language(p: "Pipeline", detected_whisper: str) -> "Language":
    src_cfg = p._config.stt.source_language
    if src_cfg != "auto":
        return languages.get(src_cfg)
    # "auto": map the detected Whisper code to the first matching Language.
    for lang in languages.LANGUAGES.values():
        if lang.whisper == detected_whisper:
            return lang
    return languages.get("English")


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


# -- chatbox submit helpers ---------------------------------------------------


def submit_to_chatbox(
    p: "Pipeline", original: str, translations: list[tuple[str, str]], utterance_id: int
) -> None:
    if not p._config.osc.send_to_vrchat:
        return
    p._chatbox.submit_message(original, translations, utterance_id)


def safe_submit(
    p: "Pipeline", original: str, translations: list[tuple[str, str]], utterance_id: int
) -> None:
    """`submit_to_chatbox` that can't take its caller down: a failure
    publishes ``AppError("CHATBOX_SEND_FAILED")`` instead of propagating
    (typing still resolves, GUI-thread `submit_typed` never sees it)."""
    try:
        submit_to_chatbox(p, original, translations, utterance_id)
    except Exception as exc:  # noqa: BLE001 -- a send failure is not fatal
        logger.exception("chatbox submit failed")
        p._bus.publish(AppError("CHATBOX_SEND_FAILED", str(exc)))


def safe_submit_partial(p: "Pipeline", text: str) -> None:
    """`ChatboxSender.submit_partial` guarded the same way `safe_submit`
    guards `submit_to_chatbox`: a send failure publishes ``AppError`` instead
    of taking down the STT worker."""
    try:
        p._chatbox.submit_partial(text)
    except Exception as exc:  # noqa: BLE001 -- a send failure is not fatal
        logger.exception("chatbox partial submit failed")
        p._bus.publish(AppError("CHATBOX_SEND_FAILED", str(exc)))


# -- typed text ---------------------------------------------------------------


def submit_typed(p: "Pipeline", text: str) -> bool:
    """Send typed text straight through translation to the chatbox, bypassing
    STT and mute/captioning gating (utterance id 0). Returns False
    (PIPELINE_NOT_RUNNING) when not started, keeping the text uncaptured."""
    if not text or not text.strip():
        return False
    if not p._started:
        p._bus.publish(
            AppError(
                "PIPELINE_NOT_RUNNING",
                "Engines are still loading. Try again in a moment",
            )
        )
        return False
    src_cfg = p._config.stt.source_language
    src = languages.get("English") if src_cfg == "auto" else languages.get(src_cfg)
    p._bus.publish(
        PhraseRecognized(
            utterance_id=0,
            text=text,
            language=src.whisper,
            avg_logprob=0.0,
            no_speech_prob=0.0,
        )
    )
    if p._mt is not None and p._config.translate.enabled:
        p._enqueue(p._mt_queue, _MtJob(0, text, src, manage_typing=False))
    else:
        # Runs on the caller's (GUI) thread: never propagate a chatbox
        # failure back into it.
        safe_submit(p, text, [], 0)
    return True
