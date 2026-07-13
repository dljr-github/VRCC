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
from vrcc.core.events import AppError, PhraseRecognized, PhraseTranslated
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


# -- STT job processing ------------------------------------------------------


def process_stt_job(p: "Pipeline", job: _SttJob, stop: "threading.Event") -> None:
    key = (job.utterance_id, job.samples_id)

    if job.speculative:
        result = p._transcribe(job.samples)
        if result is _NO_ENGINE:
            return  # engine swapped out mid-flight: drop the job
        if stop.is_set():
            # Stopped (maybe restarted) mid-transcribe: this result belongs
            # to an abandoned run, must not touch a new run's shared state.
            return
        p._spec.store_result(key, result)
        return

    # Final: reuse the speculative's cached result on identical samples,
    # else transcribe fresh. The spec lock is released before _transcribe
    # (never nested inside _stt_lock), preserving lock ordering.
    result = p._spec.pop_result(key)
    if result is _MISSING:
        result = p._transcribe(job.samples)
        if result is _NO_ENGINE:
            return  # engine swapped out mid-flight: drop the job
    if stop.is_set():
        return  # abandoned mid-call: discard, publish nothing
    forward_final(p, job.utterance_id, result)


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
