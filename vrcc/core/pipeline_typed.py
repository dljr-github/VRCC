"""Typed-text send: routes GUI-typed text through translation to the chatbox,
bypassing STT and mute/captioning gating. Import direction: this module
imports pipeline_jobs and pipeline_send (never the reverse at runtime).
"""

from __future__ import annotations

import queue
from typing import TYPE_CHECKING

from vrcc.core import languages
from vrcc.core.events import AppError, PhraseRecognized
from vrcc.core.pipeline_jobs import _MtJob
from vrcc.core.pipeline_send import safe_submit

if TYPE_CHECKING:
    from vrcc.core.pipeline import Pipeline


def submit_typed(p: "Pipeline", text: str) -> bool:
    """Send typed text straight through translation to the chatbox, bypassing
    STT and mute/captioning gating. Each call gets its own unique negative
    utterance id (see ``Pipeline._next_message_id``): a shared id would let a
    second Send's recognized() remap CaptionModel's row lookup before the
    first's async translate/send completes, stamping the wrong row. Returns
    False (PIPELINE_NOT_RUNNING) when not started, and False (PIPELINE_BUSY)
    when the translation queue is full, keeping the text uncaptured either
    way."""
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
    utterance_id = p._next_message_id()
    src_cfg = p._config.stt.source_language
    src = languages.get("English") if src_cfg == "auto" else languages.get(src_cfg)
    translating = p._mt is not None and p._config.translate.enabled

    if translating:
        # This call runs on the GUI thread (the Send button), unlike the
        # STT-worker-driven enqueues above: the blocking backpressure of
        # _enqueue would freeze input/repaints/Stop behind a slow model. A
        # full queue refuses immediately instead of waiting for a slot, and
        # skips the recognized() publish below so no phantom row appears.
        try:
            p._mt_queue.put_nowait(_MtJob(utterance_id, text, src, manage_typing=False))
        except queue.Full:
            p._bus.publish(
                AppError(
                    "PIPELINE_BUSY",
                    "Still catching up, try again in a moment",
                )
            )
            return False

    p._bus.publish(
        PhraseRecognized(
            utterance_id=utterance_id,
            text=text,
            language=src.whisper,
            avg_logprob=0.0,
            no_speech_prob=0.0,
        )
    )
    if not translating:
        # Runs on the caller's (GUI) thread: never propagate a chatbox
        # failure back into it.
        safe_submit(p, text, [], utterance_id)
    return True
