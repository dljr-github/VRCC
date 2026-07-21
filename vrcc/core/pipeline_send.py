"""Chatbox submit helpers for the pipeline's STT/MT workers.

Split out of pipeline_jobs to keep both files under the line cap; import
direction matches pipeline_jobs (never imports back from it).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from vrcc.core.events import AppError

if TYPE_CHECKING:
    from vrcc.core.pipeline import Pipeline

# Same logger as the orchestrator: one operational stream for the pipeline.
logger = logging.getLogger("vrcc.core.pipeline")


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
