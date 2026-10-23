"""Chatbox submit helpers for the pipeline's STT/MT workers.

Split out of pipeline_jobs to keep both files under the line cap; import
direction matches pipeline_jobs (never imports back from it).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from vrcc.core.events import AppError
from vrcc.osc.chatbox_format import CHATBOX_LIMIT, format_message

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


def chatbox_translations(
    p: "Pipeline",
    original: str,
    src: object,
    all_targets: list,
    targets: list,
    translations: list[tuple[str, str]],
    plain: list[tuple[str, str]],
) -> list[tuple[str, str]]:
    """The ``[(name, text), ...]`` the chatbox should carry for a phrase.

    `translations` may carry pinyin reading lines that `plain` does not.
    The reading roughly quadruples a Chinese translation, so a sentence that
    fits alone overflows with it: truncation then drops the original and cuts
    the reading, and split mode loses the line break and scatters hanzi and
    pinyin across parts. The caption log keeps the reading; the chatbox
    carries it only when the whole message still fits `CHATBOX_LIMIT`.
    """
    submitted = _with_skipped_targets(
        p, original, src, all_targets, targets, translations
    )
    if translations is not plain and (
        len(format_message(original, submitted, p._config.osc)) > CHATBOX_LIMIT
    ):
        submitted = _with_skipped_targets(p, original, src, all_targets, targets, plain)
    return submitted


def _with_skipped_targets(
    p: "Pipeline",
    original: str,
    src: object,
    all_targets: list,
    targets: list,
    translations: list[tuple[str, str]],
) -> list[tuple[str, str]]:
    """Hiding the original presumes every configured target carries a
    translation. A skipped source-matching target is served by the original
    text itself, so that text re-enters the message in the target's slot;
    without it, readers of the source language get nothing at all."""
    if len(targets) == len(all_targets) or p._config.osc.include_original:
        return translations
    submitted = list(translations)
    for i, lang in enumerate(all_targets):
        if lang == src:
            submitted.insert(i, (lang.display, original))
    return submitted
