"""One DEBUG record per transcription, for answering questions about a real session.

Grep a log for ``stt_decision``. The record carries numbers and a verdict only:
caption text is never logged, because captions are private speech.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

SAMPLE_RATE = 16000

TAG = "stt_decision"


@dataclass
class Decision:
    """Fields are None when the backend has no such signal."""

    engine: str
    audio_s: float
    verdict: str = "accepted"
    language: str | None = None  # what the model detected, before any bias
    language_prob: float | None = None
    alt_language: str | None = None
    alt_prob: float | None = None
    used_language: str | None = None  # what the kept decode ran in
    redecode_s: float | None = None  # set only when bias forced a second decode
    avg_logprob: float | None = None
    no_speech: float | None = None
    compression: float | None = None
    chars: int = 0

    @property
    def biased(self) -> bool:
        return self.redecode_s is not None


def audio_seconds(samples: np.ndarray) -> float:
    return len(samples) / SAMPLE_RATE


def top_two(probabilities) -> tuple[tuple[str, float] | None, tuple[str, float] | None]:
    """Best and runner-up of ``(code, probability)`` pairs."""
    ranked = sorted(probabilities or (), key=lambda item: item[1], reverse=True)
    first = (ranked[0][0], float(ranked[0][1])) if ranked else None
    second = (ranked[1][0], float(ranked[1][1])) if len(ranked) > 1 else None
    return first, second


def with_detection(decision: Decision, probabilities) -> Decision:
    """Fill the detected and runner-up language from ``(code, prob)`` pairs."""
    first, second = top_two(probabilities)
    if first:
        decision.language, decision.language_prob = first
    if second:
        decision.alt_language, decision.alt_prob = second
    return decision


def sensevoice_language_probs(
    logits: np.ndarray, language_tokens: dict[int, tuple[str, int]]
) -> list[tuple[str, float]]:
    """Softmax of the first output frame, read off the language-tag tokens.

    Two tags can map to one Whisper code (Cantonese and Mandarin); the larger
    probability stands for that code.
    """
    if logits.shape[0] == 0 or not language_tokens:
        return []
    frame = logits[0].astype(np.float64)
    frame = np.exp(frame - frame.max())
    frame /= frame.sum()
    best: dict[str, float] = {}
    for token, (code, _slot) in language_tokens.items():
        best[code] = max(best.get(code, 0.0), float(frame[token]))
    return list(best.items())


def _num(value: float | None, places: int) -> str:
    return "-" if value is None else f"{value:.{places}f}"


def format_decision(d: Decision) -> str:
    return (
        f"{TAG} engine={d.engine} audio_s={d.audio_s:.2f}"
        f" lang={d.language or '-'} lang_p={_num(d.language_prob, 3)}"
        f" alt={d.alt_language or '-'} alt_p={_num(d.alt_prob, 3)}"
        f" used={d.used_language or '-'} biased={int(d.biased)}"
        f" redecode_s={_num(d.redecode_s, 3)}"
        f" avg_logprob={_num(d.avg_logprob, 3)} no_speech={_num(d.no_speech, 3)}"
        f" compression={_num(d.compression, 2)}"
        f" verdict={d.verdict} chars={d.chars}"
    )


def log_decision(logger: logging.Logger, decision: Decision) -> None:
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("%s", format_decision(decision))
