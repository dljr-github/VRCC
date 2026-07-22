"""Regression guard for the F4 sentence-injection fix: with sentence_inject
on, three sentences separated by short gaps must not get chopped into a
speculative fragment mid-clause (``sentence_min_words=3`` stops a comma pause
from reading as a complete sentence on its own). Runs the real threaded
``Pipeline`` (not just the segmenter), since the bug this guards against was
a timing interaction between ``sentence_inject`` and
``speculative_silence_ms`` that only showed up under the threaded pacing.
sentence_inject is off by default, so this test enables it explicitly to
keep covering that interaction.
"""

from __future__ import annotations

import numpy as np
import pytest

from vrcc.core.config import VadConfig

from ._harness import (
    SAMPLE_RATE,
    find_cached_whisper,
    load_fixture,
    norm_words,
    pipeline_phrases,
    wer,
)

_WHISPER = find_cached_whisper()

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(_WHISPER is None, reason="no cached whisper model"),
]


def _gap(seconds: float) -> np.ndarray:
    return np.zeros(int(SAMPLE_RATE * seconds), dtype=np.float32)


def test_f4_does_not_oversegment_with_sentence_inject(stt):
    sent0 = load_fixture("sent0.wav")
    sent1 = load_fixture("sent1.wav")
    sent2 = load_fixture("sent2.wav")
    gap = _gap(0.45)
    audio = np.concatenate([sent0, gap, sent1, gap, sent2])

    reference = stt.transcribe(audio)
    assert reference is not None and reference.text.strip(), "reference transcription failed"

    default_vad = VadConfig()
    phrases = pipeline_phrases(
        audio,
        stt,
        drain_s=3.5,
        sentence_inject=True,
        sentence_min_words=default_vad.sentence_min_words,
        speculative_silence_ms=default_vad.speculative_silence_ms,
    )
    assert phrases, "pipeline produced no recognized phrase"

    joined = " ".join(phrases).strip()
    error = wer(reference.text, joined)
    assert error <= 0.15, (
        f"WER {error:.2%} too high: {joined!r} vs reference {reference.text!r}"
    )

    assert not any(norm_words(p) == ["hello", "there"] for p in phrases), (
        f"first clause was split into a bare 2-word fragment: {phrases!r}"
    )

    hello_phrase = next((p for p in phrases if "hello" in norm_words(p)), None)
    assert hello_phrase is not None, f"no phrase contains 'hello': {phrases!r}"
    assert "today" in norm_words(hello_phrase), (
        f"first clause was chopped mid-sentence: {hello_phrase!r}"
    )
