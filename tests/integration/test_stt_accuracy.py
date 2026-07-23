"""End-to-end STT accuracy against the real cached Whisper model: clean
transcription word error rate and onset survival after leading silence. Needs
a whisper model already downloaded on this machine (see
``_harness.find_cached_whisper``); skips otherwise. Runs the segmenter
directly (not the threaded pipeline), so each assertion isolates one stage.
"""

from __future__ import annotations

import numpy as np
import pytest

from ._harness import SAMPLE_RATE, find_cached_whisper, finals_text, load_fixture, wer

_WHISPER = find_cached_whisper()

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(_WHISPER is None, reason="no cached whisper model"),
]


def test_clean_transcription_low_wer(stt):
    audio = load_fixture("tts_sentences.wav")
    reference = stt.transcribe(audio)
    assert reference is not None and reference.text.strip(), "reference transcription failed"

    texts = finals_text(audio, stt)
    joined = " ".join(texts).strip()
    assert joined, "no finalized utterance produced any text"

    error = wer(reference.text, joined)
    assert error <= 0.15, (
        f"WER {error:.2%} too high: {joined!r} vs reference {reference.text!r}"
    )
    lowered = joined.lower()
    for word in ("testing", "captions", "sentences"):
        assert word in lowered, f"content word {word!r} missing from {joined!r}"


def test_onset_first_word_survives(stt):
    sent = load_fixture("sent0.wav")
    lead_silence = np.zeros(int(SAMPLE_RATE * 1.2), dtype=np.float32)
    audio = np.concatenate([lead_silence, sent])

    texts = finals_text(audio, stt)
    assert texts, "no finalized utterance produced any text"
    assert texts[0].strip().lower().startswith("hello"), (
        f"onset clipped: first finalized transcript is {texts[0]!r}"
    )
