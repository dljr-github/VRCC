"""Sweet-spot gate for live sentence captions: a continuous multi-sentence
utterance must stream out as more than one caption while it is still being
spoken (latency), and the streamed captions together must say exactly the
same words as the whole-utterance transcription, with no drop and no
double-send (accuracy, the no-regression guarantee). A single word must still
produce exactly one caption. Runs the real threaded pipeline against the real
cached Whisper model (see ``_harness.find_cached_whisper``); skips otherwise.
"""

from __future__ import annotations

import pytest

from ._harness import (
    find_cached_whisper,
    finals_text,
    load_fixture,
    norm_words,
    pipeline_phrases,
)

_WHISPER = find_cached_whisper()

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(_WHISPER is None, reason="no cached whisper model"),
]


@pytest.mark.parametrize("clip", ["multi/m1.wav", "multi/m2.wav", "multi/m3.wav"])
def test_multi_sentence_streams_and_matches_reference(stt, clip):
    audio = load_fixture(clip)
    streamed = pipeline_phrases(audio, stt, sentence_inject=True)
    ref = finals_text(audio, stt)

    assert len(streamed) >= 2, (
        f"{clip}: expected the utterance to stream out as multiple captions "
        f"during speech, got a single caption at the stop: {streamed!r}"
    )
    assert norm_words(" ".join(streamed)) == norm_words(" ".join(ref)), (
        f"{clip}: streamed captions do not say the same words as the "
        f"whole-utterance reference (a drop or a double-send)\n"
        f"streamed: {streamed!r}\nreference: {ref!r}"
    )


def test_single_word_emits_one_caption(stt):
    audio = load_fixture("no.wav")
    streamed = pipeline_phrases(audio, stt, sentence_inject=True)
    assert len(streamed) == 1, (
        f"expected exactly one caption for a single word, got {streamed!r}"
    )
    assert norm_words(streamed[0])
