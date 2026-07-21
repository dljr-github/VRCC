"""Tests for :mod:`vrcc.core.pipeline_jobs.forward_final` deduping the final
against the sentences the partials already committed (``Pipeline._commits``),
plus the commit-record clearing on finalize and discard. Split out of
test_pipeline.py to keep both files under the line cap.
"""

from __future__ import annotations

from vrcc.audio.segmenter import SegDiscard
from vrcc.core import pipeline_jobs
from vrcc.core.events import PhraseRecognized

from .conftest import collect, make_pipeline, make_result


def test_final_sends_only_the_uncommitted_tail():
    # Partials already streamed the first two sentences of a continuous
    # utterance; the final must send only the sentence they could not confirm.
    # The tail is one of the leftover sentences, so it goes out under a fresh
    # negative id (the per-sentence branch), not the utterance id.
    env = make_pipeline(mt=None)
    recognized = collect(env.bus, PhraseRecognized)
    env.pipeline._commits.uncommitted(1, ["This is one.", "That is two."])
    pipeline_jobs.forward_final(
        env.pipeline,
        1,
        make_result(text="This is one. That is two. And this three."),
    )
    assert [e.text for e in recognized] == ["And this three."]
    assert len(recognized) == 1
    assert recognized[0].utterance_id < 0
    assert [text for text, _ in env.chatbox.submits] == ["And this three."]
    assert env.chatbox.submits[0][1] < 0


def test_final_sends_non_adjacent_leftovers_as_separate_ordered_captions():
    # A short sentence the partials skipped ("Yes.") sits BEFORE the sentence
    # they committed, so the leftovers are non-adjacent. Each must go out as its
    # own caption in text order, never merged into "Yes. Bye.".
    env = make_pipeline(mt=None)
    recognized = collect(env.bus, PhraseRecognized)
    env.pipeline._commits.uncommitted(1, ["I went to the store today."])
    pipeline_jobs.forward_final(
        env.pipeline,
        1,
        make_result(text="Yes. I went to the store today. Bye."),
    )
    assert [e.text for e in recognized] == ["Yes.", "Bye."]
    assert [text for text, _ in env.chatbox.submits] == ["Yes.", "Bye."]
    ids = [uid for _, uid in env.chatbox.submits]
    assert all(uid < 0 for uid in ids)
    assert len(set(ids)) == 2


def test_nothing_precommitted_sends_verbatim():
    # Empty _commits (the case every pre-dedup test hits): the whole result
    # text goes out once, byte-identical to the old whole-utterance final.
    env = make_pipeline(mt=None)
    recognized = collect(env.bus, PhraseRecognized)
    pipeline_jobs.forward_final(
        env.pipeline, 1, make_result(text="Hello there friend.")
    )
    assert [e.text for e in recognized] == ["Hello there friend."]
    assert env.chatbox.submits == [("Hello there friend.", 1)]


def test_all_committed_sends_nothing_but_finalizes():
    # Every sentence already streamed out: no caption, but typing resolves and
    # the utterance still finalizes so the caches stay bounded.
    env = make_pipeline(mt=None)
    recognized = collect(env.bus, PhraseRecognized)
    env.pipeline._begin_typing(1)
    env.pipeline._commits.uncommitted(1, ["Only one here."])
    pipeline_jobs.forward_final(env.pipeline, 1, make_result(text="Only one here."))
    assert recognized == []
    assert env.chatbox.submits == []
    assert env.chatbox.typing[-1] is False
    assert env.pipeline._spec.last_finalized() == 1


def test_finalize_clears_the_commit_record():
    # Finalizing an utterance drops its committed-sentence row so it cannot leak.
    env = make_pipeline(mt=None)
    env.pipeline._commits.uncommitted(1, ["This is one.", "That is two."])
    pipeline_jobs.forward_final(
        env.pipeline,
        1,
        make_result(text="This is one. That is two. And this three."),
    )
    assert env.pipeline._commits.committed_count(1) == 0


def test_discard_clears_the_commit_record():
    # A discarded utterance's partial commits must not leak either.
    env = make_pipeline(mt=None)
    env.pipeline._commits.uncommitted(2, ["x sentence here."])
    pipeline_jobs.handle_discard(env.pipeline, SegDiscard(utterance_id=2))
    assert env.pipeline._commits.committed_count(2) == 0
