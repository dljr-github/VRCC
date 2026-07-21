"""Tests for :mod:`vrcc.core.pipeline` -- live partial transcription:
`SegPartial` dispatch, `handle_partial` coalescing (`_partial_pending`
caps in-flight partials at one), and mid-utterance sentence commits.

A stable complete sentence (seen in two consecutive partials' followed
lists, per `CommitTracker.stable_new`) is committed to the chatbox as its
own message, its own id, without finalizing the utterance or resetting the
segmenter buffer. The raw partial text itself never reaches the log or the
chatbox until a sentence in it is stable. Split out of `test_pipeline.py`
to keep both files under the line cap.
"""

from __future__ import annotations

import threading
import time

from vrcc.audio.segmenter import SegPartial
from vrcc.core import pipeline_jobs
from vrcc.core.config import AppConfig, VadConfig
from vrcc.core.events import AppError, PhraseRecognized
from vrcc.core.pipeline_jobs import _SttJob

from .conftest import FakeStt, collect, make_pipeline, make_result, running, sample


def _partial_job(uid: int, s) -> _SttJob:
    return _SttJob(
        utterance_id=uid, samples=s, speculative=False, samples_id=id(s), partial=True
    )


def _wait_until(predicate, timeout: float = 2.0, interval: float = 0.005) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return bool(predicate())


# -- 1: stability (two partials) --------------------------------------------


def test_two_stable_partials_commit_two_sentences_with_distinct_ids():
    env = make_pipeline(mt=None, stt=FakeStt(results=[
        make_result(text="This is one. That is two. tail"),
        make_result(text="This is one. That is two. tail end"),
    ]))
    recognized = collect(env.bus, PhraseRecognized)
    pipeline_jobs._process_partial_job(env.pipeline, _partial_job(1, sample()), threading.Event())
    assert recognized == []  # first sighting: not yet stable, nothing committed

    pipeline_jobs._process_partial_job(env.pipeline, _partial_job(1, sample()), threading.Event())
    assert [e.text for e in recognized] == ["This is one.", "That is two."]
    ids = [e.utterance_id for e in recognized]
    assert len(set(ids)) == 2  # distinct per sentence: no coalescing in the chatbox
    assert all(i < 0 for i in ids)  # typed/committed ids never collide with segmenter ids
    assert env.chatbox.submits == [("This is one.", ids[0]), ("That is two.", ids[1])]


# -- 2: dedup (a later partial does not re-commit) --------------------------


def test_later_partial_does_not_recommit_already_committed_sentences():
    env = make_pipeline(mt=None, stt=FakeStt(results=[
        make_result(text="This is one. That is two. tail"),
        make_result(text="This is one. That is two. tail end"),
        make_result(text="This is one. That is two. Now a third one. tail"),
        make_result(text="This is one. That is two. Now a third one. tail more"),
    ]))
    recognized = collect(env.bus, PhraseRecognized)
    stop = threading.Event()
    pipeline_jobs._process_partial_job(env.pipeline, _partial_job(1, sample()), stop)
    pipeline_jobs._process_partial_job(env.pipeline, _partial_job(1, sample()), stop)
    assert [e.text for e in recognized] == ["This is one.", "That is two."]

    # Third partial: the new sentence is only first-sighted, not yet stable.
    pipeline_jobs._process_partial_job(env.pipeline, _partial_job(1, sample()), stop)
    assert [e.text for e in recognized] == ["This is one.", "That is two."]

    # Fourth partial: the new sentence stabilizes; S1/S2 are never re-sent.
    pipeline_jobs._process_partial_job(env.pipeline, _partial_job(1, sample()), stop)
    assert [e.text for e in recognized] == ["This is one.", "That is two.", "Now a third one."]


def test_partial_does_not_finalize_or_send_on_first_sighting():
    env = make_pipeline(mt=None, stt=FakeStt(result=make_result(text="hello there")))
    recognized = collect(env.bus, PhraseRecognized)
    with running(env.pipeline):
        env.pipeline._on_seg_event(SegPartial(utterance_id=1, samples=sample()))
        assert _wait_until(lambda: env.stt.calls == 1)
        time.sleep(0.02)
    assert recognized == []
    assert env.chatbox.submits == []
    assert env.pipeline._spec._last_finalized == 0


# -- 3: muted mid-transcribe commits nothing --------------------------------


def test_captioning_off_mid_transcribe_commits_nothing():
    # Prime the two-partial stability gate with captioning on, then flip
    # captioning off while the second (now-stable) partial is blocked inside
    # transcribe. The commit must not go through even though the sentence
    # would otherwise have stabilized.
    env = make_pipeline(mt=None, stt=FakeStt(results=[
        make_result(text="This is one. tail"),
        make_result(text="This is one. tail more"),
    ]))
    recognized = collect(env.bus, PhraseRecognized)
    with running(env.pipeline):
        env.pipeline._on_seg_event(SegPartial(utterance_id=1, samples=sample()))
        assert _wait_until(lambda: env.stt.calls == 1)
        assert _wait_until(lambda: env.pipeline._partial_pending is False)
        assert recognized == []  # first sighting: not yet stable

        env.stt.gate.clear()
        env.stt.entered.clear()
        env.pipeline._on_seg_event(SegPartial(utterance_id=1, samples=sample()))
        assert env.stt.entered.wait(2.0)  # worker is inside the second transcribe
        env.pipeline.set_captioning(False)
        env.stt.gate.set()
        assert _wait_until(lambda: env.pipeline._partial_pending is False)
        time.sleep(0.02)
    assert recognized == []
    assert env.chatbox.submits == []


# -- 4: unchanged machinery (coalescing, pending flag, last_finalized) ------


def test_second_partial_while_one_pending_is_coalesced():
    env = make_pipeline(stt=FakeStt())
    env.stt.gate.clear()  # block inside transcribe so the pending flag stays set
    with running(env.pipeline):
        env.pipeline._on_seg_event(SegPartial(utterance_id=1, samples=sample()))
        assert env.stt.entered.wait(2.0)  # worker is inside transcribe
        env.pipeline._on_seg_event(SegPartial(utterance_id=1, samples=sample()))
        time.sleep(0.02)
        assert env.pipeline._stt_queue.qsize() == 0  # second was never enqueued
        assert env.stt.calls == 1
        env.stt.gate.set()
        assert _wait_until(lambda: env.pipeline._partial_pending is False)


def test_pending_flag_clears_after_processing_so_a_later_partial_can_fire():
    env = make_pipeline(stt=FakeStt(result=make_result(text="hello")))
    with running(env.pipeline):
        env.pipeline._on_seg_event(SegPartial(utterance_id=1, samples=sample()))
        assert _wait_until(lambda: env.stt.calls == 1)
        assert _wait_until(lambda: env.pipeline._partial_pending is False)
        env.pipeline._on_seg_event(SegPartial(utterance_id=1, samples=sample()))
        assert _wait_until(lambda: env.stt.calls == 2)
    assert env.stt.calls == 2


def test_partial_exception_clears_pending_flag_and_later_partial_is_not_coalesced():
    # First transcribe raises; the second succeeds -> proves the pending flag
    # was cleared on the exception path, so it does not wedge live captions
    # for the rest of the run (see _process_partial_job's try/finally).
    env = make_pipeline(mt=None, stt=FakeStt(results=[RuntimeError("stt boom"), make_result(text="hi")]))
    errors = collect(env.bus, AppError)
    recognized = collect(env.bus, PhraseRecognized)
    with running(env.pipeline):
        env.pipeline._on_seg_event(SegPartial(utterance_id=1, samples=sample()))
        assert _wait_until(lambda: any(e.code == "STT_JOB_FAILED" for e in errors))
        assert _wait_until(lambda: env.pipeline._partial_pending is False)
        env.pipeline._on_seg_event(SegPartial(utterance_id=1, samples=sample()))
        assert _wait_until(lambda: env.stt.calls == 2)
        time.sleep(0.02)
    assert env.stt.calls == 2  # second partial was NOT coalesced away
    assert recognized == []  # "hi" has no terminal punctuation: nothing to commit


def test_sentence_inject_disabled_produces_no_commit_no_pending_flag():
    cfg = AppConfig(vad=VadConfig(sentence_inject=False))
    env = make_pipeline(config=cfg, stt=FakeStt(result=make_result(text="hello")))
    recognized = collect(env.bus, PhraseRecognized)
    with running(env.pipeline):
        env.pipeline._on_seg_event(SegPartial(utterance_id=1, samples=sample()))
        time.sleep(0.02)
    assert recognized == []
    assert env.chatbox.submits == []
    assert env.stt.calls == 0
    assert env.pipeline._partial_pending is False


def test_partial_at_or_below_last_finalized_commits_nothing():
    # The utterance finalized between two partials that would otherwise make
    # a sentence stable: the guard must drop the second partial outright, not
    # just happen to see nothing new (that would pass even without the guard).
    env = make_pipeline(mt=None, stt=FakeStt(result=make_result(text="This is one. That is two. tail")))
    recognized = collect(env.bus, PhraseRecognized)
    pipeline_jobs._process_partial_job(env.pipeline, _partial_job(3, sample()), threading.Event())
    assert recognized == []  # first sighting: not yet stable, nothing to drop yet

    env.pipeline._spec.mark_finalized(3)  # utterances up to 3 are finalized
    pipeline_jobs._process_partial_job(env.pipeline, _partial_job(3, sample()), threading.Event())
    assert recognized == []
    assert env.chatbox.submits == []


def test_partial_for_newer_utterance_still_commits():
    # The guard is strictly <= last_finalized: a partial for a later utterance
    # (the current one) must still be free to commit a stable sentence.
    env = make_pipeline(mt=None, stt=FakeStt(results=[
        make_result(text="This is one. tail"),
        make_result(text="This is one. tail more"),
    ]))
    recognized = collect(env.bus, PhraseRecognized)
    env.pipeline._spec.mark_finalized(3)
    pipeline_jobs._process_partial_job(env.pipeline, _partial_job(4, sample()), threading.Event())
    assert recognized == []
    pipeline_jobs._process_partial_job(env.pipeline, _partial_job(4, sample()), threading.Event())
    assert [e.text for e in recognized] == ["This is one."]
    assert env.chatbox.submits == [("This is one.", recognized[0].utterance_id)]


# -- 5: sentence_inject=False is a hard gate ---------------------------------


def test_sentence_inject_disabled_commits_nothing():
    cfg = AppConfig(vad=VadConfig(sentence_inject=False))
    env = make_pipeline(mt=None, config=cfg, stt=FakeStt(results=[
        make_result(text="This is one. tail"),
        make_result(text="This is one. tail more"),
    ]))
    recognized = collect(env.bus, PhraseRecognized)
    stop = threading.Event()
    pipeline_jobs._process_partial_job(env.pipeline, _partial_job(1, sample()), stop)
    pipeline_jobs._process_partial_job(env.pipeline, _partial_job(1, sample()), stop)
    assert recognized == []
    assert env.chatbox.submits == []
