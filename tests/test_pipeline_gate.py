"""Tests for :mod:`vrcc.core.pipeline_jobs.forward_final` -- the send-time
re-check of `_should_caption()`. `handle_final` only gates at enqueue time;
a result that finishes transcribing AFTER the user muted or turned
captioning off must still be caught before it reaches the chatbox.
Split out of `test_pipeline.py` to keep both files under the line cap.
"""

from __future__ import annotations

import threading

from vrcc.audio.segmenter import SegDiscard
from vrcc.core import pipeline_jobs
from vrcc.core.events import PhraseRecognized
from vrcc.core.pipeline_jobs import _SttJob

from .conftest import FakeMute, FakeStt, collect, make_pipeline, make_result, sample


def _final_job(uid: int, s) -> _SttJob:
    return _SttJob(utterance_id=uid, samples=s, speculative=False, samples_id=id(s))


def _spec_job(uid: int, s) -> _SttJob:
    return _SttJob(utterance_id=uid, samples=s, speculative=True, samples_id=id(s))


class _OrderedCommitRecorder:
    """Stand-in segmenter recording request_commit into a shared ordered log
    alongside the chatbox's submits, so a test can assert the commit lands
    BEFORE the send."""

    def __init__(self, log: list) -> None:
        self.commits: list = []
        self._log = log

    def request_commit(self, utterance_id: int) -> None:
        self.commits.append(utterance_id)
        self._log.append(("commit", utterance_id))


def test_request_commit_precedes_the_early_send():
    # The early inject must cut the utterance BEFORE forward_final's blocking
    # enqueue: request_commit only flags the segmenter (read on its next
    # frame), so cutting first stops it appending the next sentence's onset to
    # the still-open buffer during that block, which would otherwise drop the
    # onset. The emitted-early guard is still set after, so the dedupe holds.
    env = make_pipeline(mt=None, stt=FakeStt(result=make_result(text="Hello there now.")))
    seg = _OrderedCommitRecorder(env.chatbox.log)
    env.pipeline._segmenter = seg
    s = sample()
    env.pipeline._begin_typing(1)
    env.pipeline._spec.note_speculative(1, id(s))
    pipeline_jobs.process_stt_job(env.pipeline, _spec_job(1, s), threading.Event())

    ordered = [e[0] for e in env.chatbox.log if e[0] in ("commit", "submit")]
    assert ordered == ["commit", "submit"]  # commit cuts before the send
    assert seg.commits == [1]
    assert env.chatbox.submits == [("Hello there now.", 1)]
    assert 1 in env.pipeline._spec._emitted_early  # dedupe guard set after send
    # A natural final racing the commit must still not double-send.
    pipeline_jobs.process_stt_job(env.pipeline, _final_job(1, s), threading.Event())
    assert env.chatbox.submits == [("Hello there now.", 1)]
    assert env.stt.calls == 1  # the final neither re-sent nor re-transcribed


def test_should_inject_sentence_blocks_on_high_no_speech_prob():
    # Measured hallucination case: a complete-looking sentence at
    # no_speech_prob 0.562 must not commit early; the whole-utterance final
    # is left to handle it.
    env = make_pipeline()
    result = make_result(text="Hello there now.", no_speech_prob=0.562)
    assert pipeline_jobs._should_inject_sentence(env.pipeline, result) is False


def test_should_inject_sentence_blocks_on_low_avg_logprob():
    env = make_pipeline()
    result = make_result(text="Hello there now.", avg_logprob=-0.5)
    assert pipeline_jobs._should_inject_sentence(env.pipeline, result) is False


def test_should_inject_sentence_allows_confident_result_just_inside_gate():
    env = make_pipeline()
    result = make_result(text="Hello there now.", no_speech_prob=0.2, avg_logprob=-0.3)
    assert pipeline_jobs._should_inject_sentence(env.pipeline, result) is True


def test_forward_final_valid_src_publishes_enqueues_mt_and_finalizes():
    # Pins forward_final's observable behavior across the _send_caption
    # extraction: a normal final (valid src, MT enabled) still publishes
    # PhraseRecognized, enqueues exactly one _MtJob owning typing, and still
    # finalizes (bumps last_finalized) even though the helper it now calls
    # does not itself finalize.
    env = make_pipeline()
    recognized = collect(env.bus, PhraseRecognized)
    result = make_result()
    pipeline_jobs.forward_final(env.pipeline, 1, result)
    assert [e.text for e in recognized] == [result.text]
    assert env.pipeline._mt_queue.qsize() == 1
    job = env.pipeline._mt_queue.get_nowait()
    assert isinstance(job, pipeline_jobs._MtJob)
    assert (job.utterance_id, job.text, job.manage_typing) == (1, result.text, True)
    assert env.pipeline._typing.is_owned_by_mt(1)
    assert env.pipeline._spec._last_finalized >= 1


def test_forward_final_regated_by_captioning_off_does_not_send():
    env = make_pipeline(mt=None)
    recognized = collect(env.bus, PhraseRecognized)
    env.pipeline._begin_typing(1)
    env.pipeline.set_captioning(False)  # gate closes between enqueue and send
    pipeline_jobs.forward_final(env.pipeline, 1, make_result())
    assert recognized == []
    assert env.chatbox.submits == []
    assert env.chatbox.typing[-1] is False  # typing indicator still resolves
    assert env.pipeline._spec._last_finalized >= 1  # still bounds the caches


def test_handle_discard_drops_cache_and_resolves_typing_regardless_of_terminal():
    # drop_discarded and _resolve_typing must run for every discard.
    env = make_pipeline()
    env.pipeline._begin_typing(1)
    env.pipeline._spec.note_speculative(1, 99)
    pipeline_jobs.handle_discard(env.pipeline, SegDiscard(utterance_id=1, terminal=False))
    assert env.pipeline._spec._pending == {}
    assert env.pipeline._typing._in_flight == set()


def test_forward_final_regated_by_mute_does_not_send():
    mute = FakeMute(caption=True)
    env = make_pipeline(mt=None, mute=mute)
    recognized = collect(env.bus, PhraseRecognized)
    env.pipeline._begin_typing(1)
    mute.caption = False  # user muted between enqueue and send
    pipeline_jobs.forward_final(env.pipeline, 1, make_result())
    assert recognized == []
    assert env.chatbox.submits == []
    assert env.chatbox.typing[-1] is False


def test_forward_final_quality_gated_none_resolves_typing():
    # A quality-gated (None) result sends nothing downstream; the finalize
    # step must still resolve typing.
    env = make_pipeline(mt=None)
    env.pipeline._begin_typing(1)
    pipeline_jobs.forward_final(env.pipeline, 1, None)
    assert env.chatbox.typing[-1] is False


def test_final_no_engine_drop_resolves_typing():
    # The engine was swapped out while the final was in flight, so it drops
    # before forward_final. Typing must still resolve, not left stuck.
    env = make_pipeline(mt=None)
    env.pipeline._begin_typing(1)
    env.pipeline.set_stt(None)  # transcribe now returns _NO_ENGINE
    pipeline_jobs.process_stt_job(env.pipeline, _final_job(1, sample()), threading.Event())
    assert env.chatbox.typing[-1] is False


def test_final_stop_set_drop_resolves_typing():
    # The run stopped mid-transcribe, so the final drops before forward_final.
    # Ids are monotonic across runs, so resolving here is safe even if a
    # restart already began.
    env = make_pipeline(mt=None)
    env.pipeline._begin_typing(1)
    stop = threading.Event()
    stop.set()
    pipeline_jobs.process_stt_job(env.pipeline, _final_job(1, sample()), stop)
    assert env.chatbox.typing[-1] is False
