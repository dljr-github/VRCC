"""Tests for :mod:`vrcc.core.pipeline_jobs.forward_final` -- the send-time
re-check of `_should_caption()`. `handle_final` only gates at enqueue time;
a result that finishes transcribing AFTER the user muted or turned
captioning off must still be caught before it reaches the chatbox.
Split out of `test_pipeline.py` to keep both files under the line cap.
"""

from __future__ import annotations

import threading

from vrcc.audio.segmenter import SegDiscard, SegSpeculative
from vrcc.core import pipeline_jobs
from vrcc.core.events import PhraseRecognized
from vrcc.core.pipeline_jobs import _SttJob

from .conftest import (
    FakeMute,
    FakeStt,
    collect,
    fill_stt_queue,
    make_pipeline,
    make_result,
    sample,
)


def _final_job(uid: int, s) -> _SttJob:
    return _SttJob(utterance_id=uid, samples=s, speculative=False, samples_id=id(s))


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
    assert 1 in env.pipeline._typing._owned_by_mt
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


def test_handle_discard_drops_cache_and_resolves_typing():
    # drop_discarded and _resolve_typing must run for every discard.
    env = make_pipeline()
    env.pipeline._begin_typing(1)
    env.pipeline._spec.note_speculative(1, 99)
    pipeline_jobs.handle_discard(env.pipeline, SegDiscard(utterance_id=1))
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


# -- speculative discarded while queued (never reaches the engine) ----------


def test_speculative_discarded_while_queued_never_reaches_engine():
    # SegDiscard landing before the job reaches the engine must skip the
    # transcribe call entirely, not just throw the result away afterward
    # (store_result already covers a discard that lands mid-transcribe).
    env = make_pipeline(stt=FakeStt())
    s = sample()
    env.pipeline._spec.note_speculative(1, id(s))
    env.pipeline._spec.drop_discarded(1)  # marks (1, id(s)) stale
    job = _SttJob(utterance_id=1, samples=s, speculative=True, samples_id=id(s))
    pipeline_jobs.process_stt_job(env.pipeline, job, threading.Event())
    assert env.stt.calls == 0


def test_typing_never_turns_on_for_a_skipped_speculative_later_discarded():
    # handle_speculative notes the speculative before the put and takes the
    # note back on a full queue, never reaching _begin_typing, so a later
    # discard for that utterance must not leave typing stuck on (it was
    # never turned on to begin with).
    env = make_pipeline()
    pipeline = env.pipeline
    fill_stt_queue(pipeline)
    pipeline._on_seg_event(SegSpeculative(utterance_id=7, samples=sample()))
    assert pipeline._skipped_speculatives == 1
    assert env.chatbox.typing == []  # never turned on

    pipeline._on_seg_event(SegDiscard(utterance_id=7))
    assert 7 not in pipeline._typing._in_flight
    assert True not in env.chatbox.typing  # still never turned on


def test_speculative_is_noted_before_the_worker_can_see_its_job():
    # A discarded speculative leaves its key stale, the next snapshot for the
    # same utterance can land at the same address, and the worker may dequeue
    # the job the moment it is put. The un-stale in note_speculative has to
    # come first, or consume_stale eats the new job.
    env = make_pipeline(stt=FakeStt())
    p = env.pipeline
    stop = threading.Event()
    s = sample()
    p._on_seg_event(SegSpeculative(utterance_id=1, samples=s))
    pipeline_jobs.process_stt_job(p, p._stt_queue.get_nowait(), stop)
    p._on_seg_event(SegDiscard(utterance_id=1))
    assert (1, id(s)) in p._spec._stale

    real_put = p._stt_queue.put_nowait

    def worker_wins_the_race(job):
        real_put(job)
        pipeline_jobs.process_stt_job(p, p._stt_queue.get_nowait(), stop)

    p._stt_queue.put_nowait = worker_wins_the_race
    p._on_seg_event(SegSpeculative(utterance_id=1, samples=s))  # the same key

    assert env.stt.calls == 2
    assert (1, id(s)) in p._spec._cache
