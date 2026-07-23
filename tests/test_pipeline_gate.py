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
