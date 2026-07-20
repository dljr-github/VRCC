"""Tests for :class:`vrcc.core.pipeline_state.TypingTracker`'s atomic
begin/resolve callback (concurrency fix) and its MT-ownership exemption used
by the natural-final-after-early-send dedupe branch in pipeline_jobs.

Split out from test_pipeline.py to stay under the line cap; see that file's
early-injection section for the sibling mt=None dedupe test.
"""

from __future__ import annotations

import threading
from types import SimpleNamespace

from vrcc.core import pipeline_jobs
from vrcc.core.pipeline_jobs import _SttJob
from vrcc.core.pipeline_state import TypingTracker

from .conftest import FakeStt, make_pipeline, make_result, sample


def _spec_job(uid: int, s) -> _SttJob:
    return _SttJob(utterance_id=uid, samples=s, speculative=True, samples_id=id(s))


def _final_job(uid: int, s) -> _SttJob:
    return _SttJob(utterance_id=uid, samples=s, speculative=False, samples_id=id(s))


# -- Bug 2: begin/resolve hold the tracker lock across the callback --------


def test_begin_and_resolve_invoke_callback_while_lock_held():
    # A concurrent begin() can only interleave with a resolve()'s check if
    # the lock is released before the callback runs. A non-blocking acquire
    # from inside the callback proves it is still held (plain Lock, not
    # reentrant): acquire() only succeeds if nobody else holds it.
    tracker = TypingTracker()
    seen: list[tuple[bool, bool]] = []

    def on_change(value: bool) -> None:
        acquired = tracker._lock.acquire(blocking=False)
        seen.append((value, not acquired))
        if acquired:
            tracker._lock.release()

    tracker.begin(1, on_change)
    tracker.resolve(1, on_change)

    assert seen == [(True, True), (False, True)]


def test_resolve_skips_callback_while_another_utterance_in_flight():
    tracker = TypingTracker()
    calls: list[bool] = []
    tracker.begin(1, calls.append)
    tracker.begin(2, calls.append)
    tracker.resolve(1, calls.append)  # utterance 2 still in flight: no False
    assert calls == [True, True]
    tracker.resolve(2, calls.append)
    assert calls == [True, True, False]


# -- Bug 1: dedupe branch must not resolve typing while MT owns uid --------


def test_final_after_early_send_does_not_resolve_typing_while_mt_owns():
    # Translation active (the default fixture): the natural final racing the
    # early-injected commit must not turn typing off while the MT job it
    # queued (own_by_mt, in forward_final) still owns utterance 1.
    env = make_pipeline(stt=FakeStt(result=make_result(text="Hello there now.")))
    env.pipeline._segmenter = SimpleNamespace(request_commit=lambda uid: None)
    s = sample()
    env.pipeline._begin_typing(1)
    env.pipeline._spec.note_speculative(1, id(s))
    pipeline_jobs.process_stt_job(env.pipeline, _spec_job(1, s), threading.Event())
    assert env.pipeline._typing.is_owned_by_mt(1)
    typing_before = list(env.chatbox.typing)

    pipeline_jobs.process_stt_job(env.pipeline, _final_job(1, s), threading.Event())
    assert env.chatbox.typing == typing_before  # dedupe branch: no early turn-off
    assert 1 in env.pipeline._typing._in_flight

    # The MT job's own completion still resolves it.
    job = env.pipeline._mt_queue.get_nowait()
    pipeline_jobs.process_mt_job(env.pipeline, job, threading.Event())
    assert env.chatbox.typing[-1] is False
    assert 1 not in env.pipeline._typing._in_flight
