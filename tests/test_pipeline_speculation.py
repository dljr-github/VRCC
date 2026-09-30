"""Speculation is optional work: queued finals and busy engines take precedence."""

from __future__ import annotations

import threading

from vrcc.audio.segmenter import SegDiscard, SegFinal, SegSpeculative
from vrcc.core import pipeline_jobs
from vrcc.core.events import PhraseRecognized

from .conftest import collect, make_pipeline, running, sample, wait_until


def test_speculative_skips_a_nonfull_queue_without_losing_either_final():
    env = make_pipeline(mt=None)  # drive jobs synchronously, without workers
    p = env.pipeline
    recognized = collect(env.bus, PhraseRecognized)
    first, second = sample(), sample()
    p._on_seg_event(SegFinal(1, first))
    assert not p._stt_queue.full()

    p._on_seg_event(SegSpeculative(2, second))

    assert p._stt_queue.qsize() == 1
    assert p._skipped_speculatives == 1
    assert 2 not in p._spec._pending
    assert env.chatbox.typing == []
    p._on_seg_event(SegFinal(2, second))
    while not p._stt_queue.empty():
        pipeline_jobs.process_stt_job(p, p._stt_queue.get_nowait(), p._stop_flag)
    assert [e.utterance_id for e in recognized] == [1, 2]
    assert [uid for _text, uid in env.chatbox.submits] == [1, 2]
    assert env.stt.calls == 2


def test_speculative_is_not_admitted_while_shared_engine_is_busy():
    env = make_pipeline(mt=None)
    p = env.pipeline
    with p.stt_slot.borrow():
        p._on_seg_event(SegSpeculative(1, sample()))
        assert p._stt_queue.empty()
    assert p._skipped_speculatives == 1
    assert p._spec._pending == {}
    assert env.chatbox.typing == []


def test_busy_engine_after_admission_skips_speculation_but_final_runs_once():
    env = make_pipeline(mt=None)
    p = env.pipeline
    recognized = collect(env.bus, PhraseRecognized)
    samples = sample()
    p._on_seg_event(SegSpeculative(1, samples))
    job = p._stt_queue.get_nowait()
    finished = threading.Event()

    def process():
        try:
            pipeline_jobs.process_stt_job(p, job, p._stop_flag)
        finally:
            finished.set()

    worker = threading.Thread(target=process, daemon=True)
    try:
        with p.stt_slot.borrow():  # heard stream won the slot after admission
            worker.start()
            assert finished.wait(1.0), "speculation waited for the occupied engine"
            assert env.stt.calls == 0
    finally:
        worker.join(2.0)
    assert not worker.is_alive()
    assert p._spec._pending == {}
    assert p._spec._cache == {}
    assert p._stats.snapshot()["skipped_speculative_calls"] == 1
    assert recognized == []

    p._on_seg_event(SegFinal(1, samples))
    pipeline_jobs.process_stt_job(p, p._stt_queue.get_nowait(), p._stop_flag)
    assert env.stt.calls == 1
    assert [e.utterance_id for e in recognized] == [1]
    assert [uid for _text, uid in env.chatbox.submits] == [1]
    assert p._typing._in_flight == set()
    assert env.chatbox.typing[-1] is False


def test_idle_speculative_result_is_still_reused_for_its_final():
    env = make_pipeline(mt=None)
    p = env.pipeline
    samples = sample()
    p._on_seg_event(SegSpeculative(1, samples))
    pipeline_jobs.process_stt_job(p, p._stt_queue.get_nowait(), p._stop_flag)
    p._on_seg_event(SegFinal(1, samples))
    pipeline_jobs.process_stt_job(p, p._stt_queue.get_nowait(), p._stop_flag)
    assert env.stt.calls == 1
    assert p._stats.snapshot()["reuse_count"] == 1
    assert len(env.chatbox.submits) == 1


def test_discard_after_busy_admission_skip_leaves_no_pending_state():
    env = make_pipeline()
    p = env.pipeline
    with p.stt_slot.borrow():
        p._on_seg_event(SegSpeculative(1, sample()))
    p._on_seg_event(SegDiscard(1))
    assert p._stt_queue.empty()
    assert p._spec._pending == {}
    assert p._spec._stale == set()
    assert p._spec._cache == {}
    assert p._typing._in_flight == set()


def test_stopping_after_busy_speculative_skip_does_not_wait_for_engine():
    env = make_pipeline()
    p = env.pipeline
    with running(p):
        with p.stt_slot.borrow():
            p._on_seg_event(SegSpeculative(1, sample()))
            assert p._stt_queue.empty()
            worker = p._stt_thread
            p.stop()
            assert not worker.is_alive()
    assert env.stt.calls == 0
    assert env.chatbox.submits == []


def test_inflight_foreground_call_skips_speculation_and_keeps_later_final():
    env = make_pipeline(mt=None)
    p = env.pipeline
    env.stt.gate.clear()
    samples = sample()
    with running(p):
        try:
            p._on_seg_event(SegFinal(1, sample()))
            assert env.stt.entered.wait(2.0)
            p._on_seg_event(SegSpeculative(2, samples))
            assert p._stt_queue.empty()
            p._on_seg_event(SegFinal(2, samples))
        finally:
            env.stt.gate.set()
        assert wait_until(lambda: len(env.chatbox.submits) == 2)
    assert [uid for _text, uid in env.chatbox.submits] == [1, 2]
    assert env.stt.calls == 2


def test_skipped_call_stats_reset_and_fold_into_session_summary(caplog):
    import logging

    from vrcc.core.pipeline_stats import log_summary

    env = make_pipeline()
    p = env.pipeline
    p._stats.record_skipped_speculative()
    p._skipped_speculatives = 2
    with caplog.at_level(logging.INFO, logger="vrcc.core.pipeline"):
        log_summary(p)
    assert "Skipped 3 speculatives" in caplog.text
    assert "0 calls (0 speculative, 0 final)" in caplog.text
    p._stats.reset()
    assert p._stats.snapshot()["skipped_speculative_calls"] == 0
