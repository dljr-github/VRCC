"""Tests for :mod:`vrcc.core.pipeline` -- live partial transcription:
`SegPartial` dispatch, `handle_partial` coalescing (`_partial_pending`
caps in-flight partials at one), and the partial chatbox send. Split out
of `test_pipeline.py` to keep both files under the line cap.
"""

from __future__ import annotations

import time

from vrcc.audio.segmenter import SegPartial
from vrcc.core.config import AppConfig, VadConfig
from vrcc.core.events import AppError, PhrasePartial, PhraseRecognized

from .conftest import FakeStt, collect, make_pipeline, make_result, running, sample


def _wait_until(predicate, timeout: float = 2.0, interval: float = 0.005) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return bool(predicate())


def test_partial_publishes_phrasepartial_and_sends_partial_to_chatbox():
    env = make_pipeline(stt=FakeStt(result=make_result(text="hello")))
    partials = collect(env.bus, PhrasePartial)
    recognized = collect(env.bus, PhraseRecognized)
    with running(env.pipeline):
        env.pipeline._on_seg_event(SegPartial(utterance_id=1, samples=sample()))
        assert _wait_until(lambda: len(partials) == 1)
        time.sleep(0.02)
    assert [(e.utterance_id, e.text) for e in partials] == [(1, "hello")]
    assert recognized == []  # never treated as a final/speculative result
    assert env.chatbox.submits == []  # no ChatboxSent-publishing send
    assert env.chatbox.partials == ["hello"]  # but the partial line WAS sent
    # stop()'s best-effort typing-off is the only entry: nothing began typing.
    assert env.chatbox.typing == [False]


def test_partial_does_not_forward_final_or_resolve_typing():
    env = make_pipeline(mt=None, stt=FakeStt(result=make_result(text="hello there")))
    with running(env.pipeline):
        env.pipeline._on_seg_event(SegPartial(utterance_id=1, samples=sample()))
        assert _wait_until(lambda: env.chatbox.partials == ["hello there"])
        time.sleep(0.02)
    assert env.chatbox.submits == []
    assert env.pipeline._spec._last_finalized == 0


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
    partials = collect(env.bus, PhrasePartial)
    with running(env.pipeline):
        env.pipeline._on_seg_event(SegPartial(utterance_id=1, samples=sample()))
        assert _wait_until(lambda: len(partials) == 1)
        assert _wait_until(lambda: env.pipeline._partial_pending is False)
        env.pipeline._on_seg_event(SegPartial(utterance_id=1, samples=sample()))
        assert _wait_until(lambda: len(partials) == 2)
    assert env.stt.calls == 2


def test_partial_exception_clears_pending_flag_and_later_partial_is_not_coalesced():
    # First transcribe raises; the second succeeds -> proves the pending flag
    # was cleared on the exception path, so it does not wedge live captions
    # for the rest of the run (see _process_partial_job's try/finally).
    env = make_pipeline(stt=FakeStt(results=[RuntimeError("stt boom"), make_result(text="hi")]))
    errors = collect(env.bus, AppError)
    partials = collect(env.bus, PhrasePartial)
    with running(env.pipeline):
        env.pipeline._on_seg_event(SegPartial(utterance_id=1, samples=sample()))
        assert _wait_until(lambda: any(e.code == "STT_JOB_FAILED" for e in errors))
        assert _wait_until(lambda: env.pipeline._partial_pending is False)
        env.pipeline._on_seg_event(SegPartial(utterance_id=1, samples=sample()))
        assert _wait_until(lambda: len(partials) == 1)
    assert env.stt.calls == 2  # second partial was NOT coalesced away
    assert [(e.utterance_id, e.text) for e in partials] == [(1, "hi")]


def test_live_partials_disabled_produces_no_event_no_send_no_pending_flag():
    cfg = AppConfig(vad=VadConfig(live_partials=False))
    env = make_pipeline(config=cfg, stt=FakeStt(result=make_result(text="hello")))
    partials = collect(env.bus, PhrasePartial)
    with running(env.pipeline):
        env.pipeline._on_seg_event(SegPartial(utterance_id=1, samples=sample()))
        time.sleep(0.02)
    assert partials == []
    assert env.chatbox.partials == []
    assert env.stt.calls == 0
    assert env.pipeline._partial_pending is False
