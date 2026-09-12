"""Tests for :mod:`vrcc.core.pipeline` -- the typed-submit path and the
live engine hot-swap primitives (``set_stt``/``set_mt``/``set_swapping``).
"""

from __future__ import annotations

import queue
import threading
import time

from vrcc.audio.segmenter import SegFinal
from vrcc.core.events import AppError, PhraseRecognized, PhraseTranslated

from .conftest import (
    FakeChatbox,
    FakeMt,
    FakeMute,
    FakeSegmenter,
    FakeStt,
    collect,
    make_pipeline,
    make_result,
    running,
    sample,
    wait_until,
)


class _FullQueue:
    """A ``queue.Queue`` stand-in whose ``put_nowait`` always raises
    ``Full``. ``put`` (the blocking backpressure call) records that it was
    reached and raises too, so a test can prove ``submit_typed`` never falls
    back to the blocking path on the GUI thread."""

    def __init__(self) -> None:
        self.put_calls = 0

    def put_nowait(self, item) -> None:
        raise queue.Full()

    def put(self, item, timeout=None) -> None:
        self.put_calls += 1
        raise AssertionError("submit_typed must not use the blocking enqueue")


# -- typed path -------------------------------------------------------------


def test_submit_typed_chatbox_failure_does_not_propagate_to_caller():
    chatbox = FakeChatbox(fail_submits=True)
    env = make_pipeline(mt=None, chatbox=chatbox)
    errors = collect(env.bus, AppError)
    with running(env.pipeline):
        env.pipeline.submit_typed("hello")  # must not raise into the (GUI) caller
    assert any(e.code == "CHATBOX_SEND_FAILED" for e in errors)


def test_submit_typed_bypasses_stt_and_mute_and_translates():
    env = make_pipeline(mute=FakeMute(caption=False))
    recognized = collect(env.bus, PhraseRecognized)
    translated = collect(env.bus, PhraseTranslated)
    with running(env.pipeline):
        env.pipeline.submit_typed("typed hello")
        assert wait_until(lambda: len(env.chatbox.submits) == 1)
        assert wait_until(lambda: len(translated) == 1)
    assert env.stt.calls == 0  # STT bypassed
    # A typed submission gets a unique negative id (never the segmenter's 0+
    # ids), so a first-message id must be -1 here.
    assert [(e.utterance_id, e.text) for e in recognized] == [(-1, "typed hello")]
    assert translated[0].utterance_id == -1
    text, uid = env.chatbox.submits[0]
    assert uid == -1
    assert text == "typed hello\nJapanese:typed hello"


def test_submit_typed_without_mt_sends_original():
    env = make_pipeline(mt=None)
    recognized = collect(env.bus, PhraseRecognized)
    with running(env.pipeline):
        env.pipeline.submit_typed("just this")
        assert wait_until(lambda: len(env.chatbox.submits) == 1)
    assert [(e.utterance_id, e.text) for e in recognized] == [(-1, "just this")]
    assert env.chatbox.submits[0] == ("just this", -1)


def test_submit_typed_uses_unique_ids_across_calls():
    # Regression: every typed message used to share utterance_id 0, so a
    # second Send before the first's async translate/send completed remapped
    # CaptionModel's row lookup and stamped the wrong row. Each submission now
    # gets its own, never-repeating id, and it never collides with the
    # segmenter's non-negative ids.
    env = make_pipeline(mt=None)
    recognized = collect(env.bus, PhraseRecognized)
    with running(env.pipeline):
        env.pipeline.submit_typed("first")
        env.pipeline.submit_typed("second")
        assert wait_until(lambda: len(env.chatbox.submits) == 2)
    ids = [e.utterance_id for e in recognized]
    assert len(ids) == 2
    assert len(set(ids)) == 2  # never repeats
    assert all(i < 0 for i in ids)  # never collides with a segmenter id


def test_submit_typed_before_start_publishes_apperror_and_nothing_else():
    # Before start() the MT queue has no consumer: enqueueing would orphan
    # the job (start() swaps in a fresh queue) and a full queue would spin
    # the GUI thread forever. submit_typed must refuse instead.
    env = make_pipeline()
    errors = collect(env.bus, AppError)
    recognized = collect(env.bus, PhraseRecognized)
    accepted = env.pipeline.submit_typed("typed while loading")
    assert accepted is False  # caller keeps the user's text on refusal
    assert [e.code for e in errors] == ["PIPELINE_NOT_RUNNING"]
    assert recognized == []  # no phantom caption-log entry
    assert env.chatbox.submits == []
    assert env.pipeline._mt_queue.empty()  # nothing enqueued


def test_submit_typed_after_stop_publishes_apperror():
    env = make_pipeline()
    env.pipeline.start()
    env.pipeline.stop()
    errors = collect(env.bus, AppError)
    env.pipeline.submit_typed("typed after stop")
    assert [e.code for e in errors] == ["PIPELINE_NOT_RUNNING"]
    assert env.chatbox.submits == []


def test_submit_typed_empty_text_before_start_publishes_nothing():
    env = make_pipeline()
    errors = collect(env.bus, AppError)
    env.pipeline.submit_typed("   ")
    assert errors == []


def test_submit_typed_works_after_start():
    # The gate must not affect the normal started path.
    env = make_pipeline()
    errors = collect(env.bus, AppError)
    with running(env.pipeline):
        assert env.pipeline.submit_typed("after start") is True
        assert wait_until(lambda: len(env.chatbox.submits) == 1)
    assert not any(e.code == "PIPELINE_NOT_RUNNING" for e in errors)


def test_submit_typed_full_mt_queue_refuses_without_blocking():
    # Regression: submit_typed used to route through the blocking backpressure
    # helper (_enqueue), which is fine on a worker thread but freezes the GUI
    # thread (input/repaints/Stop) when a slow model leaves the MT queue full.
    # The Send button must refuse instantly instead of waiting for a slot.
    env = make_pipeline(mt=FakeMt())
    errors = collect(env.bus, AppError)
    recognized = collect(env.bus, PhraseRecognized)
    fake = _FullQueue()
    with running(env.pipeline):
        real_queue = env.pipeline._mt_queue  # the object the MT worker thread
        # actually reads (passed by reference as a start() thread arg);
        # reassigning the attribute only affects what a NEW submit_typed call
        # sees, so it must be swapped back before stop() tears the worker down
        # even if the assertion below fails.
        env.pipeline._mt_queue = fake
        try:
            accepted = env.pipeline.submit_typed("busy text")
        finally:
            env.pipeline._mt_queue = real_queue
    assert accepted is False  # caller keeps the user's text on refusal
    assert [e.code for e in errors] == ["PIPELINE_BUSY"]
    assert recognized == []  # no phantom caption-log entry
    assert env.chatbox.submits == []
    assert fake.put_calls == 0  # never fell back to the blocking put()


def test_submit_typed_enqueues_normally_when_queue_has_room():
    # Regression companion: the non-blocking put_nowait path must still
    # accept a normal submission and report success.
    env = make_pipeline(mt=FakeMt())
    errors = collect(env.bus, AppError)
    with running(env.pipeline):
        assert env.pipeline.submit_typed("room to spare") is True
        assert wait_until(lambda: len(env.chatbox.submits) == 1)
    assert not any(e.code == "PIPELINE_BUSY" for e in errors)


# -- hotswap primitives (Task 2): live engine swap + swapping gate ----------


def test_set_stt_swaps_engine_used_by_next_job():
    # A live set_stt() must be picked up by the NEXT job the worker runs.
    a = FakeStt(result=make_result(text="AAA"))
    b = FakeStt(result=make_result(text="BBB"))
    env = make_pipeline(stt=a)
    with running(env.pipeline):
        env.pipeline._on_seg_event(SegFinal(utterance_id=1, samples=sample()))
        assert wait_until(lambda: env.pipeline._spec._last_finalized >= 1)
        assert a.calls == 1  # first utterance used the original engine

        env.pipeline.set_stt(b)
        env.pipeline._on_seg_event(SegFinal(utterance_id=2, samples=sample()))
        assert wait_until(lambda: env.pipeline._spec._last_finalized >= 2)
    assert b.calls == 1, "second utterance must use the swapped-in engine"
    assert a.calls == 1, "original engine must not be called again after swap"


def test_swapping_gate_blocks_new_stt_jobs():
    # set_swapping(True) pauses new-caption creation (like a paused capture);
    # set_swapping(False) resumes it.
    stt = FakeStt(result=make_result(text="X"))
    env = make_pipeline(stt=stt)
    with running(env.pipeline):
        env.pipeline.set_swapping(True)
        # _handle_final's gated path runs synchronously on this thread: no job
        # is ever enqueued, so the worker never transcribes.
        env.pipeline._on_seg_event(SegFinal(utterance_id=1, samples=sample()))
        time.sleep(0.05)
        assert stt.calls == 0, "no transcription while swapping"

        env.pipeline.set_swapping(False)
        env.pipeline._on_seg_event(SegFinal(utterance_id=2, samples=sample()))
        assert wait_until(lambda: stt.calls == 1), "captioning resumes after the swap"


def test_detach_stt_returns_engine_and_none_result_drops_job():
    # detach_stt() removes+returns the engine; with none installed a final job
    # is dropped cleanly (no crash, no submit, no STT_JOB_FAILED error).
    stt = FakeStt(result=make_result(text="X"))
    env = make_pipeline(stt=stt)
    recognized = collect(env.bus, PhraseRecognized)
    errors = collect(env.bus, AppError)
    with running(env.pipeline):
        assert env.pipeline.detach_stt() is stt
        env.pipeline._on_seg_event(SegFinal(utterance_id=1, samples=sample()))
        time.sleep(0.05)  # let the worker pull the job and drop it
    assert env.chatbox.submits == []
    assert recognized == []
    assert not any(e.code == "STT_JOB_FAILED" for e in errors)


def test_set_mt_none_sends_original_text():
    # Swapping MT out to None must not drop the caption: the original text
    # still reaches the chatbox (translation gracefully skipped).
    env = make_pipeline(
        stt=FakeStt(result=make_result(text="hello")), mt=FakeMt()
    )
    with running(env.pipeline):
        env.pipeline.set_mt(None)
        env.pipeline._on_seg_event(SegFinal(utterance_id=1, samples=sample()))
        assert wait_until(lambda: len(env.chatbox.submits) == 1)
    assert any("hello" in text for text, _uid in env.chatbox.submits)


# -- enqueue-then-detach races (direct unit tests on the job processors) -----


def test_process_mt_job_engine_detached_mid_flight_sends_original():
    # The enqueue-then-detach race: the job was already created when
    # detach_mt() pulled the engine. Processing must send the ORIGINAL text
    # and say on the bus that no translation is coming, so the caption row
    # leaves "translating…" even with sending off (nothing else would).
    from vrcc.core import languages, pipeline_jobs
    from vrcc.core.pipeline_jobs import _MtJob

    env = make_pipeline(mt=FakeMt())
    translated = collect(env.bus, PhraseTranslated)
    job = _MtJob(1, "hello", languages.get("English"), manage_typing=True)
    env.pipeline.detach_mt()  # engine leaves AFTER the job exists
    pipeline_jobs.process_mt_job(env.pipeline, job, threading.Event())
    assert env.chatbox.submits == [("hello", 1)]
    assert [(e.utterance_id, e.original, e.translations) for e in translated] == [
        (1, "hello", ())
    ]
    assert env.mt.calls == []  # the detached engine is never invoked


def test_process_stt_job_speculative_dropped_when_engine_detached():
    # Speculative job racing detach_stt(): _call_engine returns the _NO_ENGINE
    # sentinel and the job is dropped whole -- no cache write, no crash.
    from vrcc.core import pipeline_jobs
    from vrcc.core.pipeline_jobs import _SttJob
    from vrcc.core.pipeline_state import _MISSING

    stt = FakeStt(result=make_result(text="X"))
    env = make_pipeline(stt=stt)
    samples = sample()
    key = (1, id(samples))
    env.pipeline._spec.note_speculative(*key)
    env.pipeline.detach_stt()  # engine leaves AFTER the job exists
    pipeline_jobs.process_stt_job(
        env.pipeline, _SttJob(1, samples, True, key[1]), threading.Event()
    )
    assert stt.calls == 0  # the detached engine is never invoked
    assert env.pipeline._spec.pop_result(key) is _MISSING  # nothing cached


# -- the MT-worker deadlock (translation switched on after start) -----------


class _PerFrameSegmenter(FakeSegmenter):
    """Emits one SegFinal per fed frame, so each frame becomes its own
    STT/MT job pair -- the shape that fills both queues past JOB_QUEUE_MAX."""

    def __init__(self) -> None:
        super().__init__()
        self._next_id = 0

    def process(self, frame):
        super().process(frame)
        self._next_id += 1
        return [SegFinal(utterance_id=self._next_id, samples=frame)]


def test_set_mt_after_start_does_not_deadlock_the_stt_worker():
    # Regression: starting with mt=None spawned no MT worker, so set_mt()
    # installed an engine with nothing ever consuming _mt_queue. Job 5
    # blocked the STT worker's _enqueue behind a full MT queue, and the
    # segmenter thread (the sole consumer of the frame queue) stalled with
    # it. 12 frames, JOB_QUEUE_MAX == 4, reproduces it.
    env = make_pipeline(mt=None, segmenter=_PerFrameSegmenter())
    env.config.audio.energy_gate_enabled = False
    with running(env.pipeline):
        env.pipeline.set_mt(FakeMt())
        for _ in range(12):
            env.source.on_frame(sample())
        assert wait_until(lambda: env.stt.calls >= 12, timeout=3.0), env.stt.calls
        assert wait_until(
            lambda: len(env.chatbox.submits) >= 12, timeout=3.0
        ), env.chatbox.submits
        assert env.pipeline._stt_queue.qsize() == 0
        assert env.pipeline._mt_queue.qsize() == 0


def test_mt_worker_thread_exists_after_start_even_when_mt_is_none():
    # The MT worker must run whether or not an engine was installed at start,
    # so a later set_mt() always has a consumer waiting for it.
    env = make_pipeline(mt=None)
    with running(env.pipeline):
        assert env.pipeline._mt_thread is not None
        assert env.pipeline._mt_thread.is_alive()


class _BlockingMt:
    """Never returns until released -- proves a shed job never reaches the
    engine (the test releases it only after asserting that)."""

    def __init__(self) -> None:
        self.calls = 0
        self.gate = threading.Event()

    def translate(self, text, src, targets):
        self.calls += 1
        self.gate.wait(2.0)
        return [(t.display, f"{t.display}:{text}") for t in targets]


def test_full_mt_queue_sheds_the_oldest_job_untranslated_in_order():
    # Shed at the consumer. process_mt_job checks the queue it was just
    # dequeued from (not the engine) before doing any translation work, so a
    # backlog drains as untranslated sends instead of blocking the STT worker
    # behind a slow engine. Never shed at the producer: a late translation
    # landing after an already-sent original would preempt it
    # (osc/chatbox.py's coalesce-latest-wins clears the pending queue on
    # every submit).
    from vrcc.core import languages, pipeline_jobs
    from vrcc.core.pipeline_jobs import _MtJob

    mt = _BlockingMt()
    env = make_pipeline(mt=mt)
    src = languages.get("English")
    for i in range(4):
        env.pipeline._mt_queue.put_nowait(
            _MtJob(i, f"backlog {i}", src, manage_typing=False)
        )
    assert env.pipeline._mt_queue.full()

    for i in range(4, 7):
        job = _MtJob(i, f"oldest {i}", src, manage_typing=True)
        pipeline_jobs.process_mt_job(env.pipeline, job, threading.Event())

    assert mt.calls == 0  # the engine was never entered by a shed job
    assert env.chatbox.submits == [
        ("oldest 4", 4), ("oldest 5", 5), ("oldest 6", 6)
    ]  # FIFO: oldest shed first, in the order they were dequeued
