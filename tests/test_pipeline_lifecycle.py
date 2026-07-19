"""Tests for :mod:`vrcc.core.pipeline` -- the energy pre-gate, frame-queue
backpressure, and start/stop lifecycle (including the zombie-worker case).
"""

from __future__ import annotations

import logging
import time

import numpy as np
import pytest

from vrcc.audio.segmenter import SegFinal
from vrcc.core.config import AppConfig, AudioConfig
from vrcc.core.events import AppError, MicLevel, PhraseRecognized

from .conftest import FakeSource, FakeStt, collect, make_pipeline, running, sample


def _wait_until(predicate, timeout: float = 2.0, interval: float = 0.005) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return bool(predicate())


# -- energy pre-gate ---------------------------------------------------------

# energy_threshold=1000 -> float32 rms cutoff 1000/32768 ~= 0.0305.
_GATED_CFG = dict(energy_gate_enabled=True, energy_threshold=1000)


def test_energy_gate_blocks_quiet_frames_when_idle():
    cfg = AppConfig(audio=AudioConfig(**_GATED_CFG))
    env = make_pipeline(config=cfg)
    levels = collect(env.bus, MicLevel)
    with running(env.pipeline):
        env.pipeline._on_frame(sample(v=0.001))  # rms 0.001 < 0.0305: gated
        assert _wait_until(lambda: len(levels) == 1)  # meter still flows
        time.sleep(0.02)
    assert env.segmenter.frames == []  # never reached the VAD/segmenter
    assert levels[0].vad_prob == 0.0


def test_energy_gate_passes_loud_frames():
    cfg = AppConfig(audio=AudioConfig(**_GATED_CFG))
    env = make_pipeline(config=cfg)
    with running(env.pipeline):
        env.pipeline._on_frame(sample(v=0.1))  # rms 0.1 >= 0.0305: passes
        assert _wait_until(lambda: len(env.segmenter.frames) == 1)


def test_energy_gate_lets_quiet_frames_flow_mid_utterance():
    # The gate only blocks utterance STARTS: once the segmenter is ACTIVE,
    # every frame flows until finalize (a quiet word tail must not be cut).
    # Activation happens after start(), which resets in-flight segmenter
    # state, so the fake goes active only once the pipeline runs.
    cfg = AppConfig(audio=AudioConfig(**_GATED_CFG))
    env = make_pipeline(config=cfg)
    with running(env.pipeline):
        env.segmenter.active = True
        env.pipeline._on_frame(sample(v=0.001))
        assert _wait_until(lambda: len(env.segmenter.frames) == 1)


def test_energy_gate_disabled_frames_flow_unchanged():
    env = make_pipeline()  # default config: gate disabled
    with running(env.pipeline):
        env.pipeline._on_frame(sample(v=0.001))
        assert _wait_until(lambda: len(env.segmenter.frames) == 1)


# -- frame queue backpressure ----------------------------------------------


def test_frame_queue_drops_oldest_over_capacity(caplog):
    env = make_pipeline()  # NOT started: nothing drains the frame queue
    pipeline = env.pipeline
    for i in range(100):
        pipeline._on_frame(np.full(1, i, dtype=np.float32))
    assert pipeline._frame_queue.qsize() == 100
    assert pipeline._dropped_frames == 0

    with caplog.at_level(logging.WARNING, logger="vrcc.core.pipeline"):
        pipeline._on_frame(np.full(1, 100, dtype=np.float32))
        pipeline._on_frame(np.full(1, 101, dtype=np.float32))

    assert pipeline._frame_queue.qsize() == 100
    assert pipeline._dropped_frames == 2
    # oldest (0, 1) dropped; newest (101) retained
    drained = []
    while not pipeline._frame_queue.empty():
        drained.append(int(pipeline._frame_queue.get_nowait()[0]))
    assert drained[0] == 2
    assert drained[-1] == 101
    assert sum("dropping oldest" in r.message for r in caplog.records) == 1


# -- lifecycle --------------------------------------------------------------


def test_zombie_stt_worker_from_timed_out_stop_cannot_touch_new_run():
    # stop()'s join times out while the STT worker is blocked inside
    # transcribe -> the worker is abandoned as a zombie. Because workers are
    # bound to their run's queue + stop event, a restart must be invisible
    # to the zombie: it exits via its OWN queue's sentinel, publishes
    # nothing, and never consumes from the new run's queue.
    stt = FakeStt()
    env = make_pipeline(stt=stt)
    p = env.pipeline
    p._join_timeout_s = 0.2  # keep the timed-out joins fast
    recognized = collect(env.bus, PhraseRecognized)

    try:
        p.start()
        old_thread = p._stt_thread
        old_queue = p._stt_queue
        stt.gate.clear()  # block the worker inside transcribe
        p._on_seg_event(SegFinal(utterance_id=1, samples=sample()))
        assert stt.entered.wait(2.0)

        p.stop()  # STT join times out: worker still blocked in transcribe
        assert old_thread.is_alive()

        p.start()  # new run: fresh queues, fresh workers
        new_thread = p._stt_thread
        assert new_thread is not old_thread
        assert p._stt_queue is not old_queue

        stt.gate.set()  # release the zombie's transcribe call

        # Zombie exits via its OWN old queue's sentinel, publishing nothing.
        assert _wait_until(lambda: not old_thread.is_alive())
        assert old_queue.empty()  # the old sentinel was consumed
        assert recognized == []  # the abandoned result was discarded

        # The new run works normally and each job is routed exactly once.
        p._on_seg_event(SegFinal(utterance_id=2, samples=sample()))
        assert _wait_until(lambda: p._spec._last_finalized >= 2)
        assert _wait_until(lambda: len(env.chatbox.submits) == 1)
    finally:
        stt.gate.set()
        p.stop()

    assert stt.calls == 2  # zombie's call + the new run's; no double consume
    assert [e.utterance_id for e in recognized] == [2]
    assert len(env.chatbox.submits) == 1  # no duplicate sends


def test_start_failure_unwinds_workers_and_stays_not_started():
    class BoomSource(FakeSource):
        def start(self, on_frame) -> None:
            raise RuntimeError("no mic")

    env = make_pipeline(source=BoomSource())
    with pytest.raises(RuntimeError, match="no mic"):
        env.pipeline.start()
    # Workers were unwound; the pipeline is as if start() never happened.
    assert env.pipeline._seg_thread is None
    assert env.pipeline._stt_thread is None
    assert env.pipeline._mt_thread is None
    errors = collect(env.bus, AppError)
    env.pipeline.submit_typed("hello")  # not started -> refused, no freeze
    assert [e.code for e in errors] == ["PIPELINE_NOT_RUNNING"]
    env.pipeline.stop()  # safe no-op


def test_start_is_idempotent():
    env = make_pipeline()
    pipeline = env.pipeline
    pipeline.start()
    stt_thread = pipeline._stt_thread
    pipeline.start()  # must not spawn a second worker set
    assert pipeline._stt_thread is stt_thread
    pipeline.stop()


def test_stop_is_idempotent_and_safe_before_start():
    env = make_pipeline()
    pipeline = env.pipeline
    pipeline.stop()  # never started
    pipeline.stop()  # double stop
    pipeline.start()
    pipeline.stop()
    pipeline.stop()
    assert env.source.stopped is True


def test_start_wires_source_callback():
    env = make_pipeline()
    with running(env.pipeline):
        assert env.source.started is True
        assert env.source.on_frame is not None


# -- live source swap -------------------------------------------------------


def test_restart_source_swaps_live_source_and_keeps_capturing():
    env = make_pipeline()
    p = env.pipeline
    p.start()
    old, new = env.source, FakeSource()
    try:
        assert p.restart_source(new) is True  # capture still running afterwards
        assert old.stopped is True
        assert new.started is True
        assert new.on_frame is not None
        assert p._source is new
    finally:
        p.stop()


def test_restart_source_while_stopped_installs_without_capturing():
    env = make_pipeline()
    p = env.pipeline
    new = FakeSource()
    assert p.restart_source(new) is False  # not running -> stays stopped
    assert new.started is False
    assert p._source is new
    p.start()  # a later start() uses the newly installed source
    assert new.started is True
    p.stop()


def test_restart_source_failed_open_reraises_and_leaves_pipeline_consistent():
    class BoomSource(FakeSource):
        def start(self, on_frame) -> None:
            raise RuntimeError("device gone")

    env = make_pipeline()
    p = env.pipeline
    p.start()
    with pytest.raises(RuntimeError, match="device gone"):
        p.restart_source(BoomSource())
    # start() unwound itself: consistent, not running, workers torn down.
    assert p._started is False
    assert p._seg_thread is None
    assert p._stt_thread is None
    p.stop()  # safe no-op


# -- PortAudio host re-init --------------------------------------------------


def test_reinit_audio_and_resume_cycles_capture_through_reinit():
    # A hotplugged mic is invisible to PortAudio until the host is cycled with
    # NO stream open. reinit_audio_and_resume must stop capture first, run the
    # reinit callback while the old source is stopped, THEN build and start
    # the new source (so it resolves against the refreshed device list).
    env = make_pipeline()
    p = env.pipeline
    p.start()
    order: list[str] = []
    old_source = env.source
    old_stop = old_source.stop

    def tracking_stop() -> None:
        order.append("stop")
        old_stop()

    old_source.stop = tracking_stop

    def reinit() -> None:
        assert order == ["stop"]  # runs only while no stream is open
        order.append("reinit")

    new_source = FakeSource()
    try:
        running_after = p.reinit_audio_and_resume(reinit, lambda: new_source)
        assert order == ["stop", "reinit"]
        assert running_after is True
        assert p._source is new_source
        assert new_source.started is True
        assert new_source.on_frame is not None
    finally:
        p.stop()


def test_reinit_audio_and_resume_while_stopped_installs_without_capturing():
    env = make_pipeline()
    p = env.pipeline
    calls: list[str] = []

    def reinit() -> None:
        calls.append("reinit")

    new_source = FakeSource()
    assert p.reinit_audio_and_resume(reinit, lambda: new_source) is False
    assert calls == ["reinit"]
    assert new_source.started is False
    assert p._source is new_source
    p.start()  # a later start() uses the newly installed source
    assert new_source.started is True
    p.stop()
