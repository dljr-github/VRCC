"""Tests for :mod:`vrcc.core.pipeline_stats` -- the per-call STT timing
accumulator, the whole-session cumulative totals it feeds, and the
end-of-run summary line. ``SttCallStats`` is exercised directly;
``log_summary`` is exercised against a lightweight stand-in (message
content, the restart fold-in) and through a real :class:`Pipeline` run (the
``start()`` reset and the speculative/final/reuse split as the pipeline
itself would produce them).
"""

from __future__ import annotations

import logging
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from vrcc.audio.segmenter import SegFinal, SegSpeculative
from vrcc.core import pipeline_jobs, pipeline_stats
from vrcc.core.pipeline_stats import (
    InputStats, LatencyTracker, SessionStats, SttCallStats, log_summary,
)

from .conftest import FakeStt, make_pipeline, make_result, running, sample, wait_until

_LOGGER_NAME = "vrcc.core.pipeline"

# The counters SttCallStats.snapshot() reports besides run_start, all zero on
# a fresh instance or after reset().
_ZERO_CALL_COUNTS = {
    "speculative_calls": 0,
    "final_calls": 0,
    "reuse_count": 0,
    "total_wall_s": 0.0,
    "max_wall_s": 0.0,
    "total_wait_s": 0.0,
    "max_wait_s": 0.0,
    "total_audio_s": 0.0,
    "latency_samples": [],
}


def _without_run_start(snap: dict) -> dict:
    """run_start is a real time.monotonic() reading: never equal between two
    independent instances, so callers checking "the rest is zeroed" compare
    with this key dropped."""
    return {k: v for k, v in snap.items() if k != "run_start"}


def _fake_pipeline(**counters) -> SimpleNamespace:
    counters.setdefault("_dropped_frames", 0)
    counters.setdefault("_skipped_speculatives", 0)
    counters.setdefault("_stale_speculatives", 0)
    return SimpleNamespace(
        _stats=SttCallStats(), _input=InputStats(), _session=SessionStats(),
        **counters,
    )


# -- SttCallStats: counters accumulate correctly -----------------------------


def test_record_call_accumulates_totals_and_tracks_the_max():
    stats = SttCallStats()
    stats.record_call(speculative=True, audio_s=1.0, wait_s=0.05, call_s=0.15)
    stats.record_call(speculative=True, audio_s=2.0, wait_s=0.1, call_s=0.4)

    snap = stats.snapshot()
    assert snap["speculative_calls"] == 2
    assert snap["final_calls"] == 0
    assert snap["total_audio_s"] == 3.0
    assert snap["total_wall_s"] == 0.7  # (0.05+0.15) + (0.1+0.4)
    assert snap["max_wall_s"] == 0.5  # the larger of the two calls


def test_record_call_tracks_wait_and_call_as_separate_totals():
    # A lock wait and a slow model must land in different counters, so a
    # model swap stalling a call reads apart from a genuinely slow engine.
    stats = SttCallStats()
    stats.record_call(speculative=False, audio_s=1.0, wait_s=0.3, call_s=0.1)
    stats.record_call(speculative=False, audio_s=1.0, wait_s=0.05, call_s=0.2)

    snap = stats.snapshot()
    assert snap["total_wait_s"] == pytest.approx(0.35)
    assert snap["max_wait_s"] == pytest.approx(0.3)  # the first call's wait
    assert snap["total_wall_s"] == pytest.approx(0.65)  # unaffected by the split
    assert snap["max_wall_s"] == pytest.approx(0.4)  # 0.3+0.1, the slower call overall


def test_snapshot_starts_at_zero():
    snap = SttCallStats().snapshot()
    assert isinstance(snap["run_start"], float)
    assert _without_run_start(snap) == _ZERO_CALL_COUNTS


def test_reset_refreshes_run_start():
    stats = SttCallStats()
    first = stats.snapshot()["run_start"]
    stats.reset()
    second = stats.snapshot()["run_start"]
    assert second >= first  # a later (or, on a coarse clock, equal) reading


# -- per-call recording distinguishes speculative, final, reuse -------------


def test_record_call_splits_speculative_and_final_counts():
    stats = SttCallStats()
    stats.record_call(speculative=True, audio_s=1.0, wait_s=0.0, call_s=0.1)
    stats.record_call(speculative=False, audio_s=1.0, wait_s=0.0, call_s=0.1)
    stats.record_call(speculative=False, audio_s=1.0, wait_s=0.0, call_s=0.1)

    snap = stats.snapshot()
    assert snap["speculative_calls"] == 1
    assert snap["final_calls"] == 2


def test_record_reuse_does_not_touch_call_counts():
    stats = SttCallStats()
    stats.record_call(speculative=True, audio_s=1.0, wait_s=0.0, call_s=0.1)
    stats.record_reuse()
    stats.record_reuse()

    snap = stats.snapshot()
    assert snap["reuse_count"] == 2
    assert snap["speculative_calls"] == 1
    assert snap["final_calls"] == 0
    # a reuse contributes no wall/audio time: it made no engine call
    assert snap["total_wall_s"] == 0.1
    assert snap["total_audio_s"] == 1.0


def test_reset_clears_every_counter():
    stats = SttCallStats()
    stats.record_call(speculative=True, audio_s=1.0, wait_s=0.0, call_s=0.1)
    stats.record_call(speculative=False, audio_s=1.0, wait_s=0.0, call_s=0.1)
    stats.record_reuse()

    stats.reset()

    assert _without_run_start(stats.snapshot()) == _ZERO_CALL_COUNTS


# -- log_summary: the line reports what it claims ----------------------------


def test_summary_line_reports_the_call_split_per_call_speed_and_keep_up_ratio(
    caplog, monkeypatch
):
    # run_start is read once, when SttCallStats() is constructed; the
    # keep-up ratio's elapsed time is read once more at summary time -- 10.0s
    # apart, by construction, not by sleeping.
    times = iter([0.0, 10.0])
    monkeypatch.setattr(
        "vrcc.core.pipeline_stats.time.monotonic", lambda: next(times)
    )
    p = _fake_pipeline(_dropped_frames=5, _skipped_speculatives=2, _stale_speculatives=1)
    p._stats.record_call(speculative=True, audio_s=2.0, wait_s=0.0, call_s=1.0)
    p._stats.record_call(speculative=False, audio_s=2.0, wait_s=0.0, call_s=1.0)
    p._stats.record_reuse()

    with caplog.at_level(logging.INFO, logger=_LOGGER_NAME):
        log_summary(p)

    lines = [r.message for r in caplog.records if r.name == _LOGGER_NAME]
    assert len(lines) == 1
    line = lines[0]
    # 2 calls, 4.0s fed to the engine in 2.0s of engine time (0.50x per
    # call); that same 2.0s against the run's 10.0s wall clock is the
    # keep-up ratio (0.20x: under 1.0, so the engine was not saturated).
    assert line == (
        "STT run: 2 calls (1 speculative, 1 final). 2.0s engine time on "
        "4.0s of audio fed to it (0.50x per call). Engine busy 2.0s of "
        "10.0s run time (0.20x). Average 1.00s per call, slowest 1.00s; "
        "0.00s of that was spent waiting for the engine lock (0%), slowest "
        "wait 0.00s. 1 finals reused a speculative. Dropped 5 frames, about "
        "0.2s of audio. Skipped 2 speculatives on a full queue "
        "(backpressure) and 1 more because the speaker kept talking "
        "(normal, costs nothing). Input level (frame RMS): n/a. n/a of "
        "frames clipped. 0 finals suppressed by the quality gate. "
        "Finalize-to-chatbox latency: n/a."
    )


def test_summary_line_reports_na_when_the_run_made_no_calls(caplog, monkeypatch):
    # No calls at all, and the run-start/summary-time reads land on the same
    # instant: nothing to divide by either way, must not raise or claim a
    # bogus ratio.
    monkeypatch.setattr("vrcc.core.pipeline_stats.time.monotonic", lambda: 5.0)
    p = _fake_pipeline()

    with caplog.at_level(logging.INFO, logger=_LOGGER_NAME):
        log_summary(p)

    lines = [r.message for r in caplog.records if r.name == _LOGGER_NAME]
    assert len(lines) == 1
    assert lines[0] == (
        "STT run: 0 calls (0 speculative, 0 final). 0.0s engine time on "
        "0.0s of audio fed to it (n/a per call). Engine busy 0.0s of 0.0s "
        "run time (n/a). Average 0.00s per call, slowest 0.00s; 0.00s of "
        "that was spent waiting for the engine lock (n/a), slowest wait "
        "0.00s. 0 finals reused a speculative. Dropped 0 frames, about "
        "0.0s of audio. Skipped 0 speculatives on a full queue "
        "(backpressure) and 0 more because the speaker kept talking "
        "(normal, costs nothing). Input level (frame RMS): n/a. n/a of "
        "frames clipped. 0 finals suppressed by the quality gate. "
        "Finalize-to-chatbox latency: n/a."
    )


def test_summary_never_raises_when_pipeline_is_missing_attributes():
    # A stats failure must not break stop(): log_summary swallows and logs
    # DEBUG instead of propagating.
    broken = SimpleNamespace()  # no _stats, no _session, no counters at all
    log_summary(broken)  # must not raise


# -- restarting folds into the session instead of fragmenting it (defect 3) -


def test_log_summary_with_restarting_folds_in_but_does_not_log(caplog):
    p = _fake_pipeline(_dropped_frames=3)
    p._stats.record_call(speculative=False, audio_s=1.0, wait_s=0.0, call_s=0.5)

    with caplog.at_level(logging.INFO, logger=_LOGGER_NAME):
        log_summary(p, restarting=True)

    assert not any(r.name == _LOGGER_NAME for r in caplog.records)
    assert p._session.final_calls == 1
    assert p._session.dropped_frames == 3


def test_log_summary_after_a_restart_reports_the_whole_session(caplog):
    p = _fake_pipeline(_dropped_frames=3)
    p._stats.record_call(speculative=False, audio_s=1.0, wait_s=0.0, call_s=0.5)
    log_summary(p, restarting=True)  # mid-session device swap: no line yet

    # start() would call begin_run() here, resetting the per-run counters;
    # p._session must not be touched by that.
    p._stats.reset()
    p._dropped_frames = 2
    p._stats.record_call(speculative=False, audio_s=1.0, wait_s=0.0, call_s=0.5)

    with caplog.at_level(logging.INFO, logger=_LOGGER_NAME):
        log_summary(p)  # the real stop: one line for the whole session

    lines = [
        r.message for r in caplog.records
        if r.name == _LOGGER_NAME and r.message.startswith("STT run:")
    ]
    assert len(lines) == 1
    assert "2 calls (0 speculative, 2 final)" in lines[0]
    assert "Dropped 5 frames" in lines[0]


# -- log_summary through a real Pipeline run ---------------------------------


def test_stop_logs_one_summary_line_with_the_run_counts(caplog):
    env = make_pipeline(stt=FakeStt(result=make_result()))
    s = sample()
    with caplog.at_level(logging.INFO, logger=_LOGGER_NAME):
        with running(env.pipeline):
            env.pipeline._on_seg_event(SegSpeculative(utterance_id=1, samples=s))
            env.pipeline._on_seg_event(SegFinal(utterance_id=1, samples=s))
            assert wait_until(lambda: env.pipeline._spec._last_finalized >= 1)

    summaries = [
        r.message for r in caplog.records
        if r.name == _LOGGER_NAME and r.message.startswith("STT run:")
    ]
    assert len(summaries) == 1
    # One speculative call transcribed the audio; the final reused its result.
    assert "1 calls (1 speculative, 0 final)" in summaries[0]
    assert "1 finals reused a speculative" in summaries[0]


def test_restart_via_stop_does_not_fragment_the_session_summary(caplog):
    # restart_source/reinit_audio_and_resume call stop(restarting=True) then
    # start() again for a live device swap. The log must carry one line for
    # the whole session, not a fragment per swap, and a drop from before the
    # swap must not be lost when start() resets the per-run counters.
    env = make_pipeline(stt=FakeStt(result=make_result()))
    p = env.pipeline
    s = sample()
    with caplog.at_level(logging.INFO, logger=_LOGGER_NAME):
        p.start()
        p._on_seg_event(SegFinal(utterance_id=1, samples=s))
        assert wait_until(lambda: p._spec._last_finalized >= 1)
        pipeline_stats.note_dropped_frame(p)
        p.stop(restarting=True)  # simulated device swap mid-session

        p.start()
        p._on_seg_event(SegFinal(utterance_id=2, samples=s))
        assert wait_until(lambda: p._spec._last_finalized >= 2)
        p.stop()  # the real stop

    summaries = [
        r.message for r in caplog.records
        if r.name == _LOGGER_NAME and r.message.startswith("STT run:")
    ]
    assert len(summaries) == 1
    assert "2 calls (0 speculative, 2 final)" in summaries[0]
    assert "Dropped 1 frames" in summaries[0]


def test_fresh_start_resets_the_stats_from_the_prior_run():
    env = make_pipeline(stt=FakeStt(result=make_result()))
    p = env.pipeline
    s = sample()
    p.start()
    p._on_seg_event(SegFinal(utterance_id=1, samples=s))
    assert wait_until(lambda: p._spec._last_finalized >= 1)
    p.stop()
    assert p._stats.snapshot()["final_calls"] >= 1  # the stopped run's stats

    p.start()  # a fresh run must not carry the prior run's numbers forward
    assert _without_run_start(p._stats.snapshot()) == _ZERO_CALL_COUNTS
    p.stop()


def test_a_call_that_outlives_stop_is_not_recorded_into_the_next_run():
    # A worker abandoned by stop()'s join timeout returns into the run that
    # follows; its call belongs to neither summary.
    env = make_pipeline(stt=FakeStt(result=make_result()))
    p = env.pipeline
    p._join_timeout_s = 0.05
    env.stt.gate.clear()
    p.start()
    zombie = p._stt_thread
    p._on_seg_event(SegFinal(utterance_id=1, samples=sample()))
    assert env.stt.entered.wait(2.0)
    p.stop()  # the join times out with the worker still inside transcribe
    p.start()

    env.stt.gate.set()
    zombie.join(2.0)

    assert not zombie.is_alive()
    assert _without_run_start(p._stats.snapshot()) == _ZERO_CALL_COUNTS
    p.stop()


# -- InputStats: what the mic delivered --------------------------------------


def test_input_stats_record_level_and_frame_accumulate():
    stats = InputStats()
    stats.record_level(0.1)
    stats.record_level(0.9)
    stats.record_frame(np.full(8, 0.1, dtype=np.float32))  # well under scale
    stats.record_frame(np.array([1.0, 0.0, -1.0, 0.2], dtype=np.float32))  # clips

    snap = stats.snapshot()
    assert snap["rms_samples"] == [0.1, 0.9]
    assert snap["frame_count"] == 2
    assert snap["clipped_frames"] == 1  # only the second frame touched full scale


def test_input_stats_record_gate_suppressed_counts_each_call():
    stats = InputStats()
    stats.record_gate_suppressed()
    stats.record_gate_suppressed()
    assert stats.snapshot()["gated_utterances"] == 2


def test_input_stats_reset_clears_every_counter():
    stats = InputStats()
    stats.record_level(0.5)
    stats.record_frame(np.ones(4, dtype=np.float32))
    stats.record_gate_suppressed()

    stats.reset()

    assert stats.snapshot() == {
        "rms_samples": [],
        "frame_count": 0,
        "clipped_frames": 0,
        "gated_utterances": 0,
    }


def test_gate_suppressed_final_is_counted_through_a_real_run():
    # engine.transcribe() returning None (quality gate) is only visible to
    # forward_final as result=None; this pins that it reaches InputStats.
    env = make_pipeline(stt=FakeStt(result=None))
    with running(env.pipeline):
        env.pipeline._on_seg_event(SegFinal(utterance_id=1, samples=sample()))
        assert wait_until(lambda: env.pipeline._spec._last_finalized >= 1)
    assert env.pipeline._input.snapshot()["gated_utterances"] == 1


# -- LatencyTracker: finalize-to-submit span ---------------------------------


def test_latency_tracker_reports_elapsed_since_note_finalized(monkeypatch):
    times = iter([100.0, 100.75])
    monkeypatch.setattr(
        "vrcc.core.pipeline_stats.time.monotonic", lambda: next(times)
    )
    tracker = LatencyTracker()
    tracker.note_finalized(7)
    assert tracker.pop_elapsed(7) == pytest.approx(0.75)


def test_latency_tracker_pop_is_none_for_an_unknown_or_already_popped_id():
    tracker = LatencyTracker()
    assert tracker.pop_elapsed(1) is None  # typed text: never finalized
    tracker.note_finalized(2)
    tracker.pop_elapsed(2)
    assert tracker.pop_elapsed(2) is None  # already closed


def test_latency_tracker_reset_clears_pending_starts():
    tracker = LatencyTracker()
    tracker.note_finalized(1)
    tracker.reset()
    assert tracker.pop_elapsed(1) is None


def test_finalize_to_submit_latency_is_recorded_for_a_normal_final():
    env = make_pipeline(stt=FakeStt(result=make_result()), mt=None)
    p = env.pipeline
    with running(p):
        p._on_seg_event(SegFinal(utterance_id=1, samples=sample()))
        assert wait_until(lambda: len(env.chatbox.submits) == 1)
    samples_recorded = p._stats.snapshot()["latency_samples"]
    assert len(samples_recorded) == 1
    assert samples_recorded[0] >= 0.0
    assert p._latency.pop_elapsed(1) is None  # already closed by the submit


# -- lock-wait timed apart from the engine call ------------------------------


def test_lock_wait_is_recorded_separately_from_the_engine_call():
    # A model swap (or the 'heard' stream) holding stt_slot's lock stalls the
    # STT worker before it ever reaches transcribe(): that stall must show up
    # as wait_s, not get folded into call_s as if the model itself were slow.
    env = make_pipeline(stt=FakeStt(result=make_result()))
    p = env.pipeline
    held = threading.Event()
    release = threading.Event()

    def hold_lock():
        with p.stt_slot.borrow():
            held.set()
            release.wait(2.0)

    holder = threading.Thread(target=hold_lock, daemon=True)
    with running(p):
        holder.start()
        assert held.wait(2.0)
        p._on_seg_event(SegFinal(utterance_id=1, samples=sample()))
        time.sleep(0.15)  # the STT worker is now blocked acquiring the lock
        release.set()
        assert wait_until(lambda: p._spec._last_finalized >= 1)
    holder.join(2.0)

    snap = p._stats.snapshot()
    assert snap["final_calls"] == 1
    assert snap["total_wait_s"] > 0.05  # well under the 0.15s hold, coarse clock included
    assert snap["max_wait_s"] >= snap["total_wait_s"]  # one call this run


# -- _emit: the input-character and latency clauses report what they claim --


def test_summary_line_reports_input_level_clipping_gating_and_latency(
    caplog, monkeypatch
):
    monkeypatch.setattr("vrcc.core.pipeline_stats.time.monotonic", lambda: 5.0)
    p = _fake_pipeline()
    for level in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 0.5, 0.5, 0.5, 0.5):
        p._input.record_level(level)
    p._input.record_frame(np.full(4, 0.1, dtype=np.float32))
    p._input.record_frame(np.array([1.0, 0.0], dtype=np.float32))  # clips
    p._input.record_gate_suppressed()
    p._input.record_gate_suppressed()
    p._stats.record_latency(1.0)
    p._stats.record_latency(3.0)

    with caplog.at_level(logging.INFO, logger=_LOGGER_NAME):
        log_summary(p)

    line = [r.message for r in caplog.records if r.name == _LOGGER_NAME][0]
    # sorted levels [0.0, 0.2, 0.4, 0.5, 0.5, 0.5, 0.5, 0.6, 0.8, 1.0]: p10
    # interpolates between index 0 and 1, median sits on the repeated 0.5s,
    # p90 interpolates between index 8 and 9 (see _percentile).
    assert (
        "Input level (frame RMS): p10 0.180, median 0.500, p90 0.820 of 1.0 "
        "full scale." in line
    )
    assert "50.00% of frames clipped." in line
    assert "2 finals suppressed by the quality gate." in line
    assert "Finalize-to-chatbox latency: median 2.00s, p90 2.80s." in line
