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
from types import SimpleNamespace

from vrcc.audio.segmenter import SegFinal, SegSpeculative
from vrcc.core.pipeline_stats import SessionStats, SttCallStats, log_summary

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
    "total_audio_s": 0.0,
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
        _stats=SttCallStats(), _session=SessionStats(), **counters
    )


# -- SttCallStats: counters accumulate correctly -----------------------------


def test_record_call_accumulates_totals_and_tracks_the_max():
    stats = SttCallStats()
    stats.record_call(speculative=True, audio_s=1.0, wall_s=0.2)
    stats.record_call(speculative=True, audio_s=2.0, wall_s=0.5)

    snap = stats.snapshot()
    assert snap["speculative_calls"] == 2
    assert snap["final_calls"] == 0
    assert snap["total_audio_s"] == 3.0
    assert snap["total_wall_s"] == 0.7
    assert snap["max_wall_s"] == 0.5  # the larger of the two calls


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
    stats.record_call(speculative=True, audio_s=1.0, wall_s=0.1)
    stats.record_call(speculative=False, audio_s=1.0, wall_s=0.1)
    stats.record_call(speculative=False, audio_s=1.0, wall_s=0.1)

    snap = stats.snapshot()
    assert snap["speculative_calls"] == 1
    assert snap["final_calls"] == 2


def test_record_reuse_does_not_touch_call_counts():
    stats = SttCallStats()
    stats.record_call(speculative=True, audio_s=1.0, wall_s=0.1)
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
    stats.record_call(speculative=True, audio_s=1.0, wall_s=0.1)
    stats.record_call(speculative=False, audio_s=1.0, wall_s=0.1)
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
    p._stats.record_call(speculative=True, audio_s=2.0, wall_s=1.0)
    p._stats.record_call(speculative=False, audio_s=2.0, wall_s=1.0)
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
        "10.0s run time (0.20x). Average 1.00s per call, slowest 1.00s. "
        "1 finals reused a speculative. Dropped 5 frames, about 0.2s of "
        "audio. Skipped 2 speculatives on a full queue (backpressure) and "
        "1 more because the speaker kept talking (normal, costs nothing)."
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
        "run time (n/a). Average 0.00s per call, slowest 0.00s. 0 finals "
        "reused a speculative. Dropped 0 frames, about 0.0s of audio. "
        "Skipped 0 speculatives on a full queue (backpressure) and 0 more "
        "because the speaker kept talking (normal, costs nothing)."
    )


def test_summary_never_raises_when_pipeline_is_missing_attributes():
    # A stats failure must not break stop(): log_summary swallows and logs
    # DEBUG instead of propagating.
    broken = SimpleNamespace()  # no _stats, no _session, no counters at all
    log_summary(broken)  # must not raise


# -- restarting folds into the session instead of fragmenting it (defect 3) -


def test_log_summary_with_restarting_folds_in_but_does_not_log(caplog):
    p = _fake_pipeline(_dropped_frames=3)
    p._stats.record_call(speculative=False, audio_s=1.0, wall_s=0.5)

    with caplog.at_level(logging.INFO, logger=_LOGGER_NAME):
        log_summary(p, restarting=True)

    assert not any(r.name == _LOGGER_NAME for r in caplog.records)
    assert p._session.final_calls == 1
    assert p._session.dropped_frames == 3


def test_log_summary_after_a_restart_reports_the_whole_session(caplog):
    p = _fake_pipeline(_dropped_frames=3)
    p._stats.record_call(speculative=False, audio_s=1.0, wall_s=0.5)
    log_summary(p, restarting=True)  # mid-session device swap: no line yet

    # start() would call begin_run() here, resetting the per-run counters;
    # p._session must not be touched by that.
    p._stats.reset()
    p._dropped_frames = 2
    p._stats.record_call(speculative=False, audio_s=1.0, wall_s=0.5)

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
        p._note_dropped_frame()
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
