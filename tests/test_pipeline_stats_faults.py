"""Tests for pipeline_stats.py's handling of non-finite (NaN/infinite) level
readings: one bad frame must not poison the whole run's RMS distribution,
but the fault it represents must still be named in the summary. Split from
test_pipeline_stats.py, which is already at the 500-line structure cap.
"""

from __future__ import annotations

import logging

from vrcc.core.pipeline_stats import InputStats, log_summary

from tests.test_pipeline_stats import _LOGGER_NAME, _fake_pipeline

# -- InputStats.record_level: excluded from the distribution, still counted -


def test_record_level_keeps_a_nan_reading_out_of_the_distribution_but_counts_it():
    stats = InputStats()
    stats.record_level(0.2)
    stats.record_level(float("nan"))
    stats.record_level(0.4)

    snap = stats.snapshot()
    assert snap["rms_samples"] == [0.2, 0.4]
    assert snap["nonfinite_levels"] == 1


def test_record_level_keeps_infinite_readings_out_of_the_distribution_but_counts_them():
    stats = InputStats()
    stats.record_level(0.2)
    stats.record_level(float("inf"))
    stats.record_level(float("-inf"))
    stats.record_level(0.4)

    snap = stats.snapshot()
    assert snap["rms_samples"] == [0.2, 0.4]
    assert snap["nonfinite_levels"] == 2


# -- log_summary: the fault is named, the distribution stays readable -------

# sorted [0.0, 0.2, 0.4, 0.5, 0.5, 0.5, 0.5, 0.6, 0.8, 1.0]: p10 interpolates
# between index 0 and 1, median sits on the repeated 0.5s, p90 interpolates
# between index 8 and 9. Pinned already, without any non-finite reading, by
# test_summary_line_reports_input_level_clipping_gating_and_latency.
_GOOD_LEVELS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 0.5, 0.5, 0.5, 0.5)
_GOOD_RMS_TEXT = "p10 0.180, median 0.500, p90 0.820 of 1.0 full scale"


def _line_for(levels, caplog, monkeypatch) -> str:
    monkeypatch.setattr("vrcc.core.pipeline_stats.time.monotonic", lambda: 5.0)
    p = _fake_pipeline()
    for level in levels:
        p._input.record_level(level)
    with caplog.at_level(logging.INFO, logger=_LOGGER_NAME):
        log_summary(p)
    return [r.message for r in caplog.records if r.name == _LOGGER_NAME][0]


def test_one_nan_reading_among_many_still_yields_the_good_distribution_and_is_named(
    caplog, monkeypatch
):
    line = _line_for((*_GOOD_LEVELS, float("nan")), caplog, monkeypatch)
    assert (
        f"Input level (frame RMS): {_GOOD_RMS_TEXT} "
        "(non-finite readings excluded: 1)." in line
    )


def test_one_infinite_reading_among_many_still_yields_the_good_distribution_and_is_named(
    caplog, monkeypatch
):
    line = _line_for((*_GOOD_LEVELS, float("inf")), caplog, monkeypatch)
    assert (
        f"Input level (frame RMS): {_GOOD_RMS_TEXT} "
        "(non-finite readings excluded: 1)." in line
    )


def test_a_clean_session_summary_carries_no_extra_clause(caplog, monkeypatch):
    line = _line_for(_GOOD_LEVELS, caplog, monkeypatch)
    assert f"Input level (frame RMS): {_GOOD_RMS_TEXT}." in line
    assert "non-finite" not in line


def test_all_nonfinite_readings_report_no_fabricated_distribution_but_name_the_fault(
    caplog, monkeypatch
):
    # No sample here is ever finite: _percentiles must not be asked for a
    # distribution that does not exist, and the summary must not read like a
    # quiet session when every reading it got was a fault.
    line = _line_for(
        (float("nan"), float("inf"), float("-inf")), caplog, monkeypatch
    )
    assert "Input level (frame RMS): n/a (non-finite readings excluded: 3)." in line
