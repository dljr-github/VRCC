"""Tests for :mod:`vrcc.core.calibrate`.

Real timing is not asserted on: a CI box or a busy laptop makes any absolute
number flaky. What is asserted is the shape (a positive median, the clamp, the
fingerprint invalidation, and that a failed probe cannot take the wizard down),
with the probe stubbed wherever a specific speed is needed.
"""

from __future__ import annotations

import pytest

from vrcc.core import calibrate
from vrcc.core.config import AppConfig


def test_probe_returns_a_positive_median():
    assert calibrate.probe_seconds() > 0.0


def test_probe_stays_within_its_time_budget():
    # The wizard blocks on this, so a runaway probe is a visible stall. The
    # bound is generous (the budget plus the warm-up and a slow-CI allowance)
    # because the point is catching a loop that never ends, not timing.
    import time

    start = time.perf_counter()
    calibrate.probe_seconds()
    assert time.perf_counter() - start < 5.0


def test_measure_factor_clamps_a_fast_machine_to_one(monkeypatch):
    monkeypatch.setattr(
        calibrate, "probe_seconds", lambda: calibrate.REFERENCE_GEMM_S / 10
    )
    assert calibrate.measure_factor() == 1.0


def test_measure_factor_reports_a_slow_machine(monkeypatch):
    monkeypatch.setattr(
        calibrate, "probe_seconds", lambda: calibrate.REFERENCE_GEMM_S * 4
    )
    assert calibrate.measure_factor() == pytest.approx(4.0)


def test_measure_factor_survives_a_failing_probe(monkeypatch):
    def boom():
        raise RuntimeError("no BLAS for you")

    monkeypatch.setattr(calibrate, "probe_seconds", boom)

    # None, not 1.0: the wizard must still open, but a failure is not a
    # measurement and must not be cached as one.
    assert calibrate.measure_factor() is None


def test_a_failed_probe_is_not_cached_as_reference_speed(monkeypatch):
    # Caching the 1.0 fallback would make one transient failure permanent:
    # _stored cannot tell it from a real clamp, so nothing would ever re-probe.
    cfg = AppConfig()

    def boom():
        raise RuntimeError("no BLAS for you")

    monkeypatch.setattr(calibrate, "probe_seconds", boom)
    assert calibrate.cached_factor(cfg) == 1.0
    assert cfg.hardware.cpu_factor == 0.0
    assert cfg.hardware.cpu_factor_fingerprint == ""

    monkeypatch.setattr(
        calibrate, "probe_seconds", lambda: calibrate.REFERENCE_GEMM_S * 4
    )
    assert calibrate.cached_factor(cfg) == pytest.approx(4.0)


def test_a_stored_factor_below_one_is_floored(monkeypatch):
    # The ranking documents the factor as never below 1.0 and relies on it to
    # be one-directional. A hand-edited config must not be able to pull a model
    # back inside a latency gate it does not fit.
    cfg = AppConfig()
    cfg.hardware.cpu_factor = 0.1
    cfg.hardware.cpu_factor_fingerprint = calibrate.machine_fingerprint()

    assert calibrate.stored_factor(cfg) == 1.0
    assert calibrate.cached_factor(cfg) == 1.0


def test_cached_factor_probes_once_and_stores_it(monkeypatch):
    calls = []

    def fake_probe():
        calls.append(1)
        return calibrate.REFERENCE_GEMM_S * 3

    monkeypatch.setattr(calibrate, "probe_seconds", fake_probe)
    cfg = AppConfig()

    first = calibrate.cached_factor(cfg)
    second = calibrate.cached_factor(cfg)

    assert first == pytest.approx(3.0)
    assert second == pytest.approx(3.0)
    assert len(calls) == 1  # the second call read the stored value
    assert cfg.hardware.cpu_factor == pytest.approx(3.0)
    assert cfg.hardware.cpu_factor_fingerprint == calibrate.machine_fingerprint()


def test_cached_factor_caches_a_reference_speed_machine(monkeypatch):
    # Regression: a measured 1.0 must not be mistaken for the "never probed"
    # sentinel. measure_factor clamps with max(1.0, ...), so every machine at
    # or above reference speed stores exactly 1.0, which is precisely the
    # class that would re-probe forever if the cache tested the value instead
    # of whether anything was recorded.
    calls = []

    def fake_probe():
        calls.append(1)
        return calibrate.REFERENCE_GEMM_S / 2  # well past reference: clamps to 1.0

    monkeypatch.setattr(calibrate, "probe_seconds", fake_probe)
    cfg = AppConfig()

    assert [calibrate.cached_factor(cfg) for _ in range(4)] == [1.0] * 4
    assert len(calls) == 1
    assert cfg.hardware.cpu_factor == 1.0


def test_cached_factor_reprobes_when_the_machine_changed(monkeypatch):
    monkeypatch.setattr(
        calibrate, "probe_seconds", lambda: calibrate.REFERENCE_GEMM_S * 2
    )
    cfg = AppConfig()
    # A config carried to another PC on a portable install: the number is
    # real, but it was measured somewhere else.
    cfg.hardware.cpu_factor = 9.0
    cfg.hardware.cpu_factor_fingerprint = "some-other-machine|4|whatever"

    assert calibrate.cached_factor(cfg) == pytest.approx(2.0)
    assert cfg.hardware.cpu_factor_fingerprint == calibrate.machine_fingerprint()


def test_fingerprint_is_stable_within_a_machine():
    assert calibrate.machine_fingerprint() == calibrate.machine_fingerprint()


def test_unprobed_config_defaults_to_zero():
    # 0.0 is the "never probed" sentinel; callers must not treat it as a
    # factor, or an unprobed config would rank every model as infinitely fast.
    assert AppConfig().hardware.cpu_factor == 0.0


def test_remeasure_discards_a_stored_reading(monkeypatch):
    # A probe taken while the machine was busy reads slow and is filed under a
    # fingerprint that cannot change with load, so without this there is no way
    # back. Measured 1.20 on the reference machine mid-benchmark, 1.00 idle.
    readings = iter([calibrate.REFERENCE_GEMM_S * 3, calibrate.REFERENCE_GEMM_S])
    monkeypatch.setattr(calibrate, "probe_seconds", lambda: next(readings))
    cfg = AppConfig()

    assert calibrate.cached_factor(cfg) == pytest.approx(3.0)
    assert calibrate.cached_factor(cfg) == pytest.approx(3.0)  # cached

    assert calibrate.cached_factor(cfg, remeasure=True) == pytest.approx(1.0)
    assert cfg.hardware.cpu_factor == pytest.approx(1.0)
    # And the fresh reading is what the next plain call sees.
    assert calibrate.cached_factor(cfg) == pytest.approx(1.0)
