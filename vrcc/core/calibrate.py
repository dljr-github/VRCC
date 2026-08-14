"""Machine-speed probe backing the model recommender (Qt-free).

``STT_BENCH`` in :mod:`vrcc.core.recommend` was measured on one machine, so its
latency column describes that machine and no other. This module measures how
much slower the machine in front of us is, as a single multiplier the ranking
can apply.

The probe is a fixed matrix multiply rather than real transcription because it
runs inside the first-run wizard, where no voice model is downloaded yet:
deciding what to download is the wizard's whole job.
"""

from __future__ import annotations

import logging
import math
import os
import platform
import statistics
import time

logger = logging.getLogger("vrcc.core.calibrate")

# Seconds for one _MATRIX_N GEMM on the machine STT_BENCH was recorded on
# (Ryzen 9 9950X3D, benchmarks/rtx-5090-ryzen-9950x3d.json), measured the way
# probe_seconds() runs it: fresh process, default BLAS threading, no
# environment overrides.
#
# Deliberately the SLOW end of that machine's own spread, not the middle.
# Eight cold starts ran 0.315 to 0.522 ms, drifting slower as the chip warmed,
# and the recommendation turns on a 1.35x cliff ("small" at 0.74 s against a
# 1.0 s gate). Taking the middle would let the reference machine's own hot
# reading cross its own cliff. Anchoring at the pessimistic end means the
# factor only rises above 1.0 for a machine slower than this one on its worst
# day, which is the same bias as the clamp in measure_factor: quiet unless
# sure.
REFERENCE_GEMM_S = 0.00052

# Probe size, chosen by correlation against real transcription latency over a
# 1/2/4/8/16 thread sweep, not by taste. Whisper's own matmuls are modest, and
# a larger GEMM parallelizes much better than Whisper does (1536 gains 7.1x
# from 1 to 16 threads where "small" gains 2.74x), so a larger probe flatters
# many-core machines. 512 tracked closest: the spread of stt/probe across that
# sweep was 1.15x on "small" against 2.26x at 1536.
_MATRIX_N = 512

# Repeat until the budget is spent, so the cost is bounded on the slow machines
# this exists to detect. The floor keeps a very slow machine from timing only
# one run. Whole probe measures 49 ms warm on the reference machine and about
# 57 ms cold, warm-up included. Cold is the honest figure: this runs once on
# the Qt thread before any window exists, so it is what the first launch after
# an update actually pays.
#
# 48 rather than a handful because the decision is a cliff: "small" measures
# 0.74 s against a 1.0 s gate, so a factor of 1.35 changes the recommendation.
# Over 15 passes, 24 repetitions spread 1.52x (enough noise to flip the
# reference machine's own pick) where 48 spread 1.14x.
_BUDGET_S = 0.25
_MIN_REPS = 3
_MAX_REPS = 48

# Untimed warm-up before the timed repetitions (see probe_seconds).
_WARMUP_S = 0.03


def probe_seconds() -> float:
    """Median seconds for one fixed GEMM on this machine.

    Raises whatever numpy raises; callers that must not fail use
    :func:`measure_factor`.
    """
    import numpy as np

    rng = np.random.default_rng(0)
    a = rng.standard_normal((_MATRIX_N, _MATRIX_N), dtype=np.float32)
    b = rng.standard_normal((_MATRIX_N, _MATRIX_N), dtype=np.float32)

    # Untimed warm-up. One multiply is not enough: it builds the BLAS thread
    # pool but leaves first-touch page faults and clock ramp in the first timed
    # sample, which measured ~25% high. In production this runs exactly once,
    # cold, so that bias would land on every real user and none of it is in
    # REFERENCE_GEMM_S.
    warm_start = time.perf_counter()
    while time.perf_counter() - warm_start < _WARMUP_S:
        a @ b

    times: list[float] = []
    spent = 0.0
    while len(times) < _MAX_REPS and (len(times) < _MIN_REPS or spent < _BUDGET_S):
        start = time.perf_counter()
        a @ b
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        spent += elapsed
    return statistics.median(times)


def measure_factor() -> float | None:
    """How many times slower than the reference machine this one is, or
    ``None`` when the probe could not run.

    Never below 1.0. The clamp is the safety property: a machine at or above
    reference speed keeps exactly the recommendation it gets today, so the
    only recommendations this can move are on slower machines, and it can only
    move them toward smaller models. A probe that reads too pessimistic costs
    some accuracy; one that reads too optimistic lands back on current
    behavior. Neither is worse than not probing.

    ``None`` rather than 1.0 on failure so a caller cannot record "reference
    speed" as a measurement that never happened: a cached 1.0 would never be
    retried, pinning a slow machine to the reference ranking for good.
    """
    try:
        return max(1.0, probe_seconds() / REFERENCE_GEMM_S)
    except Exception:
        logger.debug("machine probe failed; assuming reference speed", exc_info=True)
        return None


def machine_fingerprint() -> str:
    """Identifies the machine well enough to know a stored factor is stale.

    Portable installs travel between PCs on a stick, carrying config.json with
    them, so a cached factor has to be able to notice it was measured
    somewhere else.
    """
    return f"{platform.machine()}|{os.cpu_count()}|{platform.processor()}"


def _stored(cfg) -> float | None:
    """The factor recorded for THIS machine, or ``None`` if there is none.

    ``None`` rather than 1.0 because a measured 1.0 is the common case (every
    machine at or above reference speed clamps to it) and must be
    distinguishable from "never probed", or the cache never hits for exactly
    those machines. A factor recorded against a different fingerprint is
    ignored, not trusted.
    """
    hardware = cfg.hardware
    # isfinite before the comparison: json.loads accepts the Infinity literal
    # and pydantic keeps it as a float, and an infinite factor pushes every
    # measured model past its latency gate, leaving the ranking to fall back on
    # model id. NaN already fails the > 0.0 test.
    if not math.isfinite(hardware.cpu_factor):
        return None
    if hardware.cpu_factor > 0.0 and (
        hardware.cpu_factor_fingerprint == machine_fingerprint()
    ):
        # Floored on read, not only on write. The ranking documents the factor
        # as never below 1.0 and relies on it to stay one-directional; a
        # hand-edited or half-written config must not be able to pull a model
        # back inside a latency gate it does not fit.
        return max(1.0, hardware.cpu_factor)
    return None


def stored_factor(cfg) -> float:
    """:func:`_stored`, or 1.0 when nothing is recorded. Never probes, so a
    surface that only reports a recommendation (the Models window) costs
    nothing to open."""
    existing = _stored(cfg)
    return 1.0 if existing is None else existing


def cached_factor(cfg, remeasure: bool = False) -> float:
    """The stored factor for this machine, probing once if it is missing.

    Updates ``cfg.hardware`` in place when it probes; persisting that is the
    caller's job (the wizard already saves).

    ``remeasure`` discards a stored reading and probes again. A probe taken
    while the machine was busy reads slow and is then kept for good, since the
    fingerprint it is filed under cannot change with load: measured 1.20 on the
    reference machine during a benchmark run, against 1.00 idle. Nothing else
    re-measures, so the recommended-setup button is the one place a user can
    correct it, and that is the only caller that passes this.
    """
    existing = None if remeasure else _stored(cfg)
    if existing is not None:
        return existing

    factor = measure_factor()
    if factor is None:
        return 1.0
    cfg.hardware.cpu_factor = factor
    cfg.hardware.cpu_factor_fingerprint = machine_fingerprint()
    logger.info("machine probe: %.2fx the reference machine's GEMM time", factor)
    return factor
