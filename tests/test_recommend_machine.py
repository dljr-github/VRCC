"""Ranking inputs that describe the machine rather than the model tier.

The benchmark table in :mod:`vrcc.core.bench_tables` was recorded on one PC, so
both of these correct for the PC actually in front of the user: a measured
CPU-speed factor (:mod:`vrcc.core.calibrate`) and the card's total VRAM. Split
from test_recommend.py for the 500-line cap.
"""

from __future__ import annotations

from tests.test_recommend import _EXPECTED_WHISPER_PREFERENCE, _FakeDM, _TIERS
from vrcc.core import recommend
from vrcc.stt.registry import WHISPER_MODELS


# -- machine-speed factor ----------------------------------------------------


def test_factor_one_reproduces_the_reference_ordering():
    # Against the literal, not against WHISPER_PREFERENCE: that table is built
    # by this same call at these same defaults, so comparing the two holds for
    # any implementation and would catch nothing.
    for tier in _TIERS:
        assert (
            recommend._rank_whisper(tier, factor=1.0)
            == _EXPECTED_WHISPER_PREFERENCE[tier]
        )


def test_slow_machine_drops_models_past_the_cpu_latency_gate():
    # "small" measures 0.74s against a 1.0s gate, so it leads the CPU tier on
    # the reference machine and must not on a machine a few times slower.
    assert recommend._rank_whisper("cpu", factor=1.0)[0] == "small"

    on_2x = recommend._rank_whisper("cpu", factor=2.0)[0]
    on_8x = recommend._rank_whisper("cpu", factor=8.0)[0]

    assert on_2x == "base"  # 0.25s -> 0.50s, still inside the gate
    assert on_8x == "tiny"  # 0.13s -> 1.04s, the last one left


def test_factor_leaves_the_gpu_tiers_alone():
    # A CPU probe says nothing about a graphics card, so it must not reorder
    # a GPU tier no matter how slow the processor is.
    for tier in ("gpu_high", "gpu_low"):
        assert (
            recommend._rank_whisper(tier, factor=8.0)
            == recommend.WHISPER_PREFERENCE[tier]
        )


def test_slower_machines_never_get_a_bigger_model():
    # The safety property: the pick is monotonic in the factor. A machine that
    # probes slower can only ever be handed something smaller, so a probe that
    # reads too pessimistic costs accuracy and never costs responsiveness.
    sizes = [
        WHISPER_MODELS[recommend._rank_whisper("cpu", factor=f)[0]].size_mb
        for f in (1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 16.0, 64.0)
    ]
    assert sizes == sorted(sizes, reverse=True), sizes


def test_preset_for_tier_applies_the_factor():
    whisper, _ = recommend.preset_for_tier("cpu", (), 8.0)
    assert whisper == "tiny"


def test_profile_advice_uses_the_same_clock_as_the_model_pick():
    # Sizing the model for a slow machine and then judging its beam against the
    # reference machine's clock hands that machine the wide beam precisely when
    # it has no room for it: "base" at beam 5 measures 0.26s, which is 1.04s at
    # 4x, over the 1.0s gate the model choice just enforced.
    assert recommend.recommended_profile("base", "cpu") == "quality"
    assert recommend.recommended_profile("base", "cpu", 4.0) == "latency"
    assert recommend.recommended_profile("tiny", "cpu", 8.0) == "latency"


def test_profile_advice_ignores_the_factor_on_gpu():
    # A CPU probe says nothing about a graphics card.
    for model in ("large-v3-turbo", "small", "base"):
        assert recommend.recommended_profile(
            model, "cuda", 8.0
        ) == recommend.recommended_profile(model, "cuda")


# -- gpu_low VRAM size cap ---------------------------------------------------


def _fits_gpu_low(vram_mb, model_id):
    """Whether the model ranks inside the gpu_low budget at that card size.
    Over-cap models are not dropped, they fall to the fallback group, so a
    position test is what distinguishes them."""
    ranked = recommend._rank_whisper("gpu_low", vram_mb=vram_mb)
    inside = recommend._rank_whisper("gpu_low", vram_mb=10 ** 6)
    return ranked.index(model_id) <= inside.index(model_id)


def test_unknown_vram_keeps_the_conservative_fixed_cap():
    # No VRAM reading (no pynvml, or the import-time ranking that must not
    # touch NVML) must not silently widen the budget. Against the literal,
    # since WHISPER_PREFERENCE is itself built at vram_mb=None.
    assert (
        recommend._rank_whisper("gpu_low", vram_mb=None)
        == _EXPECTED_WHISPER_PREFERENCE["gpu_low"]
    )


def test_budget_leaves_room_for_the_translation_model():
    # The share exists so the second third can hold the MT preset. If a tier's
    # MT model outgrew its third, the split would be a fiction.
    #
    # Sized at int8, which is what a 6 GB card runs: every card below compute
    # capability 12 has int8 kernels, and only compute capability 12 and above
    # pays the float16 peaks. Comparing a 6 GB card against the float16 row
    # measures a combination that cannot exist.
    from vrcc.core.bench_tables import mt_vram_table

    mt_peak = mt_vram_table("int8_float16")[recommend._MT_PRESET["gpu_low"]]
    assert mt_peak <= 6 * 1024 // recommend._GPU_LOW_VRAM_SHARE

    # A 12 GB Blackwell card is gpu_low too, and it does pay float16.
    from vrcc.core.bench_tables import MT_VRAM_MB

    assert MT_VRAM_MB[recommend._MT_PRESET["gpu_low"]] <= (
        12 * 1024 // recommend._GPU_LOW_VRAM_SHARE
    )


def test_small_card_loses_a_model_a_large_one_keeps():
    # large-v3 peaks at 2741 MB: inside a third of 12 GB, past a third of 6.
    assert _fits_gpu_low(12 * 1024, "large-v3")
    assert not _fits_gpu_low(6 * 1024, "large-v3")


def test_a_6gb_card_is_not_handed_a_strictly_worse_model():
    # The regression this table was measured to fix. Sizing on checkpoint
    # bytes admitted medium (1530 MB file) and rejected large-v3-turbo
    # (1620 MB), although turbo peaks LOWER in memory (1531 against 1597) and
    # beats medium on both word error and latency.
    ranked = recommend._rank_whisper("gpu_low", vram_mb=6 * 1024)
    assert ranked.index("large-v3-turbo") < ranked.index("medium")


def test_tiny_always_fits_whatever_the_card():
    for gb in (2, 4, 6, 8, 12, 15):
        assert _fits_gpu_low(gb * 1024, "tiny"), gb


def test_best_downloaded_respects_the_factor():
    # Everything on disk, so the pick is the ranking's and nothing else.
    dm = _FakeDM(whisper=set(WHISPER_MODELS), mt=set())
    fast, _ = recommend.best_downloaded(dm, translate=False, tier="cpu", factor=1.0)
    slow, _ = recommend.best_downloaded(dm, translate=False, tier="cpu", factor=8.0)

    assert fast == "small"
    assert slow == "tiny"


def test_the_peak_to_file_ratio_range_the_comments_cite():
    """bench_tables and model_fit both justify measuring VRAM by citing this
    spread, so pin it rather than let a table edit quietly falsify two
    comments."""
    from vrcc.core.bench_tables import STT_VRAM_MB
    from vrcc.stt.registry import WHISPER_MODELS

    ratios = [STT_VRAM_MB[m] / WHISPER_MODELS[m].size_mb for m in STT_VRAM_MB]

    assert round(min(ratios), 2) == 0.89
    assert round(max(ratios), 2) == 6.76


def test_vram_table_never_says_a_bigger_model_is_cheaper():
    """A peak that falls as the weight count rises is a measurement error, not a
    finding. `base` sat at 444 against tiny's 507 until it was re-measured."""
    from vrcc.core.bench_tables import STT_VRAM_MB

    ladder = ["tiny", "base", "small", "medium", "large-v3"]
    peaks = [STT_VRAM_MB[model_id] for model_id in ladder]

    assert peaks == sorted(peaks), dict(zip(ladder, peaks))
