"""Ranking inputs that describe the machine rather than the model tier.

The benchmark table in :mod:`vrcc.core.bench_tables` was recorded on one PC, so
both of these correct for the PC actually in front of the user: a measured
CPU-speed factor (:mod:`vrcc.core.calibrate`) and the card's total VRAM. Split
from test_recommend.py for the 500-line cap.
"""

from __future__ import annotations

import pytest

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


def test_marginal_mt_cost_is_never_its_own_standalone_peak():
    # _MT_MARGINAL_MB is the cost of adding the translation model to an
    # already-resident STT model, not the MT model's own full footprint:
    # MT_VRAM_INT8_MB already pays for a CUDA context of its own, and the
    # measured pair shares one context rather than paying for two. A budget
    # that mistakenly charged the standalone peak would starve the voice
    # model of headroom the two processes never actually spend twice.
    from vrcc.core.bench_tables import mt_vram_table

    mt_standalone = mt_vram_table("int8_float16")[recommend._MT_PRESET["gpu_low"]]
    assert recommend._MT_MARGINAL_MB < mt_standalone


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


# -- the VRChat reservation ---------------------------------------------


@pytest.mark.parametrize(
    ("nvml_mb", "expected"),
    [
        (8151, "small"),          # the 8 GB compromise
        (10239, "small"),         # raw budget (1022) is under turbo's peak
        (12287, "large-v3-turbo"),  # headroom past the reservation reopens it
    ],
)
def test_the_decision_table_at_int8_float16(nvml_mb, expected):
    whisper, _mt = recommend.preset_for_tier(
        "gpu_low", (), 1.0, nvml_mb, "int8_float16",
    )
    assert whisper == expected


def test_float16_10gb_is_the_case_the_floor_exists_for(monkeypatch):
    # Raw budget (1022) is under small's fp16 peak (1115): unfloored, small
    # itself would not fit and the blind/non-English pick falls to "base".
    # The floor lands both back on "small".
    whisper, _mt = recommend.preset_for_tier("gpu_low", (), 1.0, 10239, "float16")
    assert whisper == "small"
    whisper_ja, _mt = recommend.preset_for_tier(
        "gpu_low", ("ja",), 1.0, 10239, "float16"
    )
    assert whisper_ja == "small"

    unfloored = 10239 - recommend._VRCHAT_RESERVE_MB - recommend._MT_MARGINAL_MB
    monkeypatch.setattr(recommend, "vram_budget_mb", lambda total_mb, compute: unfloored)
    ranked = recommend._rank_whisper(
        "gpu_low", languages=("ja",), vram_mb=10239, compute="float16"
    )
    assert ranked[0] == "base"


def test_an_english_speaker_is_not_the_floors_case():
    # Parakeet carries no measured VRAM row (onnx backend, never gated on
    # gpu_low), so an English speaker lands on it with or without the floor;
    # the floor's job is the non-English and language-blind picks above.
    whisper, _mt = recommend.preset_for_tier("gpu_low", ("en",), 1.0, 10239, "float16")
    assert whisper == "parakeet-tdt-0.6b-v3"


def test_validate_ties_the_floor_to_the_cpu_preset(monkeypatch):
    assert recommend._FLOOR_WHISPER_ID == recommend.PRESETS["cpu"][0]

    monkeypatch.setattr(recommend, "_FLOOR_WHISPER_ID", "not-the-cpu-preset")
    with pytest.raises(ValueError):
        recommend._validate()


def test_unknown_vram_is_charged_exactly_the_reservation():
    # The conservative reading this session settled on: an unreadable card
    # (no pynvml, or the import-time ranking, which must not touch NVML) gets
    # exactly what vram_budget_mb would give a card offering VRChat its
    # reservation and nothing more, not a separately chosen number.
    assert recommend._rank_whisper("gpu_low", vram_mb=None) == recommend._rank_whisper(
        "gpu_low", vram_mb=recommend._VRCHAT_RESERVE_MB
    )
