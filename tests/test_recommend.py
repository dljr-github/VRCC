"""Tests for the Qt-free model-recommendation module (``vrcc.core.recommend``):
tier detection passthrough, benchmark-derived orderings, and the
``best_downloaded`` picker.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from vrcc.core import recommend
from vrcc.stt.registry import WHISPER_MODELS
from vrcc.translate.registry import MT_MODELS

_TIERS = ["gpu_high", "gpu_low", "cpu"]


class _FakeDM:
    """Minimal DownloadManager surface: tracks which ids are 'downloaded'."""

    def __init__(self, whisper=(), mt=()) -> None:
        self._w = set(whisper)
        self._m = set(mt)

    def is_whisper_downloaded(self, model_id: str) -> bool:
        return model_id in self._w

    def is_mt_downloaded(self, spec) -> bool:
        return spec.id in self._m


def test_presets_cover_all_tiers():
    assert set(recommend.PRESETS) == set(_TIERS)


# These document the benchmark-derived outcome (STT_BENCH, measured on the
# reference machine recorded in benchmarks/rtx-5090-ryzen-9950x3d.json): a
# registry or benchmark change that reorders a tier must show up here as a
# conscious diff, not silently reshuffle the recommendations.
_EXPECTED_WHISPER_PREFERENCE = {
    # canary-1b-v2 medians 0.67 s on CUDA, past the 0.6 s GPU budget, so the
    # most accurate model is only a fallback there.
    "gpu_high": [
        "large-v3-turbo", "large-v3", "medium", "small", "base", "tiny",
        "parakeet-tdt-0.6b-v3", "distil-large-v3.5", "distil-small.en",
        "canary-1b-v2",
    ],
    # large-v3 (3090 MB) fails the gpu_low VRAM cap and drops to the
    # unrestricted tail.
    "gpu_low": [
        "large-v3-turbo", "medium", "small", "base", "tiny", "large-v3",
        "parakeet-tdt-0.6b-v3", "distil-large-v3.5", "distil-small.en",
        "canary-1b-v2",
    ],
    # On CPU both NeMo exports stay inside the 1.0 s budget and beat every
    # unrestricted model on accuracy, but they are language-restricted, so a
    # language-blind walk still keeps them behind small/base/tiny.
    "cpu": [
        "small", "base", "tiny", "medium", "large-v3-turbo", "large-v3",
        "canary-1b-v2", "parakeet-tdt-0.6b-v3",
        "distil-small.en", "distil-large-v3.5",
    ],
}


@pytest.mark.parametrize("tier", _TIERS)
def test_derived_whisper_preference_matches_benchmark_outcome(tier):
    assert recommend.WHISPER_PREFERENCE[tier] == _EXPECTED_WHISPER_PREFERENCE[tier]


def test_preset_whisper_ids_lead_their_derived_lists():
    # _validate() already enforces preset-leads; pin the concrete ids so a
    # benchmark edit that changes a tier's default is an explicit test diff.
    assert recommend.PRESETS["gpu_high"][0] == "large-v3-turbo"
    assert recommend.PRESETS["gpu_low"][0] == "large-v3-turbo"
    assert recommend.PRESETS["cpu"][0] == "small"
    for tier in _TIERS:
        assert recommend.WHISPER_PREFERENCE[tier][0] == recommend.PRESETS[tier][0]


def test_rank_whisper_synthetic_ordering_rules():
    """The ranking rules on a synthetic table: WER-band ties resolve by
    latency, unrestricted precede restricted, over-budget models sink to the
    partition tail, and unbenchmarked ids trail everything in their partition.
    """
    specs = {
        "slow-sharp": SimpleNamespace(size_mb=400, languages=None),
        "fast-loose": SimpleNamespace(size_mb=300, languages=None),
        "over-budget": SimpleNamespace(size_mb=200, languages=None),
        "unbenched-big": SimpleNamespace(size_mb=500, languages=None),
        "unbenched-small": SimpleNamespace(size_mb=100, languages=None),
        "restricted": SimpleNamespace(size_mb=50, languages=("en",)),
    }
    bench = {
        # same WER band (0.030 and 0.032 both band 10): the faster model must
        # win even though the slower one has strictly lower WER
        "slow-sharp": (0.030, 0.030, 0.30, 0.30),
        "fast-loose": (0.032, 0.032, 0.10, 0.10),
        # best WER of the partition, but over every latency budget
        "over-budget": (0.010, 0.010, 5.0, 5.0),
        # best numbers overall, but language-restricted
        "restricted": (0.001, 0.001, 0.01, 0.01),
    }
    got = recommend._rank_whisper("gpu_high", specs=specs, bench=bench)
    assert got == [
        "fast-loose", "slow-sharp",          # usable: band tie -> latency
        "over-budget",                       # gate failure -> after usable
        "unbenched-small", "unbenched-big",  # no data -> partition tail, by size
        "restricted",                        # trails every unrestricted id
    ]


def test_rank_whisper_gpu_low_size_cap():
    specs = {
        "big": SimpleNamespace(size_mb=2500, languages=None),
        "little": SimpleNamespace(size_mb=400, languages=None),
    }
    bench = {
        "big": (0.010, 0.010, 0.05, 0.05),  # best WER and latency, but...
        "little": (0.050, 0.050, 0.10, 0.10),
    }
    # ...over 2000 MB cannot be trusted beside VRChat in < 8 GB VRAM
    assert recommend._rank_whisper("gpu_low", specs=specs, bench=bench) == ["little", "big"]
    # the cap is gpu_low-only: gpu_high ranks purely on the measurements
    assert recommend._rank_whisper("gpu_high", specs=specs, bench=bench) == ["big", "little"]


# Language-aware ranking: a known spoken language (Whisper code) lets a
# restricted specialist compete against the unrestricted models. Exact
# outcomes below come from the same STT_BENCH reference run.


@pytest.mark.parametrize("tier", _TIERS)
def test_rank_whisper_language_none_is_byte_identical_to_blind_lists(tier):
    assert recommend._rank_whisper(tier, language=None) == _EXPECTED_WHISPER_PREFERENCE[tier]


def test_rank_whisper_cpu_english_puts_the_nemo_exports_first():
    # Once "en" is known, both NeMo exports beat every whisper model on CPU:
    # canary 1.8 percent at 0.32 s, parakeet 2.3 percent at 0.13 s, against
    # small's 3.7 percent at 0.75 s. Canary leads on the lower WER band.
    assert recommend._rank_whisper("cpu", language="en") == [
        "canary-1b-v2", "parakeet-tdt-0.6b-v3",
        "small", "distil-small.en", "base", "tiny",
        "medium", "distil-large-v3.5", "large-v3-turbo", "large-v3",
    ]


def test_rank_whisper_cpu_japanese_matches_language_blind_order():
    # No restricted model covers "ja", so every specialist trails and the
    # list is identical to the language-blind cpu ordering.
    assert recommend._rank_whisper("cpu", language="ja") == _EXPECTED_WHISPER_PREFERENCE["cpu"]


def test_rank_whisper_gpu_high_german_keeps_turbo_first():
    # GPU WER bands: turbo/large-v3 band 5, parakeet band 7, medium band 9,
    # so parakeet slots after large-v3 and before medium. canary serves "de"
    # but medians 0.67 s on CUDA, past the 0.6 s budget, so it trails the
    # in-budget models. The english-only distil pair cannot serve "de".
    assert recommend._rank_whisper("gpu_high", language="de") == [
        "large-v3-turbo", "large-v3", "parakeet-tdt-0.6b-v3",
        "medium", "small", "base", "tiny",
        "canary-1b-v2", "distil-large-v3.5", "distil-small.en",
    ]


def test_rank_whisper_english_only_flag_trails_without_languages_tuple():
    # english_only is honored even when a spec forgets its languages tuple:
    # the flag alone must keep the model out of a non-English leading group.
    specs = {
        "plain": SimpleNamespace(size_mb=300, languages=None, english_only=False),
        "en-flag-only": SimpleNamespace(size_mb=300, languages=None, english_only=True),
    }
    bench = {
        "plain": (0.050, 0.050, 0.10, 0.10),
        "en-flag-only": (0.010, 0.010, 0.01, 0.01),
    }
    got = recommend._rank_whisper("gpu_high", specs=specs, bench=bench, language="de")
    assert got == ["plain", "en-flag-only"]


def test_canary_competes_only_with_concrete_language():
    blind = recommend._rank_whisper("cpu")
    with_de = recommend._rank_whisper("cpu", language="de")
    # language-blind: canary trails every unrestricted id
    assert blind.index("canary-1b-v2") > blind.index("tiny")
    # concrete "de" on CPU, where canary is inside the budget: it leads
    assert with_de[0] == "canary-1b-v2"


def test_preset_for_choice_language_reranks_whisper_half_only():
    assert recommend.preset_for_choice("cpu", language="en") == (
        "canary-1b-v2", "nllb-600M-int8",
    )
    assert recommend.preset_for_choice("gpu", tier="gpu_high", language="de") == (
        "large-v3-turbo", "nllb-1.3B-int8",
    )


# Per-model performance mode, from the beam-1 vs beam-5 runs (BEAM_BENCH).


@pytest.mark.parametrize(
    ("model_id", "device", "expected"),
    [
        # Real gains for a cost too small to feel: 5.7 -> 4.7 percent for
        # 20 ms on GPU, 5.9 -> 4.9 percent for 10 ms on CPU.
        ("base", "cuda", "quality"),
        ("base", "cpu", "quality"),
        ("tiny", "cuda", "quality"),
        ("medium", "cuda", "quality"),
        # No accuracy to gain: turbo and large-v3 are unchanged or worse at
        # beam 5, and small's 0.2 point is below the noise floor.
        ("large-v3-turbo", "cuda", "latency"),
        ("large-v3", "cuda", "latency"),
        ("small", "cuda", "latency"),
        ("distil-small.en", "cuda", "latency"),
        # Already past the CPU budget at beam 1: widening only costs more.
        ("medium", "cpu", "latency"),
        ("large-v3", "cpu", "latency"),
    ],
)
def test_recommended_profile_follows_the_beam_measurements(model_id, device, expected):
    assert recommend.recommended_profile(model_id, device) == expected


@pytest.mark.parametrize("model_id", ["parakeet-tdt-0.6b-v3", "canary-1b-v2"])
def test_recommended_profile_is_silent_for_greedy_decoders(model_id):
    # The onnx-asr exports have no beam to widen, which is why the Mode
    # control greys out for them: there is nothing to recommend.
    assert recommend.recommended_profile(model_id, "cuda") is None
    assert recommend.recommended_profile(model_id, "cpu") is None


def test_recommended_profile_is_silent_for_unknown_ids():
    assert recommend.recommended_profile("not-a-model", "cuda") is None


def test_recommended_profile_stays_silent_when_unmeasured_but_in_budget(monkeypatch):
    # A whisper model with no beam-5 row, comfortably inside its budget: no
    # evidence either way, so no advice.
    monkeypatch.setitem(recommend.STT_BENCH, "small", (0.03, 0.03, 0.05, 0.05))
    monkeypatch.delitem(recommend.BEAM_BENCH, "small")
    assert recommend.recommended_profile("small", "cuda") is None


def test_best_downloaded_language_prefers_covering_specialist():
    dm = _FakeDM(whisper={"parakeet-tdt-0.6b-v3", "small"})
    # language-blind: small ranks above the restricted parakeet
    assert recommend.best_downloaded(dm, translate=False, tier="cpu")[0] == "small"
    # with "en" known, parakeet leads the cpu tier
    got = recommend.best_downloaded(dm, translate=False, tier="cpu", language="en")
    assert got[0] == "parakeet-tdt-0.6b-v3"
    # a language parakeet does not cover keeps the blind pick
    got = recommend.best_downloaded(dm, translate=False, tier="cpu", language="ja")
    assert got[0] == "small"


@pytest.mark.parametrize("tier", _TIERS)
def test_preference_lists_cover_every_registry_id(tier):
    assert set(recommend.WHISPER_PREFERENCE[tier]) == set(WHISPER_MODELS)
    assert set(recommend.MT_PREFERENCE[tier]) == set(MT_MODELS)


@pytest.mark.parametrize("tier", _TIERS)
def test_preset_is_first(tier):
    assert recommend.WHISPER_PREFERENCE[tier][0] == recommend.PRESETS[tier][0]
    assert recommend.MT_PREFERENCE[tier][0] == recommend.PRESETS[tier][1]


def test_best_downloaded_uses_detect_tier_when_none(monkeypatch):
    monkeypatch.setattr(recommend, "detect_tier", lambda: "cpu")
    dm = _FakeDM(whisper={recommend.PRESETS["cpu"][0]})
    whisper, mt = recommend.best_downloaded(dm, translate=False)
    assert whisper == recommend.PRESETS["cpu"][0]
    assert mt is None


def test_best_downloaded_picks_highest_preference_whisper():
    tier = "gpu_high"
    pref = recommend.WHISPER_PREFERENCE[tier]
    # Downloaded: the 2nd- and 4th-ranked ids -> the 2nd (higher) must win.
    dm = _FakeDM(whisper={pref[1], pref[3]})
    whisper, mt = recommend.best_downloaded(dm, translate=False, tier=tier)
    assert whisper == pref[1]
    assert mt is None


def test_best_downloaded_picks_highest_preference_mt():
    tier = "gpu_high"
    wpref = recommend.WHISPER_PREFERENCE[tier]
    mpref = recommend.MT_PREFERENCE[tier]
    dm = _FakeDM(whisper={wpref[2]}, mt={mpref[2], mpref[4]})
    whisper, mt = recommend.best_downloaded(dm, translate=True, tier=tier)
    assert whisper == wpref[2]
    assert mt == mpref[2]


def test_best_downloaded_none_when_nothing_downloaded():
    dm = _FakeDM()
    assert recommend.best_downloaded(dm, translate=True, tier="cpu") == (None, None)


def test_best_downloaded_skips_mt_when_translate_false():
    tier = "cpu"
    dm = _FakeDM(
        whisper={recommend.WHISPER_PREFERENCE[tier][0]},
        mt=set(recommend.MT_PREFERENCE[tier]),  # everything downloaded...
    )
    whisper, mt = recommend.best_downloaded(dm, translate=False, tier=tier)
    assert whisper is not None
    assert mt is None  # ...yet MT is never consulted


def test_preset_for_choice_cpu_ignores_detected_tier():
    assert recommend.preset_for_choice("cpu", tier="gpu_high") == recommend.PRESETS["cpu"]


def test_preset_for_choice_gpu_uses_given_tier():
    assert recommend.preset_for_choice("gpu", tier="gpu_high") == recommend.PRESETS["gpu_high"]
    assert recommend.preset_for_choice("gpu", tier="gpu_low") == recommend.PRESETS["gpu_low"]


def test_preset_for_choice_gpu_detects_tier_when_none(monkeypatch):
    monkeypatch.setattr(recommend, "detect_tier", lambda: "gpu_low")
    assert recommend.preset_for_choice("gpu") == recommend.PRESETS["gpu_low"]


def test_preset_for_choice_gpu_on_cpu_tier_falls_back_to_gpu_low():
    # A forced GPU choice on a machine detected as CPU-only still maps to a
    # GPU-sized preset (the smallest one), never the CPU preset.
    assert recommend.preset_for_choice("gpu", tier="cpu") == recommend.PRESETS["gpu_low"]


def _cfg_with_device(device: str) -> SimpleNamespace:
    return SimpleNamespace(stt=SimpleNamespace(device=device))


def test_tier_for_config_cpu_device_pins_cpu_tier(monkeypatch):
    monkeypatch.setattr(recommend, "detect_tier", lambda: "gpu_high")
    assert recommend.tier_for_config(_cfg_with_device("cpu")) == "cpu"


def test_tier_for_config_other_devices_follow_detected_tier(monkeypatch):
    monkeypatch.setattr(recommend, "detect_tier", lambda: "gpu_high")
    assert recommend.tier_for_config(_cfg_with_device("auto")) == "gpu_high"
    assert recommend.tier_for_config(_cfg_with_device("cuda")) == "gpu_high"


def test_detect_tier_cpu_when_no_cuda(monkeypatch):
    monkeypatch.setattr(recommend, "cuda_device_count", lambda: 0)
    assert recommend.detect_tier() == "cpu"


def test_detect_tier_gpu_high_when_vram_ample(monkeypatch):
    monkeypatch.setattr(recommend, "cuda_device_count", lambda: 1)
    monkeypatch.setattr(recommend, "total_vram_bytes", lambda: 12 * 1024 ** 3)
    assert recommend.detect_tier() == "gpu_high"


def test_detect_tier_gpu_low_when_vram_small_or_unknown(monkeypatch):
    monkeypatch.setattr(recommend, "cuda_device_count", lambda: 1)
    monkeypatch.setattr(recommend, "total_vram_bytes", lambda: 4 * 1024 ** 3)
    assert recommend.detect_tier() == "gpu_low"
    monkeypatch.setattr(recommend, "total_vram_bytes", lambda: None)
    assert recommend.detect_tier() == "gpu_low"


def test_default_device_choice_gpu_at_16gb(monkeypatch):
    from vrcc.core import recommend

    monkeypatch.setattr(recommend, "total_vram_bytes", lambda: 24 * 1024**3)
    assert recommend.default_device_choice() == "gpu"
    monkeypatch.setattr(recommend, "total_vram_bytes", lambda: 16 * 1024**3)
    assert recommend.default_device_choice() == "gpu"
    monkeypatch.setattr(recommend, "total_vram_bytes", lambda: 8 * 1024**3)
    assert recommend.default_device_choice() == "cpu"
    monkeypatch.setattr(recommend, "total_vram_bytes", lambda: None)
    assert recommend.default_device_choice() == "cpu"
