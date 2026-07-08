"""Tests for the Qt-free model-recommendation module (``vrcc.core.recommend``):
tier detection passthrough, preset-first ordering, and the ``best_downloaded``
picker.
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
