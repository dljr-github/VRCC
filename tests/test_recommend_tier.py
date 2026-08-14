"""Tests for the hardware verdict in ``vrcc.core.recommend``: tier detection
and the wizard's default device choice. Both are gated on a usable CUDA
runtime (``can_run_cuda``), not on a merely visible device: the bundled
CTranslate2 enumerates GPUs from the display driver even in an install that
ships no cuBLAS, and NVML reports VRAM the same way.
"""

from __future__ import annotations

from vrcc.core import recommend


def test_detect_tier_cpu_when_no_usable_cuda(monkeypatch):
    monkeypatch.setattr(recommend, "can_run_cuda", lambda: False)
    assert recommend.detect_tier() == "cpu"


def test_detect_tier_cpu_when_cuda_unusable_despite_high_vram(monkeypatch):
    # A visible 24 GB card whose install cannot load cuBLAS must not be
    # recommended GPU-sized models: their first load would fall over.
    monkeypatch.setattr(recommend, "can_run_cuda", lambda: False)
    monkeypatch.setattr(recommend, "total_vram_bytes", lambda index=0: 24 * 1024 ** 3)
    assert recommend.detect_tier() == "cpu"


def test_detect_tier_gpu_high_when_vram_ample(monkeypatch):
    # VRChat's own recommended spec is 16 GB, so that is the bar.
    monkeypatch.setattr(recommend, "can_run_cuda", lambda: True)
    monkeypatch.setattr(recommend, "total_vram_bytes", lambda index=0: 24 * 1024 ** 3)
    assert recommend.detect_tier() == "gpu_high"
    monkeypatch.setattr(recommend, "total_vram_bytes", lambda index=0: 16 * 1024 ** 3)
    assert recommend.detect_tier() == "gpu_high"


def test_real_16gb_cards_clear_the_16gb_bar(monkeypatch):
    # NVML reports what the driver leaves addressable, never the number on the
    # box, so no shipping 16 GB card ever reads a round 16 GiB. Comparing
    # against the nominal figure put every one of them below its own bar and
    # defaulted an RTX 4080 to the CPU. Readings are real nvidia-smi totals.
    monkeypatch.setattr(recommend, "can_run_cuda", lambda: True)
    for mib in (16376, 16380, 16303):
        monkeypatch.setattr(
            recommend, "total_vram_bytes", lambda index=0, m=mib: m * 1024 ** 2
        )
        assert recommend.detect_tier() == "gpu_high", mib
        assert recommend.default_device_choice() == "gpu", mib
    # The slack must not reach down to the next card class.
    monkeypatch.setattr(
        recommend, "total_vram_bytes", lambda index=0: 12281 * 1024 ** 2
    )
    assert recommend.detect_tier() == "gpu_low"


def test_the_two_16gb_bars_cannot_drift_apart():
    # A card that is sized gpu_high but defaulted to the CPU would download
    # models the wizard then refuses to run on the card they were picked for.
    assert recommend._GPU_DEFAULT_VRAM_BYTES == recommend._VRAM_HIGH_BYTES


def test_detect_tier_gpu_low_when_vram_small_or_unknown(monkeypatch):
    monkeypatch.setattr(recommend, "can_run_cuda", lambda: True)
    # A card below VRChat's recommended VRAM is already rationing for the game.
    for gb in (4, 8, 12):
        monkeypatch.setattr(recommend, "total_vram_bytes", lambda index=0, gb=gb: gb * 1024 ** 3)
        assert recommend.detect_tier() == "gpu_low", gb
    monkeypatch.setattr(recommend, "total_vram_bytes", lambda index=0: None)
    assert recommend.detect_tier() == "gpu_low"


def test_default_device_choice_gpu_at_16gb(monkeypatch):
    monkeypatch.setattr(recommend, "can_run_cuda", lambda: True)
    monkeypatch.setattr(recommend, "total_vram_bytes", lambda index=0: 24 * 1024 ** 3)
    assert recommend.default_device_choice() == "gpu"
    monkeypatch.setattr(recommend, "total_vram_bytes", lambda index=0: 16 * 1024 ** 3)
    assert recommend.default_device_choice() == "gpu"
    monkeypatch.setattr(recommend, "total_vram_bytes", lambda index=0: 8 * 1024 ** 3)
    assert recommend.default_device_choice() == "cpu"
    monkeypatch.setattr(recommend, "total_vram_bytes", lambda index=0: None)
    assert recommend.default_device_choice() == "cpu"


def test_default_device_choice_cpu_when_cuda_unusable(monkeypatch):
    # VRAM alone must not default the wizard to GPU: NVML reads it from the
    # display driver, which says nothing about whether this install can
    # drive the card.
    monkeypatch.setattr(recommend, "can_run_cuda", lambda: False)
    monkeypatch.setattr(recommend, "total_vram_bytes", lambda index=0: 24 * 1024 ** 3)
    assert recommend.default_device_choice() == "cpu"


def _card(monkeypatch, gb, cc):
    """A usable CUDA card of ``gb`` (as NVML reports it, just under nominal)."""
    monkeypatch.setattr(recommend, "can_run_cuda", lambda: True)
    monkeypatch.setattr(
        recommend, "total_vram_bytes", lambda index=0: int(gb * 1024**3 * 0.995)
    )
    monkeypatch.setattr(recommend, "compute_capability", lambda index=0: cc)


def test_an_old_large_card_is_not_high_tier(monkeypatch):
    # Capacity is not speed. A Tesla P100/P40 clears 16 GB easily but predates
    # tensor cores, and best_compute_type still hands it int8_float16.
    _card(monkeypatch, 16, (6, 0))
    assert recommend.detect_tier() == "gpu_low"
    _card(monkeypatch, 24, (6, 1))
    assert recommend.detect_tier() == "gpu_low"


def test_volta_is_the_floor(monkeypatch):
    _card(monkeypatch, 16, (7, 0))
    assert recommend.detect_tier() == "gpu_high"


def test_a_modern_card_still_needs_the_vram(monkeypatch):
    # The capability floor is additional to the VRAM bar, not a substitute.
    _card(monkeypatch, 11, (7, 5))
    assert recommend.detect_tier() == "gpu_low"


def test_an_unreadable_capability_does_not_demote(monkeypatch):
    # No pynvml is not evidence of an old card, and treating it as one would
    # demote every install without pynvml.
    _card(monkeypatch, 24, None)
    assert recommend.detect_tier() == "gpu_high"


def test_every_gpu_reader_sizes_against_the_configured_card(monkeypatch):
    """detect_tier, default_device_choice and detected_vram_mb must all judge
    the card stt.device_index names. Sizing the tier from card 0 while budgeting
    VRAM from card 1 is how a mixed multi-GPU box gets an incoherent answer."""
    from types import SimpleNamespace

    queried: list[int] = []
    monkeypatch.setattr(recommend, "can_run_cuda", lambda: True)
    monkeypatch.setattr(recommend, "compute_capability", lambda index=0: (8, 6))
    monkeypatch.setattr(
        recommend, "total_vram_bytes",
        lambda index=0: (queried.append(index), 24 * 1024**3)[1],
    )

    cfg = SimpleNamespace(stt=SimpleNamespace(device="auto", device_index=1))
    recommend.tier_for_config(cfg)
    recommend.default_device_choice(1)
    recommend.detected_vram_mb(1)

    assert queried == [1, 1, 1], queried
