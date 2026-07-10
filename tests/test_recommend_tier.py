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
    monkeypatch.setattr(recommend, "total_vram_bytes", lambda: 24 * 1024 ** 3)
    assert recommend.detect_tier() == "cpu"


def test_detect_tier_gpu_high_when_vram_ample(monkeypatch):
    monkeypatch.setattr(recommend, "can_run_cuda", lambda: True)
    monkeypatch.setattr(recommend, "total_vram_bytes", lambda: 12 * 1024 ** 3)
    assert recommend.detect_tier() == "gpu_high"


def test_detect_tier_gpu_low_when_vram_small_or_unknown(monkeypatch):
    monkeypatch.setattr(recommend, "can_run_cuda", lambda: True)
    monkeypatch.setattr(recommend, "total_vram_bytes", lambda: 4 * 1024 ** 3)
    assert recommend.detect_tier() == "gpu_low"
    monkeypatch.setattr(recommend, "total_vram_bytes", lambda: None)
    assert recommend.detect_tier() == "gpu_low"


def test_default_device_choice_gpu_at_16gb(monkeypatch):
    monkeypatch.setattr(recommend, "can_run_cuda", lambda: True)
    monkeypatch.setattr(recommend, "total_vram_bytes", lambda: 24 * 1024**3)
    assert recommend.default_device_choice() == "gpu"
    monkeypatch.setattr(recommend, "total_vram_bytes", lambda: 16 * 1024**3)
    assert recommend.default_device_choice() == "gpu"
    monkeypatch.setattr(recommend, "total_vram_bytes", lambda: 8 * 1024**3)
    assert recommend.default_device_choice() == "cpu"
    monkeypatch.setattr(recommend, "total_vram_bytes", lambda: None)
    assert recommend.default_device_choice() == "cpu"


def test_default_device_choice_cpu_when_cuda_unusable(monkeypatch):
    # VRAM alone must not default the wizard to GPU: NVML reads it from the
    # display driver, which says nothing about whether this install can
    # drive the card.
    monkeypatch.setattr(recommend, "can_run_cuda", lambda: False)
    monkeypatch.setattr(recommend, "total_vram_bytes", lambda: 24 * 1024**3)
    assert recommend.default_device_choice() == "cpu"
