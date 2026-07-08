from pathlib import Path

import pytest

from vrcc.gui import model_fit
from vrcc.core import hardware


def test_vram_warning_none_without_gpu(monkeypatch):
    monkeypatch.setattr(hardware, "total_vram_bytes", lambda index=0: None)
    assert model_fit.vram_warning(1600) is None


def test_vram_warning_none_when_cpu_device(monkeypatch):
    # Even with a GPU present, a model explicitly set to run on the processor
    # gets no graphics-card warning.
    monkeypatch.setattr(hardware, "total_vram_bytes", lambda index=0: 2 * 1024**3)
    assert model_fit.vram_warning(9000, device="cpu") is None


def test_vram_warning_fires_when_model_too_big(monkeypatch):
    monkeypatch.setattr(hardware, "total_vram_bytes", lambda index=0: 4 * 1024**3)
    msg = model_fit.vram_warning(6000, device="cuda")  # 6 GB model, 4 GB card
    assert msg is not None
    assert "graphics card" in msg.lower()
    assert "vram" not in msg.lower() and "gpu" not in msg.lower()


def test_vram_warning_silent_when_it_fits(monkeypatch):
    monkeypatch.setattr(hardware, "total_vram_bytes", lambda index=0: 16 * 1024**3)
    assert model_fit.vram_warning(1600, device="auto") is None


def test_disk_warning_none_when_dir_is_none():
    assert model_fit.disk_warning(None, 1600) is None


def test_disk_warning_fires_when_space_low(monkeypatch, tmp_path):
    import shutil
    from collections import namedtuple
    U = namedtuple("U", "total used free")
    monkeypatch.setattr(shutil, "disk_usage", lambda p: U(0, 0, 50 * 1024**2))
    msg = model_fit.disk_warning(tmp_path, 1600)  # need ~1.6 GB, 50 MB free
    assert msg is not None
    assert "disk" in msg.lower()


def test_disk_warning_silent_with_room(monkeypatch, tmp_path):
    import shutil
    from collections import namedtuple
    U = namedtuple("U", "total used free")
    monkeypatch.setattr(shutil, "disk_usage", lambda p: U(0, 0, 500 * 1024**3))
    assert model_fit.disk_warning(tmp_path, 1600) is None
