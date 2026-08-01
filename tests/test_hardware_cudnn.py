"""Tests for `vrcc.core.hardware`'s cuDNN sublibrary preload:
`_load_cudnn_sublibraries` and its call site in
`_preload_onnxruntime_cuda_dlls`. Split out from test_hardware.py to keep
both files under the 500-line cap.
"""

from pathlib import Path
from types import SimpleNamespace

from vrcc.core import hardware


class TestLoadCudnnSublibraries:
    """preload_dlls() omits cudnn_engines_tensor_ir64_9.dll (and a couple of
    other cuDNN sublibraries); cuDNN loads that one lazily by bare name, and
    the lazy load ignores os.add_dll_directory. _load_cudnn_sublibraries()
    forces every cudnn*.dll in the wheel resident so the lazy load finds it."""

    def _patch_find_spec(self, monkeypatch, base_dir):
        monkeypatch.setattr(
            hardware.importlib.util,
            "find_spec",
            lambda name: SimpleNamespace(submodule_search_locations=[str(base_dir)]),
        )

    def test_loads_every_cudnn_dll_from_the_wheel_bin_dir(self, monkeypatch, tmp_path):
        bin_dir = tmp_path / "nvidia" / "cudnn" / "bin"
        bin_dir.mkdir(parents=True)
        (bin_dir / "cudnn64_9.dll").touch()
        (bin_dir / "cudnn_engines_tensor_ir64_9.dll").touch()
        (bin_dir / "other.dll").touch()
        self._patch_find_spec(monkeypatch, tmp_path / "nvidia")
        # WinDLL exists only on Windows; drive the win32 branch on any CI OS.
        monkeypatch.setattr(hardware.sys, "platform", "win32")

        calls = []
        monkeypatch.setattr(
            hardware.ctypes, "WinDLL", lambda path: calls.append(path), raising=False
        )

        hardware._load_cudnn_sublibraries()

        assert sorted(Path(p).name for p in calls) == [
            "cudnn64_9.dll",
            "cudnn_engines_tensor_ir64_9.dll",
        ]

    def test_a_windll_oserror_on_one_dll_is_swallowed_and_the_rest_still_load(
        self, monkeypatch, tmp_path
    ):
        bin_dir = tmp_path / "nvidia" / "cudnn" / "bin"
        bin_dir.mkdir(parents=True)
        (bin_dir / "cudnn64_9.dll").touch()
        (bin_dir / "cudnn_engines_tensor_ir64_9.dll").touch()
        self._patch_find_spec(monkeypatch, tmp_path / "nvidia")
        monkeypatch.setattr(hardware.sys, "platform", "win32")

        calls = []

        def fake_windll(path):
            calls.append(path)
            if Path(path).name == "cudnn64_9.dll":
                raise OSError("could not locate cudnn64_9.dll")
            return object()

        monkeypatch.setattr(hardware.ctypes, "WinDLL", fake_windll, raising=False)

        hardware._load_cudnn_sublibraries()  # must not raise

        assert len(calls) == 2

    def test_no_nvidia_package_is_a_noop(self, monkeypatch):
        monkeypatch.setattr(hardware.sys, "platform", "win32")
        monkeypatch.setattr(hardware.importlib.util, "find_spec", lambda name: None)
        calls = []
        monkeypatch.setattr(
            hardware.ctypes, "WinDLL", lambda path: calls.append(path), raising=False
        )

        hardware._load_cudnn_sublibraries()  # must not raise

        assert calls == []

    def test_non_windows_is_a_noop(self, monkeypatch, tmp_path):
        bin_dir = tmp_path / "nvidia" / "cudnn" / "bin"
        bin_dir.mkdir(parents=True)
        (bin_dir / "cudnn64_9.dll").touch()
        self._patch_find_spec(monkeypatch, tmp_path / "nvidia")
        monkeypatch.setattr(hardware.sys, "platform", "linux")

        calls = []
        monkeypatch.setattr(
            hardware.ctypes, "WinDLL", lambda path: calls.append(path), raising=False
        )

        hardware._load_cudnn_sublibraries()

        assert calls == []


class TestPreloadOnnxruntimeCudnnFallthrough:
    def test_preload_onnxruntime_still_loads_cudnn_without_preload_dlls(
        self, monkeypatch
    ):
        # The cuDNN sublibrary preload must run even on an onnxruntime build
        # that predates preload_dlls (older or CPU-only builds), not just
        # not-raise: this is the branch that regressed by returning early.
        import onnxruntime

        monkeypatch.delattr(onnxruntime, "preload_dlls", raising=False)
        calls = []
        monkeypatch.setattr(
            hardware, "_load_cudnn_sublibraries", lambda: calls.append(True)
        )
        hardware._preload_onnxruntime_cuda_dlls()
        assert calls == [True]
