import logging
import sys

import ctranslate2
import pytest

from vrcc.core import hardware
from vrcc.core.bus import EventBus
from vrcc.core.events import AppError

FULL_SUPPORT = {
    "int8_float16",
    "int8_bfloat16",
    "int8",
    "float16",
    "bfloat16",
    "int8_float32",
    "float32",
}
CPU_TYPICAL = {"int8", "float32", "int8_float32"}


@pytest.fixture(autouse=True)
def _reset_driver_floor_flag(monkeypatch):
    # Each test starts from a clean "floor not failed" state regardless of
    # what an earlier test in the module did to the module-level flag.
    monkeypatch.setattr(hardware, "_driver_floor_failed", False)


class TestBestComputeType:
    def test_full_support_prefers_int8_float16(self):
        assert (
            hardware.best_compute_type("cuda", 0, supported=FULL_SUPPORT)
            == "int8_float16"
        )

    def test_cpu_typical_prefers_int8(self):
        assert hardware.best_compute_type("cpu", 0, supported=CPU_TYPICAL) == "int8"

    def test_sm120_full_support_drops_int8_variants(self):
        # sm120 (Blackwell+) rule: cc major >= 12 drops all int8* entries
        # first, so the ladder's next candidate (float16) wins even though
        # int8_float16 is technically "supported".
        result = hardware.best_compute_type(
            "cuda", 0, supported=FULL_SUPPORT, cc=(12, 0)
        )
        assert result == "float16"

    def test_empty_support_falls_back_to_float32(self):
        assert hardware.best_compute_type("cuda", 0, supported=set()) == "float32"

    def test_sm120_rule_does_not_apply_below_major_12(self):
        result = hardware.best_compute_type(
            "cuda", 0, supported=FULL_SUPPORT, cc=(8, 9)
        )
        assert result == "int8_float16"

    def test_default_supported_comes_from_ctranslate2(self, monkeypatch):
        monkeypatch.setattr(
            ctranslate2, "get_supported_compute_types", lambda device, index: CPU_TYPICAL
        )
        assert hardware.best_compute_type("cpu", 0) == "int8"


class TestResolve:
    def test_auto_device_prefers_cuda_when_usable(self, monkeypatch):
        monkeypatch.setattr(hardware, "can_run_cuda", lambda: True)
        monkeypatch.setattr(hardware, "compute_capability", lambda index: None)
        monkeypatch.setattr(
            ctranslate2, "get_supported_compute_types", lambda device, index: FULL_SUPPORT
        )

        device, index, compute = hardware.resolve("auto", 0, "auto")

        assert device == "cuda"
        assert compute == "int8_float16"

    def test_auto_device_falls_back_to_cpu_when_no_cuda(self, monkeypatch):
        monkeypatch.setattr(hardware, "can_run_cuda", lambda: False)
        monkeypatch.setattr(
            ctranslate2, "get_supported_compute_types", lambda device, index: CPU_TYPICAL
        )

        device, index, compute = hardware.resolve("auto", 0, "auto")

        assert device == "cpu"
        assert compute == "int8"

    def test_auto_device_is_cpu_when_device_visible_but_cuda_unusable(self, monkeypatch):
        monkeypatch.setattr(hardware, "cuda_device_count", lambda: 1)
        monkeypatch.setattr(hardware, "can_run_cuda", lambda: False)
        monkeypatch.setattr(
            ctranslate2, "get_supported_compute_types", lambda device, index: CPU_TYPICAL
        )

        device, index, compute = hardware.resolve("auto", 0, "auto")

        assert device == "cpu"
        assert compute == "int8"

    def test_explicit_cuda_is_not_gated_on_capability(self, monkeypatch):
        monkeypatch.setattr(hardware, "can_run_cuda", lambda: False)

        device, index, compute = hardware.resolve("cuda", 0, "float16")

        assert (device, index, compute) == ("cuda", 0, "float16")

    def test_explicit_device_and_compute_pass_through(self, monkeypatch):
        monkeypatch.setattr(hardware, "cuda_device_count", lambda: 0)

        device, index, compute = hardware.resolve("cpu", 2, "float32")

        assert (device, index, compute) == ("cpu", 2, "float32")

    def test_device_index_is_preserved(self, monkeypatch):
        monkeypatch.setattr(hardware, "cuda_device_count", lambda: 1)

        _, index, _ = hardware.resolve("cuda", 3, "int8")

        assert index == 3


class TestResolvedDevice:
    _PARAKEET = "parakeet-tdt-0.6b-v3"

    def test_auto_gpu_whisper_is_cuda(self, monkeypatch):
        monkeypatch.setattr(hardware, "can_run_cuda", lambda: True)
        assert hardware.resolved_device("auto", 0, "small") == "cuda"

    def test_auto_gpu_parakeet_is_cpu(self, monkeypatch):
        # onnx-asr auto override: a GPU is usable but the int8 graph runs on CPU.
        monkeypatch.setattr(hardware, "can_run_cuda", lambda: True)
        assert hardware.resolved_device("auto", 0, self._PARAKEET) == "cpu"

    def test_explicit_cuda_parakeet_stays_cuda(self, monkeypatch):
        # The override only applies to "auto"; a pinned "cuda" is honored.
        monkeypatch.setattr(hardware, "can_run_cuda", lambda: True)
        assert hardware.resolved_device("cuda", 0, self._PARAKEET) == "cuda"

    def test_explicit_cpu_stays_cpu(self, monkeypatch):
        monkeypatch.setattr(hardware, "can_run_cuda", lambda: True)
        assert hardware.resolved_device("cpu", 0, "small") == "cpu"

    def test_auto_without_gpu_is_cpu(self, monkeypatch):
        monkeypatch.setattr(hardware, "can_run_cuda", lambda: False)
        assert hardware.resolved_device("auto", 0, "small") == "cpu"

    def test_auto_visible_device_without_capability_is_cpu(self, monkeypatch):
        monkeypatch.setattr(hardware, "cuda_device_count", lambda: 1)
        monkeypatch.setattr(hardware, "can_run_cuda", lambda: False)
        assert hardware.resolved_device("auto", 0, "small") == "cpu"

    def test_explicit_cuda_without_capability_stays_cuda(self, monkeypatch):
        monkeypatch.setattr(hardware, "can_run_cuda", lambda: False)
        assert hardware.resolved_device("cuda", 0, "small") == "cuda"

    def test_auto_gpu_unknown_model_mirrors_resolve(self, monkeypatch):
        # No / unknown model id: no onnx override, so it matches resolve()'s cuda.
        monkeypatch.setattr(hardware, "can_run_cuda", lambda: True)
        assert hardware.resolved_device("auto", 0, None) == "cuda"

    def test_failed_driver_floor_degrades_auto_to_cpu(self, monkeypatch):
        monkeypatch.setattr(hardware, "can_run_cuda", lambda: True)
        monkeypatch.setattr(hardware, "_driver_floor_failed", True)
        assert hardware.resolved_device("auto", 0, "small") == "cpu"

    def test_failed_driver_floor_degrades_explicit_cuda_to_cpu(self, monkeypatch):
        monkeypatch.setattr(hardware, "can_run_cuda", lambda: True)
        monkeypatch.setattr(hardware, "_driver_floor_failed", True)
        assert hardware.resolved_device("cuda", 0, "small") == "cpu"


class TestCanRunCuda:
    @pytest.fixture(autouse=True)
    def _fresh_probe_cache(self, monkeypatch):
        # The probe result is cached per process; each test starts unprobed.
        monkeypatch.setattr(hardware, "_cublas_probe_attempted", False)
        monkeypatch.setattr(hardware, "_cublas_loadable", False)

    def test_no_device_is_false_without_probing(self, monkeypatch):
        # A raising probe would be swallowed by _cublas_available's except
        # clause, so record calls instead and require none happened.
        monkeypatch.setattr(hardware, "cuda_device_count", lambda: 0)
        probes = []

        def probe():
            probes.append(True)
            return True

        monkeypatch.setattr(hardware, "_probe_cublas", probe)
        assert hardware.can_run_cuda() is False
        assert probes == []

    def test_visible_device_with_unloadable_cublas_is_false(self, monkeypatch):
        monkeypatch.setattr(hardware, "cuda_device_count", lambda: 1)
        monkeypatch.setattr(hardware, "_probe_cublas", lambda: False)
        assert hardware.can_run_cuda() is False

    def test_visible_device_with_loadable_cublas_is_true(self, monkeypatch):
        monkeypatch.setattr(hardware, "cuda_device_count", lambda: 1)
        monkeypatch.setattr(hardware, "_probe_cublas", lambda: True)
        assert hardware.can_run_cuda() is True

    def test_probe_runs_once_and_the_result_is_cached(self, monkeypatch):
        monkeypatch.setattr(hardware, "cuda_device_count", lambda: 1)
        calls = []

        def probe():
            calls.append(True)
            return True

        monkeypatch.setattr(hardware, "_probe_cublas", probe)
        assert hardware.can_run_cuda() is True
        assert hardware.can_run_cuda() is True
        assert len(calls) == 1

    def test_probe_exception_reads_as_not_loadable(self, monkeypatch):
        monkeypatch.setattr(hardware, "cuda_device_count", lambda: 1)

        def boom():
            raise OSError("loader blew up")

        monkeypatch.setattr(hardware, "_probe_cublas", boom)
        assert hardware.can_run_cuda() is False

    def test_failed_probe_logs_once_naming_the_libraries(self, monkeypatch, caplog):
        monkeypatch.setattr(hardware, "cuda_device_count", lambda: 1)
        monkeypatch.setattr(hardware, "_probe_cublas", lambda: False)
        with caplog.at_level(logging.INFO, logger="vrcc.hardware"):
            hardware.can_run_cuda()
            hardware.can_run_cuda()
        infos = [r for r in caplog.records if r.levelno == logging.INFO]
        assert len(infos) == 1
        assert "cublas" in infos[0].getMessage().lower()


@pytest.mark.skipif(sys.platform != "win32", reason="Windows DLL search semantics")
class TestProbeCublasSearch:
    """The probe must look everywhere ctranslate2's own loader does: the
    secure default search (the `os.add_dll_directory` dirs), the legacy
    LoadLibrary search (PATH included), and the `%CUDA_PATH%\\bin`
    fallback."""

    def _fake_windll(self, monkeypatch, succeeds):
        calls = []

        def fake(name, winmode=None):
            calls.append((name, winmode))
            if succeeds(name, winmode):
                return object()
            raise OSError(f"cannot load {name}")

        monkeypatch.setattr(hardware.ctypes, "WinDLL", fake)
        return calls

    def test_secure_search_hit_needs_no_fallback(self, monkeypatch):
        calls = self._fake_windll(monkeypatch, lambda name, winmode: winmode is None)
        assert hardware._probe_cublas() is True
        assert calls == [(hardware._CUBLAS_NAMES[0], None)]

    def test_path_only_toolkit_loads_via_legacy_search(self, monkeypatch):
        # The CUDA toolkit installer publishes its bin dir via PATH, which
        # the secure search never consults; ctranslate2.dll resolves cuBLAS
        # with the legacy search, so the probe must count this as usable.
        calls = self._fake_windll(monkeypatch, lambda name, winmode: winmode == 0)
        assert hardware._probe_cublas() is True
        assert (hardware._CUBLAS_NAMES[0], 0) in calls

    def test_cuda_path_bin_is_the_last_resort(self, monkeypatch, tmp_path):
        monkeypatch.setenv("CUDA_PATH", str(tmp_path))
        expected = str(tmp_path / "bin" / hardware._CUBLAS_NAMES[0])
        calls = self._fake_windll(monkeypatch, lambda name, winmode: name == expected)
        assert hardware._probe_cublas() is True
        assert expected in [name for name, _ in calls]

    def test_unloadable_everywhere_is_false(self, monkeypatch):
        monkeypatch.delenv("CUDA_PATH", raising=False)
        calls = self._fake_windll(monkeypatch, lambda name, winmode: False)
        assert hardware._probe_cublas() is False
        assert len(calls) == 2 * len(hardware._CUBLAS_NAMES)


class TestDriverFloor:
    def test_no_cuda_device_never_publishes(self, monkeypatch):
        monkeypatch.setattr(hardware, "cuda_device_count", lambda: 0)
        monkeypatch.setattr(hardware, "driver_version", lambda: "500.10")
        bus = EventBus()
        errors = []
        bus.subscribe(AppError, errors.append)

        assert hardware.check_driver_floor(bus) is True
        assert errors == []

    def test_cuda_present_recent_driver_ok(self, monkeypatch):
        monkeypatch.setattr(hardware, "cuda_device_count", lambda: 1)
        monkeypatch.setattr(hardware, "driver_version", lambda: "570.20")
        bus = EventBus()
        errors = []
        bus.subscribe(AppError, errors.append)

        assert hardware.check_driver_floor(bus) is True
        assert errors == []

    def test_cuda_present_old_driver_publishes_app_error(self, monkeypatch):
        monkeypatch.setattr(hardware, "cuda_device_count", lambda: 1)
        monkeypatch.setattr(hardware, "driver_version", lambda: "550.40")
        bus = EventBus()
        errors = []
        bus.subscribe(AppError, errors.append)

        assert hardware.check_driver_floor(bus) is False
        assert len(errors) == 1
        assert errors[0].code == "DRIVER_TOO_OLD"

    def test_unknown_driver_version_does_not_publish(self, monkeypatch):
        monkeypatch.setattr(hardware, "cuda_device_count", lambda: 1)
        monkeypatch.setattr(hardware, "driver_version", lambda: None)
        bus = EventBus()
        errors = []
        bus.subscribe(AppError, errors.append)

        assert hardware.check_driver_floor(bus) is True
        assert errors == []

    def test_resolve_degrades_to_cpu_after_failed_floor_check(self, monkeypatch):
        monkeypatch.setattr(hardware, "cuda_device_count", lambda: 1)
        monkeypatch.setattr(hardware, "driver_version", lambda: "550.40")
        bus = EventBus()
        bus.subscribe(AppError, lambda e: None)
        assert hardware.check_driver_floor(bus) is False

        device, index, compute = hardware.resolve("auto", 0, "float32")

        assert device == "cpu"
        assert compute == "float32"


class TestGracefulDegradation:
    """pynvml is not installed by default; every public function must still
    behave (rather than raise) in that environment."""

    def test_cuda_device_count_never_raises_and_returns_int(self):
        count = hardware.cuda_device_count()
        assert isinstance(count, int)
        assert count >= 0

    def test_can_run_cuda_never_raises_and_returns_bool(self):
        assert isinstance(hardware.can_run_cuda(), bool)

    def test_setup_cuda_dlls_never_raises_and_returns_bool(self):
        assert isinstance(hardware.setup_cuda_dlls(), bool)

    def test_preload_onnxruntime_calls_preload_dlls_when_available(self, monkeypatch):
        import onnxruntime

        calls = []
        monkeypatch.setattr(
            onnxruntime, "preload_dlls", lambda: calls.append(True), raising=False
        )
        hardware._preload_onnxruntime_cuda_dlls()
        assert calls == [True]

    def test_preload_onnxruntime_swallows_preload_failure(self, monkeypatch):
        import onnxruntime

        def boom():
            raise OSError("cudnn64_9.dll not found")

        monkeypatch.setattr(onnxruntime, "preload_dlls", boom, raising=False)
        hardware._preload_onnxruntime_cuda_dlls()  # must not raise

    def test_preload_onnxruntime_silences_preload_chatter(self, monkeypatch, capsys):
        # preload_dlls() prints "Failed to load ..." lines for CUDA DLLs the
        # nvidia-* wheels don't ship; those must land in the debug log, never
        # on the real stdout/stderr (None in the windowed exe).
        import sys

        import onnxruntime

        def chatty():
            print("Failed to load cufft64_11.dll ...")
            print("Failed to load cudart64_12.dll ...", file=sys.stderr)

        monkeypatch.setattr(onnxruntime, "preload_dlls", chatty, raising=False)
        hardware._preload_onnxruntime_cuda_dlls()  # must not raise
        captured = capsys.readouterr()
        assert captured.out == ""
        assert captured.err == ""

    def test_preload_onnxruntime_noop_without_preload_dlls(self, monkeypatch):
        # Older onnxruntime (< 1.21) has no preload_dlls attribute.
        import onnxruntime

        monkeypatch.delattr(onnxruntime, "preload_dlls", raising=False)
        hardware._preload_onnxruntime_cuda_dlls()  # must not raise

    def test_device_names_fallback_without_pynvml(self, monkeypatch):
        monkeypatch.setattr(hardware, "_pynvml", lambda: None)
        monkeypatch.setattr(hardware, "cuda_device_count", lambda: 2)

        assert hardware.device_names() == ["CUDA device 0", "CUDA device 1"]

    def test_compute_capability_none_without_pynvml(self, monkeypatch):
        monkeypatch.setattr(hardware, "_pynvml", lambda: None)

        assert hardware.compute_capability(0) is None

    def test_driver_version_none_without_pynvml(self, monkeypatch):
        monkeypatch.setattr(hardware, "_pynvml", lambda: None)

        assert hardware.driver_version() is None

    def test_total_vram_bytes_none_without_pynvml(self, monkeypatch):
        monkeypatch.setattr(hardware, "_pynvml", lambda: None)
        assert hardware.total_vram_bytes() is None
