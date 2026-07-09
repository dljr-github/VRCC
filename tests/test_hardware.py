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
    def test_auto_device_prefers_cuda_when_available(self, monkeypatch):
        monkeypatch.setattr(hardware, "cuda_device_count", lambda: 1)
        monkeypatch.setattr(hardware, "compute_capability", lambda index: None)
        monkeypatch.setattr(
            ctranslate2, "get_supported_compute_types", lambda device, index: FULL_SUPPORT
        )

        device, index, compute = hardware.resolve("auto", 0, "auto")

        assert device == "cuda"
        assert compute == "int8_float16"

    def test_auto_device_falls_back_to_cpu_when_no_cuda(self, monkeypatch):
        monkeypatch.setattr(hardware, "cuda_device_count", lambda: 0)
        monkeypatch.setattr(
            ctranslate2, "get_supported_compute_types", lambda device, index: CPU_TYPICAL
        )

        device, index, compute = hardware.resolve("auto", 0, "auto")

        assert device == "cpu"
        assert compute == "int8"

    def test_explicit_device_and_compute_pass_through(self, monkeypatch):
        monkeypatch.setattr(hardware, "cuda_device_count", lambda: 0)

        device, index, compute = hardware.resolve("cpu", 2, "float32")

        assert (device, index, compute) == ("cpu", 2, "float32")

    def test_device_index_is_preserved(self, monkeypatch):
        monkeypatch.setattr(hardware, "cuda_device_count", lambda: 1)

        _, index, _ = hardware.resolve("cuda", 3, "int8")

        assert index == 3


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
