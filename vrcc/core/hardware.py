"""GPU/CPU hardware detection and CTranslate2 compute-type selection.

Never imports torch. CUDA discovery via `ctranslate2`; richer details via
optional `pynvml` (functions degrade rather than raise when it's absent).
`setup_cuda_dlls()` must run before the first CT2 GPU call. Zero Qt.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import logging
import os
import sys
import threading
from pathlib import Path

import ctranslate2

from vrcc.core.bus import EventBus
from vrcc.core.events import AppError

logger = logging.getLogger("vrcc.hardware")

# Preference ladder for CTranslate2 compute types, most-preferred first.
# best_compute_type() returns the first entry present in the supported set.
_COMPUTE_TYPE_LADDER = (
    "int8_float16",
    "int8_bfloat16",
    "int8",
    "float16",
    "bfloat16",
    "int8_float32",
    "float32",
)

# Minimum supported NVIDIA driver major version.
_DRIVER_FLOOR_MAJOR = 570

# Set by check_driver_floor() when a CUDA device's driver is too old; resolve()
# then downgrades every device to "cpu" until it passes. Must be called before
# resolve() for the downgrade to apply.
_driver_floor_failed = False

_nvml_lock = threading.Lock()
_nvml_module = None
_nvml_init_attempted = False


def _pynvml():
    """Return the initialized `pynvml` module, or `None` if it isn't
    importable or NVML initialization fails. Cached after the first call
    (successful or not) so we don't retry `nvmlInit()` on every call."""
    global _nvml_module, _nvml_init_attempted
    with _nvml_lock:
        if _nvml_init_attempted:
            return _nvml_module
        _nvml_init_attempted = True
        try:
            import pynvml

            pynvml.nvmlInit()
        except Exception:
            logger.debug(
                "pynvml unavailable; GPU name/compute-capability/driver "
                "details will be limited",
                exc_info=True,
            )
            _nvml_module = None
        else:
            _nvml_module = pynvml
        return _nvml_module


def setup_cuda_dlls() -> bool:
    """On Windows, add each installed `nvidia-*` wheel's `bin` (DLL) dir to the
    process DLL search path so CTranslate2 finds the CUDA runtime without a
    system-wide install, then have onnxruntime preload the CUDA/cuDNN DLLs it
    needs (its execution provider resolves them at session-build time, after
    this). Returns whether any dir was added; no-op (False) off Windows or
    without the `nvidia` package. Never raises.
    """
    if sys.platform != "win32":
        return False

    added = False
    try:
        added = _add_nvidia_dll_dirs()
    except Exception:
        logger.debug(
            "setup_cuda_dlls failed; continuing without added DLL dirs",
            exc_info=True,
        )
    _preload_onnxruntime_cuda_dlls()
    return added


def _preload_onnxruntime_cuda_dlls() -> None:
    """Best-effort ``onnxruntime.preload_dlls()`` (ORT >= 1.21): loads the
    CUDA/cuDNN DLLs from the installed nvidia-* wheels into the process so the
    CUDA execution provider (Parakeet/Canary on GPU) can build sessions in the
    packaged app. A no-op on older or CPU-only onnxruntime builds. Anything
    the preload prints is captured and demoted to a debug log entry."""
    try:
        import onnxruntime

        preload = getattr(onnxruntime, "preload_dlls", None)
        if preload is None:
            return
        # preload_dlls() prints "Failed to load ..." per CUDA DLL the wheels
        # don't ship; VRCC bundles only cuBLAS + cuDNN, so those misses are
        # expected and belong in the debug log, not on the console. Fresh
        # StringIO buffers (rather than wrapping the live streams) also keep
        # the preload's writes safe in the windowed exe, where sys.stdout and
        # sys.stderr are None.
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            preload()
        captured = (out.getvalue() + err.getvalue()).strip()
        if captured:
            logger.debug("onnxruntime.preload_dlls output:\n%s", captured)
    except Exception:
        logger.debug("onnxruntime.preload_dlls failed; continuing", exc_info=True)


def _add_nvidia_dll_dirs() -> bool:
    spec = importlib.util.find_spec("nvidia")
    if spec is None or not spec.submodule_search_locations:
        return False

    added = False
    for base in spec.submodule_search_locations:
        base_path = Path(base)
        if not base_path.is_dir():
            continue
        for pkg_dir in sorted(base_path.iterdir()):
            bin_dir = pkg_dir / "bin"
            if not bin_dir.is_dir() or not any(bin_dir.glob("*.dll")):
                continue
            os.add_dll_directory(str(bin_dir))
            added = True
    return added


def cuda_device_count() -> int:
    """Number of CUDA devices CTranslate2 can see. Returns 0 (instead of
    raising) if CTranslate2 has no CUDA support built in, or any other
    error occurs while asking."""
    try:
        return ctranslate2.get_cuda_device_count()
    except Exception:
        return 0


def device_names() -> list[str]:
    """Human-readable GPU names for every visible CUDA device, via
    `pynvml` if it's importable and NVML initializes; otherwise a generic
    `"CUDA device {i}"` placeholder per device."""
    count = cuda_device_count()
    pynvml = _pynvml()
    if pynvml is not None:
        try:
            names = []
            for i in range(count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                name = pynvml.nvmlDeviceGetName(handle)
                if isinstance(name, bytes):
                    name = name.decode("utf-8", "replace")
                names.append(name)
            return names
        except Exception:
            logger.debug("pynvml device name lookup failed", exc_info=True)

    return [f"CUDA device {i}" for i in range(count)]


def compute_capability(index: int) -> tuple[int, int] | None:
    """CUDA compute capability (major, minor) of device `index`, via
    `pynvml`. `None` if pynvml is unavailable or the lookup fails."""
    pynvml = _pynvml()
    if pynvml is None:
        return None
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(index)
        major, minor = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
        return (major, minor)
    except Exception:
        logger.debug("pynvml compute-capability lookup failed", exc_info=True)
        return None


def driver_version() -> str | None:
    """Installed NVIDIA driver version string, via `pynvml`. `None` if
    pynvml is unavailable or the lookup fails."""
    pynvml = _pynvml()
    if pynvml is None:
        return None
    try:
        version = pynvml.nvmlSystemGetDriverVersion()
        if isinstance(version, bytes):
            version = version.decode("utf-8", "replace")
        return version
    except Exception:
        logger.debug("pynvml driver-version lookup failed", exc_info=True)
        return None


def total_vram_bytes(index: int = 0) -> int | None:
    """Total VRAM (bytes) of CUDA device ``index`` via ``pynvml``. ``None`` if
    pynvml is unavailable or the lookup fails. Never raises."""
    pynvml = _pynvml()
    if pynvml is None:
        return None
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(index)
        return int(pynvml.nvmlDeviceGetMemoryInfo(handle).total)
    except Exception:  # noqa: BLE001
        logger.debug("pynvml VRAM lookup failed", exc_info=True)
        return None


def best_compute_type(
    device: str,
    index: int,
    supported: set[str] | None = None,
    cc: tuple[int, int] | None = None,
) -> str:
    """Pick the best CTranslate2 compute type for `device`/`index`: the first
    entry of the int8_float16 > ... > float32 ladder present in `supported`
    (defaults to CT2's supported set). The sm120 rule drops all `int8*` types on
    cc major >= 12 (no fast int8 kernels there). Falls back to `"float32"`.
    """
    if supported is None:
        try:
            supported = ctranslate2.get_supported_compute_types(device, index)
        except Exception:
            supported = set()
    else:
        supported = set(supported)

    if cc is not None and cc[0] >= 12:
        supported = {ct for ct in supported if not ct.startswith("int8")}

    for candidate in _COMPUTE_TYPE_LADDER:
        if candidate in supported:
            return candidate
    return "float32"


def resolve(
    device_cfg: str, device_index: int, compute_cfg: str
) -> tuple[str, int, str]:
    """Resolve config device/compute settings into concrete values.

    `"auto"` device -> `"cuda"` if a device is visible else `"cpu"`; a failed
    driver-floor check downgrades any `"cuda"` (auto or pinned) to `"cpu"`.
    `"auto"` compute -> `best_compute_type()`; other values and `device_index`
    pass through unchanged.
    """
    if device_cfg == "auto":
        device = "cuda" if cuda_device_count() > 0 else "cpu"
    else:
        device = device_cfg

    if device == "cuda" and _driver_floor_failed:
        device = "cpu"

    if compute_cfg == "auto":
        cc = compute_capability(device_index) if device == "cuda" else None
        compute = best_compute_type(device, device_index, cc=cc)
    else:
        compute = compute_cfg

    return device, device_index, compute


def check_driver_floor(bus: EventBus) -> bool:
    """Check the installed NVIDIA driver against the floor (major >= 570) when a
    CUDA device is present, publishing `AppError("DRIVER_TOO_OLD")` if too old.
    Returns True when it passes or doesn't apply (no device / unknown version);
    False (setting the module flag so `resolve()` downgrades to CPU) when too
    old. Must run before `resolve()`.
    """
    global _driver_floor_failed

    if cuda_device_count() <= 0:
        _driver_floor_failed = False
        return True

    version = driver_version()
    if version is None:
        _driver_floor_failed = False
        return True

    try:
        major = int(version.split(".")[0])
    except (ValueError, IndexError):
        _driver_floor_failed = False
        return True

    if major < _DRIVER_FLOOR_MAJOR:
        _driver_floor_failed = True
        bus.publish(
            AppError(
                code="DRIVER_TOO_OLD",
                message=(
                    f"NVIDIA driver {version} is older than the minimum "
                    f"supported version ({_DRIVER_FLOOR_MAJOR}.x); falling "
                    "back to CPU"
                ),
                detail=f"cuda device present; driver major={major}",
            )
        )
        return False

    _driver_floor_failed = False
    return True
