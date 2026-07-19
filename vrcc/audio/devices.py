"""Input-device enumeration and default-device lookup via `sounddevice`.

Windows exposes the same mic once per host API; `list_input_devices` dedupes by
name, preferring the WASAPI index. Both functions are fail-open: a `sounddevice`
failure logs at debug and returns empty/`None` rather than raising. Zero Qt.
"""

from __future__ import annotations

import logging

import sounddevice as sd

logger = logging.getLogger("vrcc.audio")


def list_input_devices() -> list[tuple[int, str]]:
    """Input-capable devices as `(index, name)`, deduped by name.

    Only `max_input_channels > 0`; for duplicate names (same mic via several
    host APIs) the WASAPI entry wins, else the first reported. Order follows
    first appearance. Never raises (failure -> empty list).
    """
    try:
        raw_devices = list(sd.query_devices())
        hostapis = list(sd.query_hostapis())
    except Exception:
        logger.debug("failed to query audio input devices", exc_info=True)
        return []

    try:
        names_in_order: list[str] = []
        candidates_by_name: dict[str, list[tuple[int, str]]] = {}

        for position, dev in enumerate(raw_devices):
            if dev.get("max_input_channels", 0) <= 0:
                continue
            index = dev.get("index", position)
            name = dev.get("name", f"Device {index}")
            hostapi_index = dev.get("hostapi")
            try:
                hostapi_name = hostapis[hostapi_index]["name"]
            except (TypeError, IndexError, KeyError):
                hostapi_name = ""

            if name not in candidates_by_name:
                candidates_by_name[name] = []
                names_in_order.append(name)
            candidates_by_name[name].append((index, hostapi_name))

        result: list[tuple[int, str]] = []
        for name in names_in_order:
            candidates = candidates_by_name[name]
            chosen_index = candidates[0][0]
            for index, hostapi_name in candidates:
                if "WASAPI" in hostapi_name:
                    chosen_index = index
                    break
            result.append((chosen_index, name))
        return result
    except Exception:
        logger.debug("failed to process audio input device list", exc_info=True)
        return []


def reinitialize_audio() -> None:
    """Cycle PortAudio so a device hotplugged after launch is visible.

    PortAudio snapshots the device list at initialization; a plain
    query_devices() will not see a mic plugged in later. Terminating and
    re-initializing rebuilds the list. Call ONLY when no stream is open (an
    open stream's handle is invalidated by _terminate). Fail-open: a failure
    logs and leaves the existing host in place."""
    try:
        sd._terminate()
        sd._initialize()
    except Exception:
        logger.debug("PortAudio re-initialization failed", exc_info=True)


def default_input_device() -> int | None:
    """The system default input device index, or `None` if there isn't one.

    Reads `sounddevice.default.device` (an `[input, output]` pair, or a bare
    int). Returns `None` if the input slot is negative (PortAudio's "no
    default"), missing, or unreadable.
    """
    try:
        device = sd.default.device
        try:
            index = device[0]
        except (TypeError, IndexError, KeyError):
            index = device  # a bare scalar applying to both directions

        if index is None or index < 0:
            return None
        return int(index)
    except Exception:
        logger.debug("failed to read default input device", exc_info=True)
        return None
