"""Input-device enumeration and default-device lookup via `sounddevice`.

Windows exposes the same mic once per host API; `list_input_devices` restricts
the result to WASAPI, falling back to MME for a mic that has no WASAPI entry,
and drops a mic that is visible only under DirectSound or WDM-KS. If that
filter would leave the list empty while input-capable devices do exist, it
falls back to the pre-existing WASAPI-preferred-else-first selection so the
picker never shows zero mics. Both functions are fail-open: a `sounddevice`
failure logs at debug and returns empty/`None` rather than raising. Zero Qt.
"""

from __future__ import annotations

import logging

import sounddevice as sd

logger = logging.getLogger("vrcc.audio")


def list_input_devices() -> list[tuple[int, str]]:
    """Input-capable devices as `(index, name)`, deduped by name.

    Only `max_input_channels > 0`. For duplicate names (same mic via several
    host APIs) the WASAPI entry wins; a mic with no WASAPI entry falls back
    to its MME entry; a mic visible only via DirectSound or WDM-KS is
    dropped. If that leaves the list empty while input-capable devices are
    present, falls back to the WASAPI-preferred-else-first selection so the
    picker never shows zero mics. Order follows first appearance. Never
    raises (failure -> empty list).
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
            chosen_index = _preferred_index(candidates_by_name[name])
            if chosen_index is not None:
                result.append((chosen_index, name))

        if not result and names_in_order:
            logger.debug(
                "no WASAPI/MME microphone found; safety net falling back to "
                "the WASAPI-preferred-else-first device list"
            )
            for name in names_in_order:
                result.append((_first_or_wasapi_index(candidates_by_name[name]), name))

        return result
    except Exception:
        logger.debug("failed to process audio input device list", exc_info=True)
        return []


def _preferred_index(candidates: list[tuple[int, str]]) -> int | None:
    """WASAPI index if one exists, else MME, else `None` (caller drops it)."""
    mme_index = None
    for index, hostapi_name in candidates:
        if "WASAPI" in hostapi_name.upper():
            return index
        if mme_index is None and "MME" in hostapi_name.upper():
            mme_index = index
    return mme_index


def _first_or_wasapi_index(candidates: list[tuple[int, str]]) -> int:
    """WASAPI index if any, else the first seen."""
    for index, hostapi_name in candidates:
        if "WASAPI" in hostapi_name.upper():
            return index
    return candidates[0][0]


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
