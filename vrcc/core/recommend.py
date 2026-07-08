"""Hardware-tier detection and quality-ordered model recommendations (Qt-free).

Tiers: ``gpu_high`` (CUDA >= 8 GB), ``gpu_low`` (< 8 GB / unknown), ``cpu``.
``PRESETS`` picks each tier's (whisper, mt); ``*_PREFERENCE`` are per-tier
best-first orderings for :func:`best_downloaded`.
"""

from __future__ import annotations

from vrcc.core.hardware import cuda_device_count, total_vram_bytes
from vrcc.stt.registry import WHISPER_MODELS
from vrcc.translate.registry import MT_MODELS

_VRAM_HIGH_BYTES = 8 * 1024 ** 3

# tier -> (whisper_id, mt_id)
PRESETS: dict[str, tuple[str, str]] = {
    "gpu_high": ("large-v3-turbo", "nllb-1.3B-int8"),
    "gpu_low": ("large-v3-turbo", "nllb-600M-int8"),
    "cpu": ("small", "nllb-600M-int8"),
}

# Per-tier whisper ids, best-first: preset leads, quality descends; distil
# English-only models trail (every id must appear as a fallback).
WHISPER_PREFERENCE: dict[str, list[str]] = {
    "gpu_high": [
        "large-v3-turbo", "large-v3", "medium", "small", "base", "tiny",
        "distil-large-v3.5", "distil-small.en",
    ],
    "gpu_low": [
        "large-v3-turbo", "medium", "small", "large-v3", "base", "tiny",
        "distil-large-v3.5", "distil-small.en",
    ],
    "cpu": [
        "small", "base", "tiny", "medium", "large-v3-turbo", "large-v3",
        "distil-large-v3.5", "distil-small.en",
    ],
}

# Per-tier MT ids, best-first: preset leads; low tiers keep the huge 3B+
# models at the tail (they won't fit) while high tiers rank them near the top.
MT_PREFERENCE: dict[str, list[str]] = {
    "gpu_high": [
        "nllb-1.3B-int8", "nllb-3.3B-int8", "madlad400-3b",
        "m2m100-1.2B-int8", "nllb-600M-int8", "m2m100-418M-int8",
    ],
    "gpu_low": [
        "nllb-600M-int8", "nllb-1.3B-int8", "m2m100-1.2B-int8",
        "m2m100-418M-int8", "nllb-3.3B-int8", "madlad400-3b",
    ],
    "cpu": [
        "nllb-600M-int8", "m2m100-418M-int8", "nllb-1.3B-int8",
        "m2m100-1.2B-int8", "nllb-3.3B-int8", "madlad400-3b",
    ],
}


def _validate() -> None:
    """Self-check the tables against the registries (dev-time invariant)."""
    for tier in PRESETS:
        if set(WHISPER_PREFERENCE[tier]) != set(WHISPER_MODELS):
            raise ValueError(f"WHISPER_PREFERENCE[{tier!r}] must cover every whisper id")
        if set(MT_PREFERENCE[tier]) != set(MT_MODELS):
            raise ValueError(f"MT_PREFERENCE[{tier!r}] must cover every MT id")
        if WHISPER_PREFERENCE[tier][0] != PRESETS[tier][0]:
            raise ValueError(f"WHISPER_PREFERENCE[{tier!r}] must lead with the preset")
        if MT_PREFERENCE[tier][0] != PRESETS[tier][1]:
            raise ValueError(f"MT_PREFERENCE[{tier!r}] must lead with the preset")


_validate()


def detect_tier() -> str:
    """Coarse hardware tier: no CUDA -> ``"cpu"``; CUDA >= 8 GB VRAM ->
    ``"gpu_high"``; < 8 GB or unknown -> ``"gpu_low"``.
    """
    if cuda_device_count() <= 0:
        return "cpu"
    vram = total_vram_bytes()
    if vram is not None and vram >= _VRAM_HIGH_BYTES:
        return "gpu_high"
    return "gpu_low"


# Cards with this much total VRAM can spare memory for near-instant captions
# alongside VRChat; smaller cards default to CPU (user decision 2026-07-08).
_GPU_DEFAULT_VRAM_BYTES = 16 * 1024**3


def default_device_choice() -> str:
    """Wizard default: ``"gpu"`` when total VRAM >= 16 GB, else ``"cpu"``."""
    vram = total_vram_bytes()
    if vram is not None and vram >= _GPU_DEFAULT_VRAM_BYTES:
        return "gpu"
    return "cpu"


def preset_for_choice(device_choice: str, tier: str | None = None) -> tuple[str, str]:
    """Preset (whisper id, mt id) for an explicit run-device choice.

    ``"cpu"`` always maps to the CPU preset regardless of hardware; ``"gpu"``
    maps to the detected (or given) GPU tier, with a CPU-only tier falling
    back to the smallest GPU preset so the choice still gets GPU-sized models.
    """
    if device_choice == "cpu":
        return PRESETS["cpu"]
    if tier is None:
        tier = detect_tier()
    return PRESETS["gpu_low" if tier == "cpu" else tier]


def tier_for_config(cfg) -> str:
    """Tier implied by the config's device choice: a forced-CPU config pins
    the ``"cpu"`` tier; anything else follows the detected hardware."""
    if cfg.stt.device == "cpu":
        return "cpu"
    return detect_tier()


def best_downloaded(
    dm, *, translate: bool, tier: str | None = None
) -> tuple[str | None, str | None]:
    """Best already-downloaded (whisper id, mt id) for ``tier``.

    Walks each tier preference best-first, returning the first id the download
    manager reports present (``None`` if none). MT is skipped when ``translate``
    is False; ``tier=None`` resolves via :func:`detect_tier`.
    """
    if tier is None:
        tier = detect_tier()
    whisper = next(
        (mid for mid in WHISPER_PREFERENCE[tier] if dm.is_whisper_downloaded(mid)),
        None,
    )
    mt = None
    if translate:
        mt = next(
            (mid for mid in MT_PREFERENCE[tier] if dm.is_mt_downloaded(MT_MODELS[mid])),
            None,
        )
    return whisper, mt
