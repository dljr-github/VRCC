"""Hardware-tier detection and the entry points that resolve a tier before
ranking within it (Qt-free).

Tiers: ``gpu_high`` (usable CUDA, >= 16 GB, tensor cores), ``gpu_low``
(smaller, older or unknown), ``cpu`` (no CUDA the install can drive).

Which model wins once a tier is known, including the VRAM budget a card
leaves the voice model, lives in :mod:`vrcc.core.recommend_rank`, re-exported
below. :func:`preset_for_choice`, :func:`tier_for_config` and
:func:`best_downloaded` stay here instead of moving with it: each falls back
to :func:`detect_tier` by a bare name, and a bare name is resolved on the
module where the function is defined, not where it is called from. Keeping
them beside the hardware probes they read (``can_run_cuda``,
``total_vram_bytes``, ``compute_capability``) is what lets a test's
``monkeypatch.setattr(recommend, "can_run_cuda", ...)`` still reach them.
"""

from __future__ import annotations

from vrcc.core.bench_tables import BEAM_BENCH, STT_BENCH  # noqa: F401
from vrcc.core.hardware import (
    best_compute_type,
    can_run_cuda,
    compute_capability,
    total_vram_bytes,
)
from vrcc.translate.registry import MT_MODELS

# VRChat's own recommended spec is 16 GB of VRAM, so a card below it is already
# rationing for the game before VRCC asks for any. Only at or above that bar is
# there headroom to hand the larger models. The wizard's GPU default
# (_GPU_DEFAULT_VRAM_BYTES) is a separate, lower bar; see that constant.
#
# The bar sits under the nominal figure because NVML reports what the driver
# leaves addressable, never the number on the box: 16 GB cards read 16303 to
# 16380 MiB and the 32 GB reference card reads 32607 MiB. Comparing against a
# round 16 GiB would put every shipping 16 GB card below its own bar.
_VRAM_NOMINAL_SLACK = 1024 ** 3 // 2
_VRAM_HIGH_BYTES = 16 * 1024 ** 3 - _VRAM_NOMINAL_SLACK

# Capacity is not speed, and a 16 GB card can be a decade old: a Tesla P100 or
# P40 clears the VRAM bar. The floor is compute capability 7.0 (Volta), the
# first architecture with tensor cores: best_compute_type hands out
# "int8_float16" whenever CTranslate2 reports it supported, without checking
# for tensor cores, and there is no fast hardware for that path before Volta.
# This is a statement about what the engines ask the card to do, not an
# estimate of how quickly it would do it.
_TENSOR_CORE_CC = (7, 0)


def resolved_compute_type(compute_type: str = "auto", device_index: int = 0) -> str:
    """What the engines will actually run at, resolved the way
    :func:`vrcc.core.hardware.resolve` resolves it for them: a pinned value
    wins, otherwise the best type the card supports.

    Here rather than beside the fit warning that first needed it, because the
    ranking has to size against the same table: a card whose supported compute
    types omit int8 always pays the float16 peak, and the two reading
    different tables put "Recommended for your PC" and "leaves little room
    for VRChat" on one row.
    """
    if compute_type != "auto":
        return compute_type
    if not can_run_cuda():
        # No card to size against, and asking CTranslate2 what a CUDA device
        # supports on a machine without one is a probe for an answer nothing
        # reads: the VRAM gate only applies on the gpu_low tier.
        return "float32"
    return best_compute_type("cuda", device_index)


def detect_tier(index: int = 0) -> str:
    """Coarse hardware tier: no usable CUDA (:func:`can_run_cuda`, which a
    visible device with no loadable cuBLAS fails) -> ``"cpu"``; usable CUDA with
    >= 16 GB VRAM on a card new enough for the fp16 path -> ``"gpu_high"``;
    anything smaller, older or unknown -> ``"gpu_low"``.

    Demotion is gentle by design: ``gpu_low`` still leads with the same voice
    model and sizes it against measured VRAM, so an old large card loses only
    the bigger translation model.
    """
    if not can_run_cuda():
        return "cpu"
    vram = total_vram_bytes(index)
    if vram is not None and vram >= _VRAM_HIGH_BYTES and _has_tensor_cores(index):
        return "gpu_high"
    return "gpu_low"


def _has_tensor_cores(index: int = 0) -> bool:
    """Whether card ``index`` is new enough for the fp16 path the engines use.

    An unreadable capability (no pynvml) is not evidence of an old card, so it
    passes: the VRAM bar still has to be cleared, and treating a missing reading
    as a failure would demote every install without pynvml.
    """
    cc = compute_capability(index)
    return cc is None or cc >= _TENSOR_CORE_CC


def detected_vram_mb(index: int = 0) -> int | None:
    """Total VRAM in MB for the ranking's budget, or ``None`` when it cannot be
    read (which keeps the conservative fallback rather than guessing).

    ``index`` is the CUDA device the models actually load onto
    (``stt.device_index``): on a mixed multi-GPU box, sizing against card 0
    would budget for a card the engine never touches.
    """
    vram = total_vram_bytes(index) if can_run_cuda() else None
    return None if vram is None else vram // (1024 ** 2)


# large-v3-turbo and nllb-600M-int8, both resident on CUDA, measured on the
# reference machine (RTX 5090, sm120): at float16, the pair takes 4198 MB; at
# int8_float16, 2339 MB. best_compute_type picks int8_float16 wherever
# CTranslate2 reports it supported. The bar itself is 8 GB (user decision
# 2026-09-12), lower than the 16 GB the tier split uses; vram_budget_mb's
# reserve is what decides how much of that 8 GB the voice model actually
# keeps, not this constant.
#
# Built the same way as _VRAM_HIGH_BYTES and under the nominal figure for the
# same reason: NVML reports what the driver leaves addressable, not the
# number on the box. Must stay <= _VRAM_HIGH_BYTES, or a card sized for the
# gpu_high tier could default to CPU.
_GPU_DEFAULT_VRAM_BYTES = 8 * 1024 ** 3 - _VRAM_NOMINAL_SLACK


def default_device_choice(index: int = 0) -> str:
    """Wizard default: ``"gpu"`` when CUDA is usable (:func:`can_run_cuda`)
    and card ``index`` has at least ``_GPU_DEFAULT_VRAM_BYTES``, else
    ``"cpu"``. VRAM alone is not enough:
    NVML reads it from the display driver, which says nothing about whether this
    install ships the CUDA runtime to drive the card. ``index`` is
    ``stt.device_index``, so a multi-GPU box judges the card the engines load
    onto rather than whichever one enumerates first."""
    if not can_run_cuda():
        return "cpu"
    vram = total_vram_bytes(index)
    if vram is not None and vram >= _GPU_DEFAULT_VRAM_BYTES:
        return "gpu"
    return "cpu"


# recommend_rank depends on nothing here (no hardware probe, no detect_tier),
# so this import carries no circularity; it is placed after the hardware
# verdict above only to keep this module reading top to bottom as tier
# detection, then the ranking built on top of it.
from vrcc.core.recommend_rank import (  # noqa: E402
    MT_PREFERENCE,
    PRESETS,  # noqa: F401
    WHISPER_PREFERENCE,  # noqa: F401
    _FLOOR_WHISPER_ID,  # noqa: F401
    _MT_MARGINAL_MB,  # noqa: F401
    _MT_PRESET,  # noqa: F401
    _VRCHAT_RESERVE_MB,  # noqa: F401
    _rank_whisper,
    _validate,  # noqa: F401
    preset_for_tier,
    recommended_profile,  # noqa: F401
    vram_budget_mb,  # noqa: F401
)


def preset_for_choice(
    device_choice: str, tier: str | None = None,
    languages: tuple[str, ...] | None = None, factor: float = 1.0,
    vram_mb: int | None = None, compute: str = "int8",
) -> tuple[str, str]:
    """Preset (whisper id, mt id) for an explicit run-device choice.

    ``"cpu"`` always maps to the CPU preset regardless of hardware; ``"gpu"``
    maps to the detected (or given) GPU tier, with a CPU-only tier falling
    back to the smallest GPU preset so the choice still gets GPU-sized models.
    ``languages`` (Whisper codes) reranks the whisper half for known spoken
    languages; the MT half is language-blind. A model that cannot detect the
    spoken language itself (the distil English pair) can still be chosen here:
    it is only reached with concrete ``languages``, which the caller has
    already written to the config. Empty/``None`` languages fall back to the
    language-blind presets, where such models never lead.
    """
    if device_choice == "cpu":
        resolved = "cpu"
    else:
        if tier is None:
            tier = detect_tier()
        resolved = "gpu_low" if tier == "cpu" else tier
    return preset_for_tier(resolved, languages or (), factor, vram_mb, compute)


def tier_for_config(cfg) -> str:
    """Tier implied by the config's device choice: a forced-CPU config pins
    the ``"cpu"`` tier; anything else follows the hardware, judged on the card
    ``stt.device_index`` names so this agrees with the VRAM budget."""
    if cfg.stt.device == "cpu":
        return "cpu"
    return detect_tier(cfg.stt.device_index)


def best_downloaded(
    dm, *, translate: bool, tier: str | None = None,
    languages: tuple[str, ...] | None = None, factor: float = 1.0,
    vram_mb: int | None = None, compute: str = "int8",
) -> tuple[str | None, str | None]:
    """Best already-downloaded (whisper id, mt id) for ``tier``.

    Walks each tier preference best-first, returning the first id the download
    manager reports present (``None`` if none). MT is skipped when ``translate``
    is False; ``tier=None`` resolves via :func:`detect_tier`. ``languages``
    (Whisper codes) reranks the whisper walk for known spoken languages.
    """
    if tier is None:
        tier = detect_tier()
    pref = _rank_whisper(
        tier, languages=languages, factor=factor, vram_mb=vram_mb, compute=compute
    )
    whisper = next((mid for mid in pref if dm.is_whisper_downloaded(mid)), None)
    mt = None
    if translate:
        mt = next(
            (mid for mid in MT_PREFERENCE[tier] if dm.is_mt_downloaded(MT_MODELS[mid])),
            None,
        )
    return whisper, mt


# Imported last: recommend_reset imports from this module, so the names it
# needs must already exist. Re-exported because reset_to_recommended and
# spoken_whisper_codes were always part of this module's surface.
from vrcc.core.recommend_reset import (  # noqa: E402,F401
    reset_to_recommended,
    spoken_whisper_codes,
)
