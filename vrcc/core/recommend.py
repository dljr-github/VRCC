"""Hardware-tier detection and benchmark-derived model recommendations (Qt-free).

Tiers: ``gpu_high`` (usable CUDA, >= 16 GB, tensor cores), ``gpu_low``
(smaller, older or unknown), ``cpu`` (no CUDA the install can drive). ``WHISPER_PREFERENCE`` and
the whisper half of ``PRESETS`` are derived at import time from the measured
``STT_BENCH`` table via :func:`_rank_whisper`, language-blind;
:func:`preset_for_choice` and :func:`best_downloaded` take optional Whisper
language codes to rerank for the languages the user says they speak
(:func:`spoken_whisper_codes` reads them off a config). ``MT_PREFERENCE``
stays hand-ordered. Both feed the per-tier walk in :func:`best_downloaded`.
"""

from __future__ import annotations

from vrcc.core.bench_tables import BEAM_BENCH, STT_BENCH, STT_VRAM_MB
from vrcc.core.hardware import (
    can_run_cuda,
    compute_capability,
    total_vram_bytes,
)
from vrcc.stt.registry import WHISPER_MODELS
from vrcc.translate.registry import MT_MODELS

# VRChat's own recommended spec is 16 GB of VRAM, so a card below it is already
# rationing for the game before VRCC asks for any. Only at or above that bar is
# there headroom to hand the larger models. Same floor the wizard uses to
# default to GPU at all (_GPU_DEFAULT_VRAM_BYTES): a card clears both or
# neither.
#
# The bar sits under the nominal figure because NVML reports what the driver
# leaves addressable, never the number on the box: 16 GB cards read 16303 to
# 16380 MiB and the 32 GB reference card reads 32607 MiB. Comparing against a
# round 16 GiB would put every shipping 16 GB card below its own bar.
_VRAM_NOMINAL_SLACK = 1024 ** 3 // 2
_VRAM_HIGH_BYTES = 16 * 1024 ** 3 - _VRAM_NOMINAL_SLACK

# Capacity is not speed, and a 16 GB card can be a decade old: a Tesla P100 or
# P40 clears the VRAM bar comfortably. The floor is compute capability 7.0
# (Volta), the first architecture with tensor cores, because best_compute_type
# hands every card below Blackwell "int8_float16" and there is no fast hardware
# for that path before then. This is a statement about what the engines ask the
# card to do, not an estimate of how quickly it would do it.
_TENSOR_CORE_CC = (7, 0)

# Quality is worth suggesting only when it buys a visible accuracy gain for a
# latency cost the user would not notice. Below this WER improvement the two
# modes transcribe the same sentence. The cost passes on either measure: a
# small proportional growth, or an absolute increase too short to perceive
# (a ratio alone would reject +20 ms on a 40 ms model).
_QUALITY_MIN_WER_GAIN = 0.002
_QUALITY_MAX_LATENCY_GROWTH = 1.15
_QUALITY_MAX_LATENCY_INCREASE_S = 0.05

# Reference-machine median-latency budgets for live captions: over-budget
# models still transcribe but lag conversation, so they only rank as
# fallbacks. VRChat's chatbox accepts a send about every 1.3 s, and the VAD
# waits 0.6 s of silence before finalizing, so an STT median past these
# budgets is what makes a caption feel detached from the sentence.
_LATENCY_GATE_S = {"cpu": 1.0, "gpu": 0.6}

# A gpu_low card holds three things at once: the voice model, the translation
# model and VRChat, so a third each is the split. The voice model is sized
# against STT_VRAM_MB, whose note explains why a checkpoint size cannot stand
# in for a measured peak; MT_VRAM_MB is what the second third covers.
# The fallback is a third of the 8 GB gpu_low used to top out at, for when VRAM
# cannot be read (no pynvml, or the import-time ranking, which must not touch
# NVML).
_GPU_LOW_VRAM_SHARE = 3
_GPU_LOW_FALLBACK_BUDGET_MB = 8 * 1024 // _GPU_LOW_VRAM_SHARE


def vram_budget_mb(total_mb: int) -> int:
    """What a card of ``total_mb`` leaves the voice model.

    Exported so the Settings fit warning applies the same rule this ranking
    does. The two disagreeing is worse than either being wrong: it would offer
    a model without comment that the recommender had just ruled out.
    """
    return total_mb // _GPU_LOW_VRAM_SHARE


def _rank_whisper(
    tier: str,
    specs=WHISPER_MODELS,
    bench=STT_BENCH,
    vram=STT_VRAM_MB,
    languages: tuple[str, ...] | None = None,
    factor: float = 1.0,
    vram_mb: int | None = None,
) -> list[str]:
    """Best-first STT ids for ``tier``, derived from the benchmark table.

    ``factor`` scales the CPU latency column to the machine actually running
    (:func:`vrcc.core.calibrate.cached_factor`), because the table's latencies
    describe one reference machine. It is never below 1.0, so it can only push
    models past the gate, never pull them back inside it. The GPU column is
    left alone: a CPU probe says nothing about a graphics card.

    Without ``languages``, unrestricted models (``spec.languages is None``)
    precede restricted ones: tier recommendation cannot know the user's
    spoken languages, so a model that may not cover them is only ever a
    fallback. With ``languages`` (Whisper codes -- more than one when the
    user speaks several), every model that can serve *all* of them competes
    in the leading partition, and models that cannot (english_only mismatch,
    a code outside ``languages``) always trail.

    Within each partition, models inside the tier's latency budget (and, on
    ``gpu_low``, inside the VRAM budget) rank by (WER band, latency): WER
    differences under ~0.3 percentage points are ties and the faster model
    wins. Over-budget models follow, fastest first (least-bad fallback),
    then unmeasured ids by size.

    An unmeasured model never leads, whatever languages it names. That rule
    used to have an exception: STT_BENCH's WER is LibriSpeech *English*, so for
    another spoken language a model naming that language looked like a better
    prior than a measured generalist, and sense-voice-small (the only
    unmeasured id) led every CJK pick on every tier. It benchmarks extremely
    well on read speech, exact on all five sherpa-onnx reference clips, but
    field testing on real VRChat speech put faster-whisper ahead, and casual
    conversation over game audio is the workload this app has. A clean-speech
    prior that loses in the field is not a prior worth leading with. It stays
    in the registry and stays pickable; it just no longer outranks a model
    whose numbers we have.
    """
    if tier not in _TIERS:
        raise KeyError(tier)
    on_gpu = tier != "cpu"
    gate = _LATENCY_GATE_S["gpu" if on_gpu else "cpu"]
    # Through vram_budget_mb, not the same arithmetic inline: this is the rule
    # the Settings fit warning reads, and re-deriving it here let the two drift
    # apart with every test still green.
    budget_mb = (_GPU_LOW_FALLBACK_BUDGET_MB if vram_mb is None
                 else vram_budget_mb(vram_mb))

    def order(ids: list[str]) -> list[str]:
        usable, over_budget, unmeasured = [], [], []
        for mid in ids:
            row = bench.get(mid)
            if row is None:
                unmeasured.append(mid)
                continue
            wer, latency = (row[0], row[2]) if on_gpu else (row[1], row[3] * factor)
            # An id with no measured footprint is not gated: a guess is what
            # this replaced. STT_VRAM_MB's note covers which ids and why.
            peak = vram.get(mid)
            fits = tier != "gpu_low" or peak is None or peak <= budget_mb
            if latency <= gate and fits:
                usable.append((int(wer * 1000) // 3, latency, mid))
            else:
                over_budget.append((latency, mid))
        usable.sort()
        over_budget.sort()
        unmeasured.sort(key=lambda m: specs[m].size_mb)
        return (
            [t[-1] for t in usable]
            + [t[-1] for t in over_budget]
            + unmeasured
        )

    def competes(spec) -> bool:
        if not languages:
            return spec.languages is None
        # english_only is checked on top of languages so a spec carrying the
        # flag without a languages tuple still trails non-English picks.
        if spec.english_only and set(languages) != {"en"}:
            return False
        return spec.languages is None or set(languages) <= set(spec.languages)

    leading = [m for m, s in specs.items() if competes(s)]
    trailing = [m for m, s in specs.items() if not competes(s)]
    return order(leading) + order(trailing)


_TIERS = ("gpu_high", "gpu_low", "cpu")

WHISPER_PREFERENCE: dict[str, list[str]] = {t: _rank_whisper(t) for t in _TIERS}

# MT ids are hand-picked per tier: no full MT benchmark exists, so size leads
# the sizing (600M fits everywhere, 1.3B needs the high tier's headroom, and
# MT_VRAM_MB / MT_VRAM_INT8_MB record what each costs). The hand-ordering is
# deliberate, not a gap: a translation benchmark would average quality over
# target languages a given user may never use (user decision 2026-07-09).
#
# NLLB leads on measured caption quality: blind A/B over 60 VRChat-style
# utterances into Japanese, Chinese and Korean, three independent judges, model
# identity randomised per record, nllb-600M 108 wins to m2m100-418M's 37 with
# 35 ties, and unanimous on 60 items to 16. It decodes faster too. An earlier
# reading here had m2m100 ahead; that comparison was taken while the Speed
# profile forced beam 1, which crippled NLLB specifically, and it does not
# survive the schema 2 migration to beam 4.
#
# NLLB is CC-BY-NC-4.0 and m2m100 is MIT, so every tier's walk below keeps an
# m2m100 reachable for a user whose use is commercial. How far down varies by
# tier, since the walk is ordered by fit rather than by license.
_MT_PRESET = {
    "gpu_high": "nllb-1.3B-int8",
    "gpu_low": "nllb-600M-int8",
    "cpu": "nllb-600M-int8",
}

# tier -> (whisper_id, mt_id); the whisper half is the head of the derived
# per-tier ordering above.
PRESETS: dict[str, tuple[str, str]] = {
    t: (WHISPER_PREFERENCE[t][0], _MT_PRESET[t]) for t in _TIERS
}

# Per-tier MT ids, best-first: preset leads; low tiers keep the huge 3B+
# models at the tail (they won't fit) while high tiers rank them near the top.
MT_PREFERENCE: dict[str, list[str]] = {
    "gpu_high": [
        "nllb-1.3B-int8", "nllb-3.3B-int8", "madlad400-3b",
        "m2m100-1.2B-int8", "nllb-600M-int8", "m2m100-418M-int8",
    ],
    "gpu_low": [
        "nllb-600M-int8", "nllb-1.3B-int8", "m2m100-418M-int8",
        "m2m100-1.2B-int8", "nllb-3.3B-int8", "madlad400-3b",
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


# Cards with this much total VRAM can spare memory for near-instant captions
# alongside VRChat; smaller cards default to CPU (user decision 2026-07-08).
# Bound to the tier bar rather than restated, so the two cannot drift into
# defaulting a card to CPU while sizing it gpu_high.
_GPU_DEFAULT_VRAM_BYTES = _VRAM_HIGH_BYTES


def default_device_choice(index: int = 0) -> str:
    """Wizard default: ``"gpu"`` when CUDA is usable (:func:`can_run_cuda`)
    and card ``index`` has >= 16 GB, else ``"cpu"``. VRAM alone is not enough:
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


def preset_for_choice(
    device_choice: str, tier: str | None = None,
    languages: tuple[str, ...] | None = None, factor: float = 1.0,
    vram_mb: int | None = None,
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
    return preset_for_tier(resolved, languages or (), factor, vram_mb)


def preset_for_tier(
    tier: str, languages: tuple[str, ...] = (), factor: float = 1.0,
    vram_mb: int | None = None,
) -> tuple[str, str]:
    """(whisper id, mt id) for an already-resolved tier, reranked for the
    spoken languages and this machine. ``PRESETS[tier]`` is the same answer for
    the language-blind reference-speed case; a surface holding the user's
    languages (the Models window) uses this so its recommendation matches the
    wizard's for the same inputs.

    Always ranked rather than served from ``PRESETS`` on the default inputs: a
    float-equality test on ``factor`` decided which of two code paths ran, and
    the ranking is ten rows and one sort.
    """
    return (
        _rank_whisper(tier, languages=languages, factor=factor, vram_mb=vram_mb)[0],
        _MT_PRESET[tier],
    )


def tier_for_config(cfg) -> str:
    """Tier implied by the config's device choice: a forced-CPU config pins
    the ``"cpu"`` tier; anything else follows the hardware, judged on the card
    ``stt.device_index`` names so this agrees with the VRAM budget."""
    if cfg.stt.device == "cpu":
        return "cpu"
    return detect_tier(cfg.stt.device_index)


def recommended_profile(
    model_id: str, device: str, factor: float = 1.0
) -> str | None:
    """Which performance mode suits ``model_id`` on ``device``, from the
    measured beam-1 (Speed) and beam-5 (Quality) runs.

    ``"quality"`` when widening the beam bought a visible accuracy gain
    without a latency the user would feel, ``"latency"`` when it did not, and
    ``None`` when the question does not apply: the onnx-asr decoders are
    greedy, and an unmeasured model or device has nothing to advise from.
    ``device`` is ``"cpu"`` or ``"cuda"``; anything else resolves to the GPU
    row, matching how the benchmark labels its devices.

    ``factor`` scales the CPU rows the way :func:`_rank_whisper` does, and has
    to be the same number: sizing a model for a slow machine and then judging
    its beam against the reference machine's clock hands that machine the wide
    beam precisely when it has no room for it.
    """
    spec = WHISPER_MODELS.get(model_id)
    if spec is None or spec.backend != "whisper":
        return None
    speed = STT_BENCH.get(model_id)
    if speed is None:
        return None

    on_cpu = device == "cpu"
    speed_wer, speed_latency = (
        (speed[1], speed[3] * factor) if on_cpu else (speed[0], speed[2])
    )
    gate = _LATENCY_GATE_S["cpu" if on_cpu else "gpu"]

    quality = BEAM_BENCH.get(model_id, {}).get("cpu" if on_cpu else "gpu")
    if quality is None:
        # Unmeasured at beam 5. A model already past its latency budget at
        # beam 1 only gets slower, so advise Speed; otherwise stay silent.
        return "latency" if speed_latency > gate else None

    quality_wer, quality_latency = quality
    if on_cpu:
        quality_latency *= factor
    if speed_latency <= 0 or quality_latency > gate:
        return "latency"
    if speed_wer - quality_wer < _QUALITY_MIN_WER_GAIN:
        return "latency"
    affordable = (
        quality_latency <= speed_latency * _QUALITY_MAX_LATENCY_GROWTH
        or quality_latency - speed_latency <= _QUALITY_MAX_LATENCY_INCREASE_S
    )
    return "quality" if affordable else "latency"


def best_downloaded(
    dm, *, translate: bool, tier: str | None = None,
    languages: tuple[str, ...] | None = None, factor: float = 1.0,
    vram_mb: int | None = None,
) -> tuple[str | None, str | None]:
    """Best already-downloaded (whisper id, mt id) for ``tier``.

    Walks each tier preference best-first, returning the first id the download
    manager reports present (``None`` if none). MT is skipped when ``translate``
    is False; ``tier=None`` resolves via :func:`detect_tier`. ``languages``
    (Whisper codes) reranks the whisper walk for known spoken languages.
    """
    if tier is None:
        tier = detect_tier()
    pref = _rank_whisper(tier, languages=languages, factor=factor, vram_mb=vram_mb)
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
