"""Which model wins once a tier is known: the VRAM budget a card leaves the
voice model, the benchmark-derived Whisper and translation preference
orderings, and the presets built on top of them.

Split out of :mod:`vrcc.core.recommend` for the line cap. The hardware
verdict (:func:`vrcc.core.recommend.detect_tier`,
:func:`vrcc.core.recommend.default_device_choice`) stays there: tests patch
the probes those functions read (``can_run_cuda``, ``total_vram_bytes``,
``compute_capability``) by name on that module, and a bare name is only ever
resolved on the module where the reading function is defined, so keeping
them together is what keeps a patch on ``recommend`` reaching them.
:func:`vram_budget_mb` moved here instead of staying beside them, because its
only caller is :func:`_rank_whisper` below; :mod:`vrcc.core.recommend`
re-exports it unchanged for the Settings fit warning that also calls it.
"""

from __future__ import annotations

from vrcc.core.bench_tables import BEAM_BENCH, STT_BENCH, stt_vram_table
from vrcc.stt.registry import WHISPER_MODELS
from vrcc.translate.registry import MT_MODELS

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

# VRChat's typical use exceeds 6 GB; the owner wants it kept at 8 GB (user
# decision 2026-09-12), and the voice model gets whatever a card has left
# after that reservation and the translation model's marginal cost. An 8 GB
# card cannot honour it and still caption: a knowing compromise, not a
# measurement, landing on the floor model with VRChat holding 6494 MB.
_VRCHAT_RESERVE_MB = 8 * 1024

# The translation model's marginal cost on top of a resident STT model,
# measured on the reference machine (RTX 5090, sm120) at int8_float16, fresh
# process per figure: large-v3-turbo alone peaked at 1314 MB, large-v3-turbo
# plus nllb-600M-int8 resident together at 2339 MB. Not nllb's own peak
# (MT_VRAM_INT8_MB already counts a context of its own); the shared ~435 MB
# CUDA context is created once, so a second full row would pay it twice.
_MT_MARGINAL_MB = 1025

# The smallest voice model this app still recommends: the floor the budget
# below cannot fall under, the same id PRESETS["cpu"] leads with (_validate ties them).
_FLOOR_WHISPER_ID = "small"


def vram_budget_mb(total_mb: int, compute: str) -> int:
    """What a card of ``total_mb`` leaves the voice model: total minus
    ``_VRCHAT_RESERVE_MB`` minus ``_MT_MARGINAL_MB``, floored at
    ``_FLOOR_WHISPER_ID``'s peak (read via :func:`stt_vram_table` for
    ``compute``) so a negative result still names a real model.

    Exported so the Settings fit warning applies the same rule this ranking
    does. The two disagreeing is worse than either being wrong: it would offer
    a model without comment that the recommender had just ruled out.
    """
    floor = stt_vram_table(compute)[_FLOOR_WHISPER_ID]
    return max(total_mb - _VRCHAT_RESERVE_MB - _MT_MARGINAL_MB, floor)


def _rank_whisper(
    tier: str,
    specs=WHISPER_MODELS,
    bench=STT_BENCH,
    vram=None,
    languages: tuple[str, ...] | None = None,
    factor: float = 1.0,
    vram_mb: int | None = None,
    compute: str = "int8",
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

    ``compute`` picks WHICH measured VRAM peak the budget is applied to: the
    same model costs 1.13x to 1.67x more at float16 than at int8_float16, and a
    card whose supported compute types omit every int8* entry always pays the
    higher one. It defaults to the int8 table for the import-time,
    machine-blind ``WHISPER_PREFERENCE``; a caller that knows the card passes
    :func:`resolved_compute_type`, which is the same value the Settings fit
    warning sizes against.

    Within each partition, models inside the tier's latency budget (and, on
    ``gpu_low``, inside the VRAM budget) rank by (WER band, latency): WER
    differences under ~0.3 percentage points are ties and the faster model
    wins. Over-budget models follow, fastest first (least-bad fallback),
    then unmeasured ids by size.

    An unmeasured model never leads, whatever languages it names. The
    temptation to except one is real: STT_BENCH's WER is LibriSpeech *English*,
    so for another spoken language a model naming that language reads like a
    better prior than a measured generalist, and sense-voice-small (the only
    unmeasured id) would lead every CJK pick on every tier. It benchmarks
    extremely well on read speech, exact on all five sherpa-onnx reference
    clips, but field testing on real VRChat speech put faster-whisper ahead,
    and casual conversation over game audio is the workload this app has. A
    clean-speech prior that loses in the field is not a prior worth leading
    with. It stays in the registry and stays pickable.
    """
    if tier not in _TIERS:
        raise KeyError(tier)
    on_gpu = tier != "cpu"
    gate = _LATENCY_GATE_S["gpu" if on_gpu else "cpu"]
    # Through vram_budget_mb, not inline, so the two cannot drift apart. An
    # unreadable card (no pynvml, or the import-time ranking, which must not
    # touch NVML) is charged the VRChat reservation alone: the floor a real 8 GB card gets.
    budget_mb = vram_budget_mb(_VRCHAT_RESERVE_MB if vram_mb is None else vram_mb, compute)
    peaks = stt_vram_table(compute) if vram is None else vram

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
            peak = peaks.get(mid)
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
    if _FLOOR_WHISPER_ID != PRESETS["cpu"][0]:
        raise ValueError("_FLOOR_WHISPER_ID must match PRESETS['cpu'][0]")
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


def preset_for_tier(
    tier: str, languages: tuple[str, ...] = (), factor: float = 1.0,
    vram_mb: int | None = None, compute: str = "int8",
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
        _rank_whisper(
            tier, languages=languages, factor=factor, vram_mb=vram_mb,
            compute=compute,
        )[0],
        _MT_PRESET[tier],
    )


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
