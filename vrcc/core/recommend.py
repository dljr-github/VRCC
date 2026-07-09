"""Hardware-tier detection and benchmark-derived model recommendations (Qt-free).

Tiers: ``gpu_high`` (CUDA >= 8 GB), ``gpu_low`` (< 8 GB / unknown), ``cpu``.
``WHISPER_PREFERENCE`` and the whisper half of ``PRESETS`` are derived at
import time from the measured ``STT_BENCH`` table via :func:`_rank_whisper`,
language-blind; :func:`preset_for_choice` and :func:`best_downloaded` take an
optional Whisper language code to rerank for a known spoken language.
``MT_PREFERENCE`` stays hand-ordered (no MT measurements yet). Both feed the
per-tier best-first walk in :func:`best_downloaded`.
"""

from __future__ import annotations

from vrcc.core.hardware import cuda_device_count, total_vram_bytes
from vrcc.stt.registry import WHISPER_MODELS
from vrcc.translate.registry import MT_MODELS

_VRAM_HIGH_BYTES = 8 * 1024 ** 3

# Measured with tools/bench_stt.py on LibriSpeech test-clean (100 utterances);
# reference machine: Ryzen 9 9950X3D + RTX 5090, full run recorded in
# benchmarks/rtx-5090-ryzen-9950x3d.json. WER is scored with the quality gates
# opened, so it measures what the model recognized rather than what the gates
# suppressed. A model added to the STT registry must be benchmarked with
# tools/bench_stt.py and its row added here, or it ranks behind every measured
# model in its partition. CPU latency is the load-sensitive column: it moved by
# about a quarter between runs on a busy machine, so treat the ordering as the
# signal and the absolute value as approximate.
# id -> (wer_gpu, wer_cpu, gpu_median_s, cpu_median_s)
STT_BENCH: dict[str, tuple[float, float, float, float]] = {
    "tiny": (0.074, 0.079, 0.03, 0.13),
    "base": (0.057, 0.059, 0.04, 0.25),
    "small": (0.037, 0.037, 0.09, 0.75),
    "medium": (0.027, 0.026, 0.17, 2.41),
    "large-v3": (0.017, 0.018, 0.24, 3.90),
    "large-v3-turbo": (0.017, 0.016, 0.07, 2.81),
    "distil-large-v3.5": (0.024, 0.023, 0.06, 2.78),
    "distil-small.en": (0.040, 0.040, 0.04, 0.64),
    "parakeet-tdt-0.6b-v3": (0.023, 0.023, 0.21, 0.13),
    "canary-1b-v2": (0.018, 0.018, 0.67, 0.32),
}

# Beam 5 (the Quality mode) against beam 1 (Speed), same runs. Only whisper
# models have a beam to widen: the onnx-asr decoders are greedy, which is why
# the Mode control greys out for them. Models absent from a device's row were
# not measured there because they already lag past the latency gate at beam 1.
# id -> {device: (wer_beam5, median_beam5_s)}
BEAM_BENCH: dict[str, dict[str, tuple[float, float]]] = {
    "tiny": {"gpu": (0.071, 0.04), "cpu": (0.070, 0.14)},
    "base": {"gpu": (0.047, 0.06), "cpu": (0.049, 0.26)},
    "small": {"gpu": (0.035, 0.09), "cpu": (0.038, 0.78)},
    "medium": {"gpu": (0.024, 0.18)},
    "large-v3": {"gpu": (0.017, 0.26)},
    "large-v3-turbo": {"gpu": (0.018, 0.07)},
    "distil-large-v3.5": {"gpu": (0.023, 0.06)},
    "distil-small.en": {"gpu": (0.041, 0.04), "cpu": (0.040, 0.81)},
}

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

# gpu_low shares < 8 GB VRAM with VRChat: float16 inflation of the CT2 int8
# checkpoint sizes plus activations does not reliably fit past ~2 GB, so
# larger models (large-v3 at 3090 MB) drop to the fallback group there.
_GPU_LOW_MAX_MODEL_MB = 2000


def _rank_whisper(
    tier: str, specs=WHISPER_MODELS, bench=STT_BENCH, language: str | None = None
) -> list[str]:
    """Best-first STT ids for ``tier``, derived from the benchmark table.

    Without ``language``, unrestricted models (``spec.languages is None``)
    precede restricted ones: tier recommendation cannot know the user's
    spoken language, so a model that may not cover it is only ever a
    fallback. With ``language`` (a Whisper code), every model that can serve
    it -- unrestricted, or restricted with the code in its ``languages`` --
    competes in the leading partition, and models that cannot serve it
    (english_only mismatch, code outside ``languages``) always trail.
    Within each partition, models inside the tier's latency budget (and, on
    ``gpu_low``, inside the VRAM size cap) rank by (WER band, latency): WER
    differences under ~0.3 percentage points are ties and the faster model
    wins. Over-budget models follow, fastest first (least-bad fallback),
    then unmeasured ids by size.
    """
    on_gpu = tier != "cpu"
    gate = _LATENCY_GATE_S["gpu" if on_gpu else "cpu"]

    def order(ids: list[str]) -> list[str]:
        usable, over_budget, unmeasured = [], [], []
        for mid in ids:
            row = bench.get(mid)
            if row is None:
                unmeasured.append(mid)
                continue
            wer, latency = (row[0], row[2]) if on_gpu else (row[1], row[3])
            fits = tier != "gpu_low" or specs[mid].size_mb <= _GPU_LOW_MAX_MODEL_MB
            if latency <= gate and fits:
                usable.append((int(wer * 1000) // 3, latency, mid))
            else:
                over_budget.append((latency, mid))
        usable.sort()
        over_budget.sort()
        unmeasured.sort(key=lambda m: specs[m].size_mb)
        return [t[-1] for t in usable] + [t[-1] for t in over_budget] + unmeasured

    def competes(spec) -> bool:
        if language is None:
            return spec.languages is None
        # english_only is checked on top of languages so a spec carrying the
        # flag without a languages tuple still trails non-English picks.
        if spec.english_only and language != "en":
            return False
        return spec.languages is None or language in spec.languages

    leading = [m for m, s in specs.items() if competes(s)]
    trailing = [m for m, s in specs.items() if not competes(s)]
    return order(leading) + order(trailing)


_TIERS = ("gpu_high", "gpu_low", "cpu")

WHISPER_PREFERENCE: dict[str, list[str]] = {t: _rank_whisper(t) for t in _TIERS}

# MT ids are still hand-picked per tier: no MT benchmark measurements exist
# yet, so size is the only sizing signal (600M fits everywhere, 1.3B needs
# the high tier's VRAM headroom). The hand-ordering is also deliberate, not
# just a gap: a translation benchmark would average quality over target
# languages a given user may never use (user decision 2026-07-09).
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


def preset_for_choice(
    device_choice: str, tier: str | None = None, language: str | None = None
) -> tuple[str, str]:
    """Preset (whisper id, mt id) for an explicit run-device choice.

    ``"cpu"`` always maps to the CPU preset regardless of hardware; ``"gpu"``
    maps to the detected (or given) GPU tier, with a CPU-only tier falling
    back to the smallest GPU preset so the choice still gets GPU-sized models.
    ``language`` (a Whisper code) reranks the whisper half for a known spoken
    language; the MT half is language-blind.
    """
    if device_choice == "cpu":
        resolved = "cpu"
    else:
        if tier is None:
            tier = detect_tier()
        resolved = "gpu_low" if tier == "cpu" else tier
    if language is None:
        return PRESETS[resolved]
    return _rank_whisper(resolved, language=language)[0], _MT_PRESET[resolved]


def tier_for_config(cfg) -> str:
    """Tier implied by the config's device choice: a forced-CPU config pins
    the ``"cpu"`` tier; anything else follows the detected hardware."""
    if cfg.stt.device == "cpu":
        return "cpu"
    return detect_tier()


def recommended_profile(model_id: str, device: str) -> str | None:
    """Which performance mode suits ``model_id`` on ``device``, from the
    measured beam-1 (Speed) and beam-5 (Quality) runs.

    ``"quality"`` when widening the beam bought a visible accuracy gain
    without a latency the user would feel, ``"latency"`` when it did not, and
    ``None`` when the question does not apply: the onnx-asr decoders are
    greedy, and an unmeasured model or device has nothing to advise from.
    ``device`` is ``"cpu"`` or ``"cuda"``; anything else resolves to the GPU
    row, matching how the benchmark labels its devices.
    """
    spec = WHISPER_MODELS.get(model_id)
    if spec is None or spec.backend != "whisper":
        return None
    speed = STT_BENCH.get(model_id)
    if speed is None:
        return None

    on_cpu = device == "cpu"
    speed_wer, speed_latency = (speed[1], speed[3]) if on_cpu else (speed[0], speed[2])
    gate = _LATENCY_GATE_S["cpu" if on_cpu else "gpu"]

    quality = BEAM_BENCH.get(model_id, {}).get("cpu" if on_cpu else "gpu")
    if quality is None:
        # Unmeasured at beam 5. A model already past its latency budget at
        # beam 1 only gets slower, so advise Speed; otherwise stay silent.
        return "latency" if speed_latency > gate else None

    quality_wer, quality_latency = quality
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
    dm, *, translate: bool, tier: str | None = None, language: str | None = None
) -> tuple[str | None, str | None]:
    """Best already-downloaded (whisper id, mt id) for ``tier``.

    Walks each tier preference best-first, returning the first id the download
    manager reports present (``None`` if none). MT is skipped when ``translate``
    is False; ``tier=None`` resolves via :func:`detect_tier`. ``language`` (a
    Whisper code) reranks the whisper walk for a known spoken language.
    """
    if tier is None:
        tier = detect_tier()
    pref = (
        WHISPER_PREFERENCE[tier]
        if language is None
        else _rank_whisper(tier, language=language)
    )
    whisper = next((mid for mid in pref if dm.is_whisper_downloaded(mid)), None)
    mt = None
    if translate:
        mt = next(
            (mid for mid in MT_PREFERENCE[tier] if dm.is_mt_downloaded(MT_MODELS[mid])),
            None,
        )
    return whisper, mt
