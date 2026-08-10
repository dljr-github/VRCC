"""Restoring settings to what the benchmarks advise, and reading the spoken
languages that steer them.

Split out of :mod:`vrcc.core.recommend` for the 500-line cap. That module ranks
models; this one writes the answer back to a config, which is the other half of
the same job and the only half that mutates anything. ``recommend`` re-exports
both public names, so existing callers and patch targets are unchanged.
"""

from __future__ import annotations

from vrcc.core import calibrate


# Fields reset_to_recommended() restores, with the value it restores them to.
# Anything absent is a personal choice the recommender has no opinion about
# (target languages, spoken language, microphone, OSC address, appearance).
# translate.beam_size is restored in the body, where the config module is
# already imported. It belongs here: the load migration only reaches configs
# written before schema 2, so a beam narrowed after that has no other way back.
_RECOMMENDED_ENGINE_FIELDS = {
    "stt": {"device": "auto", "device_index": 0, "compute_type": "auto",
            "cpu_threads": 0, "num_workers": 1},
    "translate": {"device": "auto", "device_index": 0, "compute_type": "auto",
                  "inter_threads": 1, "intra_threads": 0, "max_queued_batches": 0},
}


def reset_to_recommended(cfg, dm=None) -> dict[str, object]:
    """Restore every setting the benchmarks have an opinion about, in place.

    Picks the voice and translation models for this machine's tier and the
    configured spoken language, preferring ones already downloaded so the app
    can still caption afterwards; returns the device/compute/thread fields to
    automatic; and applies the performance mode :func:`recommended_profile`
    advises for the chosen model. Personal choices are left alone: target
    languages, spoken language, microphone, OSC address, theme and interface
    language. Returns the values it settled on, for the caller to show.
    """
    from vrcc.core import recommend
    from vrcc.core.config import TranslateConfig, apply_profile

    for section_name, fields in _RECOMMENDED_ENGINE_FIELDS.items():
        section = getattr(cfg, section_name)
        for field, value in fields.items():
            setattr(section, field, value)
    cfg.translate.beam_size = TranslateConfig().beam_size

    languages = spoken_whisper_codes(cfg)
    index = cfg.stt.device_index
    choice = recommend.default_device_choice(index)
    tier = "cpu" if choice == "cpu" else recommend.detect_tier(index)
    if tier == "cpu" and choice != "cpu":
        tier = "gpu_low"

    # The wizard's CPU verdict (a small card should stay VRChat's) must bind
    # the device too: "auto" resolves to cuda whenever CUDA is usable, which
    # would run the CPU-tier models on the GPU the verdict chose to spare and
    # compute the performance mode for the wrong device.
    if choice == "cpu":
        cfg.stt.device = "cpu"
        cfg.translate.device = "cpu"

    factor = calibrate.cached_factor(cfg)
    vram_mb = recommend.detected_vram_mb(cfg.stt.device_index)
    whisper, mt = recommend.preset_for_choice(
        choice, tier=tier, languages=languages, factor=factor, vram_mb=vram_mb
    )
    if dm is not None:
        # A recommended model that is not downloaded would leave the app
        # unable to caption until it is; prefer what is already on disk.
        have_whisper, have_mt = recommend.best_downloaded(
            dm, translate=cfg.translate.enabled, tier=tier,
            languages=languages, factor=factor, vram_mb=vram_mb,
        )
        whisper = have_whisper or whisper
        mt = have_mt or mt

    cfg.stt.model = whisper
    # The MT pick is only meaningful while translation is on: overwriting the
    # stored model while it is off would swap a possibly-downloaded choice for
    # a possibly-missing preset the user never sees applied.
    if mt is not None and cfg.translate.enabled:
        cfg.translate.model = mt

    device = "cpu" if choice == "cpu" else "cuda"
    apply_profile(cfg, recommend.recommended_profile(whisper, device, factor) or "latency")
    return {"stt_model": whisper, "mt_model": mt, "profile": cfg.gui.profile}


def _whisper_code(display_name: str) -> str | None:
    """The Whisper code for a caption-language display name, or ``None`` when
    the registry does not know it (a hand-edited config)."""
    from vrcc.core.languages import LANGUAGES

    lang = LANGUAGES.get(display_name)
    return lang.whisper if lang is not None else None


def spoken_whisper_codes(cfg) -> tuple[str, ...]:
    """Whisper codes for the languages the user says they speak, deduplicated.

    Prefers the wizard's explicit answer (``stt.spoken_languages``) and falls
    back to the single stored source language, so a config written before that
    question existed still reranks the way it always did. "auto" and unknown
    names contribute nothing: an empty result keeps the recommendation
    language-blind rather than guessing at one.

    Deduplication matters -- Chinese Simplified and Traditional are separate
    display languages sharing the Whisper code "zh", and a model covering
    "zh" covers both as far as ranking is concerned.
    """
    names = cfg.stt.spoken_languages or [cfg.stt.source_language]
    codes: list[str] = []
    for name in names:
        code = _whisper_code(name)
        if code is not None and code not in codes:
            codes.append(code)
    return tuple(codes)
