"""The decoding settings measured translation quality depends on.

These need no model and run on every suite, which is the point: the worst
translation regression this project has shipped was not a model change but a
CONFIG change. The Speed profile wrote ``translate.beam_size = 1``, and greedy
decoding does not merely translate slightly worse, it rewrites content it
cannot handle: NLLB turned "Okay 1,2,3,4,5,6,7" into "现在,我们要做什么?"
("what are we going to do now?") and "Okay" into "现在,我们要去." Nothing in the
suite objected, because every test asserted structure rather than settings.

The real-model counterparts live in tests/integration/test_translate_real.py.
Those cost a download and are opt-in; these are the ones that must never be
skipped, because they are the ones that would have caught it.
"""

from __future__ import annotations

from vrcc.core.config import PROFILES, TranslateConfig
from vrcc.core.recommend import MT_PREFERENCE, PRESETS
from vrcc.translate.registry import MT_MODELS


def test_default_decoding_is_not_greedy():
    cfg = TranslateConfig()

    assert cfg.beam_size >= 2, (
        "greedy MT decoding fabricates content rather than translating it"
    )


def test_default_anti_repetition_guards_are_on():
    # A translator that loops fills all three chatbox lines with one phrase.
    cfg = TranslateConfig()

    assert cfg.repetition_penalty > 1.0
    assert cfg.no_repeat_ngram_size > 0


def test_no_profile_bundle_touches_translation():
    """The Speed/Quality bundles are derived from speech-recognition beam
    measurements, which say nothing about translation. When they carried
    translate.beam_size, picking Speed silently made translations worse."""
    for name, bundle in PROFILES.items():
        assert "translate" not in bundle, name


def test_recommended_models_lead_with_the_measured_best_family():
    """Blind A/B over 60 VRChat-style utterances into Japanese, Chinese and
    Korean, three independent judges, identity randomised per record: nllb-600M
    beat m2m100-418M 108 to 37 with 35 ties, and 60 to 16 unanimously. If a
    preset or the preference order leads with another family again, that should
    be a decision someone made, not a drift nobody noticed."""
    assert TranslateConfig().model.startswith("nllb")
    for tier, (_whisper, mt_id) in PRESETS.items():
        assert mt_id.startswith("nllb"), (tier, mt_id)
    for tier, order in MT_PREFERENCE.items():
        assert order[0].startswith("nllb"), (tier, order[0])


def test_every_tier_can_reach_every_registered_model():
    """best_downloaded walks these lists and takes the first id present on
    disk. A model missing from a tier's walk is unreachable there however the
    user got it, so the walk has to cover the registry exactly.

    No ordering assertion beyond the head: gpu_high deliberately leads with the
    larger models and falls back to the smaller ones, so there is no single
    size ordering that holds across tiers.
    """
    for tier, order in MT_PREFERENCE.items():
        assert set(order) == set(MT_MODELS), tier
        assert len(order) == len(set(order)), tier


def test_the_permissive_alternative_stays_available():
    """m2m100 is the only non-NC option below MADLAD's 3.5 GB. It lost on
    quality, but a user who needs a commercial license has to be able to reach
    it, so it may not be dropped from the registry or from any tier's walk."""
    assert MT_MODELS["m2m100-418M-int8"].license == "MIT"
    for tier, order in MT_PREFERENCE.items():
        assert "m2m100-418M-int8" in order, tier
