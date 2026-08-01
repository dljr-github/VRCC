"""Language-aware model ranking (``vrcc.core.recommend``): which model wins
once VRCC knows which languages the user actually speaks, including the
multi-language answer the first-run wizard collects.

Split from test_recommend.py for the 500-line cap. The exact orderings here
come from the same STT_BENCH reference run the sibling module documents, and
_EXPECTED_WHISPER_PREFERENCE is imported from it so the language-blind
baseline can never drift between the two files.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from tests.test_recommend import _EXPECTED_WHISPER_PREFERENCE, _TIERS, _FakeDM
from vrcc.core import recommend
from vrcc.stt.registry import WHISPER_MODELS


# Language-aware ranking: a known spoken language (Whisper code) lets a
# restricted specialist compete against the unrestricted models. Exact
# outcomes below come from the same STT_BENCH reference run.


@pytest.mark.parametrize("tier", _TIERS)
def test_rank_whisper_language_none_is_byte_identical_to_blind_lists(tier):
    assert recommend._rank_whisper(tier, languages=None) == _EXPECTED_WHISPER_PREFERENCE[tier]


def test_rank_whisper_cpu_english_puts_parakeet_first():
    # Once "en" is known, parakeet beats every whisper model on CPU: 2.3
    # percent at 0.13 s, against small's 3.7 percent at 0.75 s. Every model
    # here serves "en", so there is no trailing partition and unmeasured
    # sense-voice-small closes the list.
    assert recommend._rank_whisper("cpu", languages=("en",)) == [
        "parakeet-tdt-0.6b-v3",
        "small", "distil-small.en", "base", "tiny",
        "medium", "distil-large-v3.5", "large-v3-turbo", "large-v3",
        "sense-voice-small",
    ]


def test_rank_whisper_cpu_japanese_leads_with_the_covering_specialist():
    # STT_BENCH's WER is English, so it cannot rank Japanese. sense-voice-small
    # is unmeasured but explicitly covers "ja", which beats a measured
    # generalist that merely does not exclude it -- see _rank_whisper.
    assert recommend._rank_whisper("cpu", languages=("ja",)) == [
        "sense-voice-small",
        "small", "base", "tiny", "medium", "large-v3-turbo", "large-v3",
        "parakeet-tdt-0.6b-v3", "distil-small.en", "distil-large-v3.5",
    ]


def test_rank_whisper_gpu_high_german_keeps_turbo_first():
    # GPU WER bands: turbo/large-v3 band 5, parakeet band 7, medium band 9,
    # so parakeet slots after large-v3 and before medium. Parakeet covers "de"
    # AND is measured, so it competes on its numbers rather than being
    # promoted. The english-only distil pair cannot serve "de", so it trails,
    # and sense-voice-small (no German) trails with them -- the specialist
    # promotion only applies to a model that actually covers the language.
    assert recommend._rank_whisper("gpu_high", languages=("de",)) == [
        "large-v3-turbo", "large-v3", "parakeet-tdt-0.6b-v3",
        "medium", "small", "base", "tiny",
        "distil-large-v3.5", "distil-small.en", "sense-voice-small",
    ]


def test_rank_whisper_english_only_flag_trails_without_languages_tuple():
    # english_only is honored even when a spec forgets its languages tuple:
    # the flag alone must keep the model out of a non-English leading group.
    specs = {
        "plain": SimpleNamespace(size_mb=300, languages=None, english_only=False),
        "en-flag-only": SimpleNamespace(size_mb=300, languages=None, english_only=True),
    }
    bench = {
        "plain": (0.050, 0.050, 0.10, 0.10),
        "en-flag-only": (0.010, 0.010, 0.01, 0.01),
    }
    got = recommend._rank_whisper("gpu_high", specs=specs, bench=bench, languages=("de",))
    assert got == ["plain", "en-flag-only"]


def test_parakeet_competes_only_with_concrete_language():
    blind = recommend._rank_whisper("cpu")
    with_de = recommend._rank_whisper("cpu", languages=("de",))
    # language-blind: parakeet trails every unrestricted id
    assert blind.index("parakeet-tdt-0.6b-v3") > blind.index("tiny")
    # concrete "de" on CPU, where parakeet is inside the budget: it leads
    assert with_de[0] == "parakeet-tdt-0.6b-v3"


def test_preset_for_choice_language_reranks_whisper_half_only():
    assert recommend.preset_for_choice("cpu", languages=("en",)) == (
        "parakeet-tdt-0.6b-v3", "nllb-600M-int8",
    )
    assert recommend.preset_for_choice("gpu", tier="gpu_high", languages=("de",)) == (
        "large-v3-turbo", "nllb-1.3B-int8",
    )


def test_preset_without_a_language_never_leads_with_a_non_detecting_model():
    # No spoken language to pin a non-detecting model to, so the
    # language-blind presets must not name one.
    for tier in _TIERS:
        assert WHISPER_MODELS[recommend.PRESETS[tier][0]].auto_language


# -- spoken_whisper_codes / multi-language ranking --------------------------


def _cfg(spoken=None, source="English"):
    from vrcc.core.config import AppConfig

    cfg = AppConfig()
    cfg.stt.source_language = source
    cfg.stt.spoken_languages = list(spoken or [])
    return cfg


def test_spoken_codes_prefer_the_multi_select():
    cfg = _cfg(spoken=["Japanese", "Korean"], source="English")
    assert recommend.spoken_whisper_codes(cfg) == ("ja", "ko")


def test_spoken_codes_fall_back_to_the_single_source_language():
    # Configs written before the wizard asked have no multi-select; they must
    # keep reranking exactly as they always did.
    assert recommend.spoken_whisper_codes(_cfg(source="Japanese")) == ("ja",)


def test_spoken_codes_drop_auto_and_unknown_names():
    assert recommend.spoken_whisper_codes(_cfg(source="auto")) == ()
    assert recommend.spoken_whisper_codes(_cfg(spoken=["Klingon"])) == ()


def test_spoken_codes_deduplicate_the_two_chinese_display_languages():
    # Simplified and Traditional are separate display languages sharing the
    # Whisper code "zh"; a model covering "zh" covers both for ranking.
    cfg = _cfg(spoken=["Chinese Simplified", "Chinese Traditional"])
    assert recommend.spoken_whisper_codes(cfg) == ("zh",)


def test_rank_whisper_multi_language_needs_one_model_covering_all():
    # SenseVoice covers all three, so it leads.
    assert recommend._rank_whisper("cpu", languages=("en", "ja", "zh"))[0] == (
        "sense-voice-small"
    )


def test_rank_whisper_no_model_covers_the_mix_falls_back_to_generalists():
    # Japanese + German: no restricted model spans both, so every specialist
    # trails and the unrestricted whisper models lead on their measurements.
    got = recommend._rank_whisper("cpu", languages=("ja", "de"))
    assert got[0] == "small"
    assert got.index("sense-voice-small") > got.index("large-v3")
    assert got.index("parakeet-tdt-0.6b-v3") > got.index("large-v3")


def test_specialist_promotion_needs_actual_coverage():
    """The promotion must check the language set, not just "is restricted and
    unmeasured" -- otherwise it also promotes inside the trailing partition,
    where a restricted model is precisely one that cannot serve the language."""
    got = recommend._rank_whisper("gpu_high", languages=("de",))
    # sense-voice-small has no German; it must not jump the distil pair it
    # shares the trailing partition with.
    assert got[-1] == "sense-voice-small"


def test_english_still_ranks_on_its_measurements():
    # STT_BENCH's WER *is* English evidence, so no promotion applies there:
    # unmeasured sense-voice-small stays at the back.
    assert recommend._rank_whisper("cpu", languages=("en",))[-1] == "sense-voice-small"


def test_best_downloaded_language_prefers_covering_specialist():
    dm = _FakeDM(whisper={"parakeet-tdt-0.6b-v3", "small"})
    # language-blind: small ranks above the restricted parakeet
    assert recommend.best_downloaded(dm, translate=False, tier="cpu")[0] == "small"
    # with "en" known, parakeet leads the cpu tier
    got = recommend.best_downloaded(dm, translate=False, tier="cpu", languages=("en",))
    assert got[0] == "parakeet-tdt-0.6b-v3"
    # a language parakeet does not cover keeps the blind pick
    got = recommend.best_downloaded(dm, translate=False, tier="cpu", languages=("ja",))
    assert got[0] == "small"


def test_preset_for_tier_blind_matches_presets():
    for tier in ("cpu", "gpu_low", "gpu_high"):
        assert recommend.preset_for_tier(tier) == recommend.PRESETS[tier]


def test_preset_for_tier_promotes_sensevoice_for_cjk():
    for tier in ("cpu", "gpu_low", "gpu_high"):
        whisper, mt = recommend.preset_for_tier(tier, ("ja",))
        assert whisper == "sense-voice-small"
        assert mt == recommend._MT_PRESET[tier]


def test_preset_for_tier_english_and_european_pick_generalist():
    assert recommend.preset_for_tier("cpu", ("en",))[0] == "parakeet-tdt-0.6b-v3"
    assert recommend.preset_for_tier("cpu", ("de", "fr"))[0] == "parakeet-tdt-0.6b-v3"


def test_preset_for_tier_uncovered_mix_falls_back_to_generalist():
    assert recommend.preset_for_tier("cpu", ("ja", "de"))[0] == "small"
