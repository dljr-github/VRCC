"""Unit tests for the shared friendly-model-name helper (Task 7 polish)."""

from vrcc.gui.model_labels import (
    mt_display_name, whisper_display_name, model_blurb, fmt_size, _fmt_size,
)


def test_mt_display_name_known_ids():
    assert mt_display_name("nllb-600M-int8") == "NLLB 600M — balanced"
    assert mt_display_name("nllb-1.3B-int8") == "NLLB 1.3B — higher quality"
    assert mt_display_name("nllb-3.3B-int8") == "NLLB 3.3B — best quality (large)"
    assert mt_display_name("m2m100-418M-int8") == "M2M100 418M — small"
    assert mt_display_name("m2m100-1.2B-int8") == "M2M100 1.2B"
    assert mt_display_name("madlad400-3b") == "MADLAD-400 3B"


def test_mt_display_name_falls_back_to_id_for_unknown():
    assert mt_display_name("some-future-model-id") == "some-future-model-id"


def test_whisper_display_name_uses_label_and_falls_back():
    from vrcc.stt.registry import WHISPER_MODELS
    some = next(iter(WHISPER_MODELS))
    assert whisper_display_name(some) == WHISPER_MODELS[some].label
    assert whisper_display_name("nope-xyz") == "nope-xyz"


def test_model_blurb_mt_flags_non_commercial_for_nllb():
    blurb = model_blurb("mt", "nllb-600M-int8")
    assert "non-commercial" in blurb.lower()
    assert "MB" in blurb or "GB" in blurb


def test_model_blurb_marks_english_only_whisper():
    from vrcc.stt.registry import WHISPER_MODELS
    eng_only = [i for i, s in WHISPER_MODELS.items() if s.english_only]
    if eng_only:
        assert "english only" in model_blurb("whisper", eng_only[0]).lower()


def test_model_blurb_unknown_is_empty():
    assert model_blurb("mt", "nope") == ""
    assert model_blurb("whisper", "nope") == ""


def test_fmt_size_is_public_and_old_name_is_an_alias():
    assert fmt_size(1620) == "~1.6 GB"
    assert fmt_size(647) == "647 MB"
    assert _fmt_size is fmt_size  # old private name kept as an alias


def test_whisper_blurb_lead_ins_are_all_distinct():
    """Design review: lead-ins like "Fastest, lower accuracy" vs "Fast, lower
    accuracy" read as near-duplicates. Every voice model must get a distinct
    descriptor so the ladder actually differentiates them."""
    from vrcc.stt.registry import WHISPER_MODELS

    lead_ins = [model_blurb("whisper", mid).split(" · ")[0] for mid in WHISPER_MODELS]
    assert len(set(lead_ins)) == len(lead_ins)


def test_model_blurb_marks_parakeet_language_restriction():
    blurb = model_blurb("whisper", "parakeet-tdt-0.6b-v3")
    assert "european languages only" in blurb.lower()
    assert "english only" not in blurb.lower()
    assert "MB" in blurb or "GB" in blurb


def test_parakeet_display_name():
    assert whisper_display_name("parakeet-tdt-0.6b-v3") == "Parakeet v3 (European languages)"
