"""Tests for the STT/MT model registry data integrity (Whisper/MT specs)
and the ``lang_token`` mapping.
"""

from __future__ import annotations

import pytest

from vrcc.core.languages import get
from vrcc.stt.registry import WHISPER_MODELS, WhisperSpec
from vrcc.translate.registry import MT_MODELS, MtModelSpec, lang_token

KNOWN_FAMILIES = {"nllb", "m2m100", "madlad"}
WHISPER_TIERS = {"fast", "balanced", "accurate"}


# --------------------------------------------------------------------------
# Whisper registry integrity
# --------------------------------------------------------------------------

def test_whisper_key_equals_spec_id():
    for key, spec in WHISPER_MODELS.items():
        assert isinstance(spec, WhisperSpec)
        assert spec.id == key


def test_whisper_sizes_are_positive():
    for spec in WHISPER_MODELS.values():
        assert spec.size_mb > 0, f"{spec.id} has non-positive size"


def test_whisper_tiers_are_known():
    for spec in WHISPER_MODELS.values():
        assert spec.tier in WHISPER_TIERS, f"{spec.id} has unknown tier {spec.tier!r}"


def test_whisper_labels_non_empty():
    for spec in WHISPER_MODELS.values():
        assert spec.label


def test_whisper_english_only_is_bool():
    for spec in WHISPER_MODELS.values():
        assert isinstance(spec.english_only, bool)


def test_whisper_expected_models_present():
    expected = {
        "tiny",
        "base",
        "small",
        "medium",
        "large-v3",
        "large-v3-turbo",
        "distil-large-v3.5",
        "distil-small.en",
    }
    assert expected <= set(WHISPER_MODELS)


def test_whisper_distil_models_are_english_only():
    assert WHISPER_MODELS["distil-large-v3.5"].english_only is True
    assert WHISPER_MODELS["distil-small.en"].english_only is True


def test_whisper_multilingual_models_not_english_only():
    for mid in ("tiny", "base", "small", "medium", "large-v3"):
        assert WHISPER_MODELS[mid].english_only is False


def test_whisper_spec_is_frozen():
    spec = WHISPER_MODELS["small"]
    with pytest.raises(Exception):
        spec.size_mb = 1  # type: ignore[misc]


# --------------------------------------------------------------------------
# MT registry integrity
# --------------------------------------------------------------------------

def test_mt_key_equals_spec_id():
    for key, spec in MT_MODELS.items():
        assert isinstance(spec, MtModelSpec)
        assert spec.id == key


def test_mt_sizes_are_positive():
    for spec in MT_MODELS.values():
        assert spec.size_mb > 0, f"{spec.id} has non-positive size"


def test_mt_families_are_known():
    for spec in MT_MODELS.values():
        assert spec.family in KNOWN_FAMILIES, f"{spec.id} family {spec.family!r}"


def test_mt_licenses_non_empty():
    for spec in MT_MODELS.values():
        assert spec.license, f"{spec.id} missing license"


def test_mt_spm_file_and_repo_non_empty():
    for spec in MT_MODELS.values():
        assert spec.spm_file, f"{spec.id} missing spm_file"
        assert spec.repo, f"{spec.id} missing repo"


def test_mt_prefix_side_valid():
    for spec in MT_MODELS.values():
        assert spec.prefix_side in {"source", "target"}


def test_mt_expected_models_present():
    expected = {
        "nllb-600M-int8",
        "nllb-1.3B-int8",
        "nllb-3.3B-int8",
        "m2m100-418M-int8",
        "m2m100-1.2B-int8",
        "madlad400-3b",
    }
    assert expected <= set(MT_MODELS)


def test_mt_spec_is_frozen():
    spec = MT_MODELS["nllb-600M-int8"]
    with pytest.raises(Exception):
        spec.repo = "x"  # type: ignore[misc]


# --------------------------------------------------------------------------
# lang_token
# --------------------------------------------------------------------------

def test_lang_token_nllb_is_verbatim():
    assert lang_token("nllb", get("Japanese")) == "jpn_Jpan"
    assert lang_token("nllb", get("English")) == "eng_Latn"


def test_lang_token_m2m100_wraps_code():
    assert lang_token("m2m100", get("Japanese")) == "__ja__"
    assert lang_token("m2m100", get("English")) == "__en__"


def test_lang_token_madlad_uses_angle_bracket():
    assert lang_token("madlad", get("Japanese")) == "<2ja>"
    assert lang_token("madlad", get("French")) == "<2fr>"


def test_lang_token_unknown_family_raises_value_error():
    with pytest.raises(ValueError):
        lang_token("bogus", get("English"))
