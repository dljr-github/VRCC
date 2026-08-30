"""Case-table tests for :func:`vrcc.translate.punctuation.normalize`.

No model, no fixtures: every case here is a plain string in, string out.
"""

from __future__ import annotations

import pytest

from vrcc.core.languages import get
from vrcc.translate.punctuation import normalize

# --------------------------------------------------------------------------
# Converts, target Japanese
# --------------------------------------------------------------------------

_JAPANESE_CONVERTS = [
    ("田中さんが外で待っています.", "田中さんが外で待っています。"),
    ("一, 二, 三, 四, 五.", "一、二、三、四、五。"),
    ("はい,わかりました.", "はい、わかりました。"),
    ("「ありがとう」.", "「ありがとう」。"),
    ("コーヒー, お願いします.", "コーヒー、お願いします。"),
    ("おはよう, 元気?", "おはよう、元気?"),
]


@pytest.mark.parametrize("text, expected", _JAPANESE_CONVERTS)
def test_converts_target_japanese(text, expected):
    assert normalize(text, get("Japanese")) == expected


# --------------------------------------------------------------------------
# Converts, target Chinese Simplified
# --------------------------------------------------------------------------

def test_converts_target_chinese_simplified():
    text = "我买了苹果, 橙子和梨."
    expected = "我买了苹果，橙子和梨。"
    assert normalize(text, get("Chinese Simplified")) == expected


# --------------------------------------------------------------------------
# Must NOT convert
# --------------------------------------------------------------------------

def test_not_converted_preceded_by_digit_period():
    assert normalize("3.5 kg です", get("Japanese")) == "3.5 kg です"


def test_not_converted_preceded_by_digit_comma():
    assert normalize("1,000円", get("Japanese")) == "1,000円"


def test_not_converted_preceded_by_latin_abbreviation():
    assert normalize("e.g. これ", get("Japanese")) == "e.g. これ"


def test_not_converted_preceded_by_latin_url():
    text = "https://example.com/a.b"
    assert normalize(text, get("Japanese")) == text


def test_not_converted_run_of_periods_stays_ascii_ellipsis():
    assert normalize("です...", get("Japanese")) == "です..."


def test_final_period_preceded_by_latin_stays_ascii_but_comma_converts():
    text = "VRChatが好き, really."
    expected = "VRChatが好き、really."
    assert normalize(text, get("Japanese")) == expected


def test_target_not_in_set_english_stays_unchanged():
    text = "Hello, world."
    assert normalize(text, get("English")) == text


def test_korean_target_excluded_stays_unchanged():
    text = "안녕하세요, 반갑습니다."
    assert normalize(text, get("Korean")) == text


def test_empty_string_returns_empty_string():
    assert normalize("", get("Japanese")) == ""


def test_lone_period_has_no_preceding_character():
    assert normalize(".", get("Japanese")) == "."


def test_lone_comma_has_no_preceding_character():
    assert normalize(",", get("Japanese")) == ","


# --------------------------------------------------------------------------
# Space absorption
# --------------------------------------------------------------------------

def test_single_trailing_space_after_comma_is_absorbed():
    assert normalize("はい, いいえ", get("Japanese")) == "はい、いいえ"


def test_run_of_spaces_after_comma_loses_only_one():
    assert normalize("はい,  いいえ", get("Japanese")) == "はい、 いいえ"
