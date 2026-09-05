"""Case-table tests for :func:`vrcc.translate.punctuation.normalize`.

No model, no fixtures: every case here is a plain string in, string out. The
strings marked "observed" are decoder output recorded on 2026-08-30 from
nllb-600M-int8 and m2m100-418M-int8 on CPU int8 with the shipped defaults.
"""

from __future__ import annotations

import pytest

from vrcc.core.languages import LANGUAGES, get
from vrcc.translate.punctuation import normalize

# --------------------------------------------------------------------------
# Converts
# --------------------------------------------------------------------------

_CONVERTS = [
    ("Japanese", "田中さんが外で待っています.", "田中さんが外で待っています。"),
    ("Japanese", "一, 二, 三, 四, 五.", "一、二、三、四、五。"),
    ("Japanese", "はい,わかりました.", "はい、わかりました。"),
    ("Japanese", "「ありがとう」.", "「ありがとう」。"),
    ("Japanese", "コーヒー, お願いします.", "コーヒー、お願いします。"),
    ("Japanese", "おはよう, 元気?", "おはよう、元気?"),
    # Observed, nllb-600M-int8.
    (
        "Japanese",
        "リンゴやオレンジと梨を買った.安かった.",
        "リンゴやオレンジと梨を買った。安かった。",
    ),
    # A clause separator, which is what U+FF0C is for. A list of coordinate
    # nouns would take the enumeration comma U+3001 instead, and normalize
    # cannot tell the two apart, so it never emits U+3001 for Chinese.
    ("Chinese Simplified", "我很好, 谢谢你.", "我很好，谢谢你。"),
    ("Chinese Traditional", "我很好, 謝謝你.", "我很好，謝謝你。"),
]


@pytest.mark.parametrize("lang, text, expected", _CONVERTS)
def test_converts(lang, text, expected):
    assert normalize(text, get(lang)) == expected


# --------------------------------------------------------------------------
# Converts through a closing bracket or quote
#
# Both checkpoints quote with ASCII " as well as the fullwidth and curly
# forms. A closer carries no script, so the mark is judged against whatever
# the bracket closes on.
# --------------------------------------------------------------------------

_THROUGH_CLOSERS = [
    # Observed, nllb-600M-int8.
    ("Chinese Simplified", '他说"是的",然后离开了.', '他说"是的"，然后离开了。'),
    # Observed, m2m100-418M-int8.
    ("Chinese Simplified", "他说“是”,然后他离开了。", "他说“是”，然后他离开了。"),
    ("Japanese", "これは(テスト).", "これは(テスト)。"),
    ("Japanese", "（はい）.", "（はい）。"),
]


@pytest.mark.parametrize("lang, text, expected", _THROUGH_CLOSERS)
def test_converts_through_a_closer(lang, text, expected):
    assert normalize(text, get(lang)) == expected


def test_closer_over_latin_does_not_convert():
    # The closer is transparent, not a licence: the character it closes on
    # still decides.
    assert normalize('he said "yes".', get("Japanese")) == 'he said "yes".'


# --------------------------------------------------------------------------
# Must NOT convert
# --------------------------------------------------------------------------

_UNCHANGED = [
    ("Japanese", "3.5 kg です"),
    ("Japanese", "1,000円"),
    ("Japanese", "e.g. これ"),
    ("Japanese", "https://example.com/a.b"),
    # An ASCII ellipsis reads better than three ideographic full stops.
    ("Japanese", "です..."),
    # A period followed straight by ASCII letters or digits is an extension
    # or a domain label, not a sentence end.
    ("Japanese", "ファイル名はテスト.txtです"),
    ("Japanese", "日本.jp"),
    ("Japanese", "写真.pngを送った"),
    # Every script outside the table, Korean included: modern Korean writes
    # the ASCII marks. A CJK anchor in the text, or the case passes whatever
    # the table lookup returns.
    ("English", "日本語, です."),
    ("Korean", "안녕하세요, 반갑습니다."),
    ("Korean", "漢字, です."),
    # No character for the mark to attach to.
    ("Japanese", ""),
    ("Japanese", "."),
    ("Japanese", ","),
]


@pytest.mark.parametrize("lang, text", _UNCHANGED)
def test_unchanged(lang, text):
    assert normalize(text, get(lang)) == text


def test_final_period_preceded_by_latin_stays_ascii_but_comma_converts():
    text = "VRChatが好き, really."
    assert normalize(text, get("Japanese")) == "VRChatが好き、really."


def test_every_language_outside_the_table_passes_through():
    # Keyed on the FLORES script subtag, so this is the pin on Latin, Cyrillic,
    # Hangul and the rest resolving to no entry rather than to a default. The
    # text has to carry a CJK anchor or the assertion holds whatever the table
    # returns, and adding a Latn entry would go unnoticed.
    text = "こんにちは, 世界."
    for lang in LANGUAGES.values():
        if lang.display in ("Japanese", "Chinese Simplified", "Chinese Traditional"):
            continue
        assert normalize(text, lang) == text, lang.display


# --------------------------------------------------------------------------
# A mark next to one the target script already uses is left alone
#
# The input is a checkpoint that mixes conventions within one hypothesis, so
# an ideographic mark abutting an ASCII one is a shape to expect. Converting
# it would double the glyph.
# --------------------------------------------------------------------------

_NO_DOUBLING = [
    ("Japanese", "です。."),
    ("Japanese", "です.。"),
    ("Japanese", "はい、,いいえ"),
    ("Chinese Simplified", "很好，."),
]


@pytest.mark.parametrize("lang, text", _NO_DOUBLING)
def test_adjacent_ideographic_mark_is_left_alone(lang, text):
    assert normalize(text, get(lang)) == text


def test_normalize_is_idempotent_over_the_case_tables():
    for lang, text, _ in _CONVERTS:
        once = normalize(text, get(lang))
        assert normalize(once, get(lang)) == once


# --------------------------------------------------------------------------
# Space absorption
# --------------------------------------------------------------------------

def test_single_trailing_space_after_comma_is_absorbed():
    assert normalize("はい, いいえ", get("Japanese")) == "はい、いいえ"


def test_no_space_to_absorb_is_the_common_case():
    # What the decoder actually hands over: MtTokenizer.decode collapses
    # whitespace, and the checkpoints write the mark tight against the word.
    assert normalize("タナカは外で待ってる,大丈夫?", get("Japanese")) == (
        "タナカは外で待ってる、大丈夫?"
    )


# --------------------------------------------------------------------------
# Range boundaries
# --------------------------------------------------------------------------

def test_ideographic_space_is_not_a_cjk_predecessor():
    # U+3000 IDEOGRAPHIC SPACE is whitespace, not something a mark attaches
    # to, so it is left out of the range that starts at U+3001.
    text = "あ　."
    assert normalize(text, get("Japanese")) == text


def test_fullwidth_latin_letter_is_not_a_cjk_predecessor():
    # unicodedata.east_asian_width matches fullwidth Latin too, which is why
    # the ranges are written out by hand rather than derived from it.
    text = "Ａ."
    assert normalize(text, get("Japanese")) == text


def test_fullwidth_digit_is_not_a_cjk_predecessor():
    text = "０."
    assert normalize(text, get("Japanese")) == text


def test_halfwidth_katakana_dakuten_is_a_cjk_predecessor():
    # U+FF9E and U+FF9F end every voiced halfwidth syllable, so a range that
    # stops at U+FF9D misses most halfwidth katakana words.
    assert normalize("ﾃﾚﾋﾞ.", get("Japanese")) == "ﾃﾚﾋﾞ。"


def test_astral_ideograph_is_a_cjk_predecessor():
    assert normalize("\U00020000.", get("Japanese")) == "\U00020000。"


def test_extension_a_ideograph_is_a_cjk_predecessor():
    assert normalize("\u3400.", get("Chinese Simplified")) == "\u3400。"


def test_compatibility_ideograph_is_a_cjk_predecessor():
    assert normalize("\uf900.", get("Chinese Simplified")) == "\uf900。"


def test_absorbed_space_does_not_push_a_mark_into_an_ellipsis():
    # normalize eats the space after a converted mark, so the neighbour test
    # has to look past it: otherwise です. ... lands on です。...
    assert normalize("です. ...", get("Japanese")) == "です. ..."
    assert normalize("ア, ,z", get("Japanese")) == "ア, ,z"


def test_latin_inside_a_quote_keeps_the_mark_ascii():
    # The anchor walks through the closer to whatever it closes on, so the
    # quoted word's script decides. A Latin word leaves the comma ASCII while
    # the sentence-final period still converts: the conservative direction,
    # since treating every closer as CJK would rewrite English sentences.
    assert normalize('他说"OK",然后走了.', get("Chinese Simplified")) == (
        '他说"OK",然后走了。'
    )


def test_mark_right_after_an_opening_bracket_does_not_convert():
    # The bracket has just opened: there is nothing before the mark for it
    # to attach to, even though an opener falls inside the CJK punctuation
    # block that is_cjk otherwise treats as a valid anchor.
    text = "「.」"
    assert normalize(text, get("Japanese")) == text


def test_mark_inside_a_closer_converts_whatever_follows_the_closer():
    # The doubling rule stops at the closer: a mark on the far side of 」
    # sits outside the quote, so the one inside still converts. The outer
    # ASCII period anchors, through the closer, on an ASCII period, which is
    # no CJK character, and stays as it was.
    assert normalize("です.」。", get("Japanese")) == "です。」。"
    assert normalize("です.」.", get("Japanese")) == "です。」."


def test_a_mark_after_any_cjk_closer_anchors_on_what_it_closes():
    # Every closing bracket is transparent, not only the six the checkpoints
    # usually emit: the anchor is the character inside, whichever bracket.
    assert normalize("他说〖OK〗.", get("Chinese Simplified")) == "他说〖OK〗."
    assert normalize("他说〖是〗.", get("Chinese Simplified")) == "他说〖是〗。"


def test_hangul_compatibility_jamo_and_hangul_syllable_both_leave_the_mark_ascii():
    # Korean is absent from _MARKS, so neither anchors a conversion; before
    # is_cjk excluded Hangul Compatibility Jamo, the jamo case converted
    # while the syllable case did not, splitting one script across the
    # predicate. Both are unchanged now.
    assert normalize("ㄱ.", get("Japanese")) == "ㄱ."
    assert normalize("가.", get("Japanese")) == "가."
