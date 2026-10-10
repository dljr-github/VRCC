"""The ordered digit instrument behind tools/bench_mt.py and the MT quality
gate. Pure text in, verdict out: no model is loaded."""

import json
from pathlib import Path

import pytest

from tools.mt_digits import (
    ADDED, DROPPED, KEPT, SPELLED, classify, compare, leading_dash,
    longest_run, merged, numbers,
)


@pytest.mark.parametrize(
    "src, out, src_lang, out_lang, expected",
    [
        ("I will be back in 5 minutes.", "5分で戻ります。", "en", "ja", KEPT),
        ("Okay 1, 2, 3, 4, 5, 6, 7.", "1, 2, 3, 4, 5, 6, 7.", "en", "en", KEPT),
        ("Year 2020", "２０２０年", "en", "ja", KEPT),
        ("pin 0000", "핀 0000", "en", "ko", KEPT),
        ("Call me at 555 1212.", "5551212に電話して", "en", "ja", KEPT),
        ("I will be back in 5 minutes.", "Volveré en cinco minutos.", "en", "es", SPELLED),
        ("Add 5 and 5, you get 10.", "5と5を足すと十になる", "en", "ja", SPELLED),
        ("I got 33", "Tengo treinta y tres", "en", "es", SPELLED),
        ("I got 33", "ich habe dreiunddreißig", "en", "de", SPELLED),
        ("I got 12", "나는 열두 개", "en", "ko", SPELLED),
        ("2:22", "두 시 이십이 분", "en", "ko", SPELLED),
        ("It's 4 4 4 in a row.", "連続で4です", "en", "ja", DROPPED),
        ("Okay 1, 2, 3.", "3, 2, 1", "en", "en", DROPPED),
        ("1 2 3", "123", "en", "en", DROPPED),
        ("pin 0000", "핀 영", "en", "ko", DROPPED),
        ("Okay 1, 2, 3.", "好的", "en", "zh", DROPPED),
        ("hello", "Hi 5", "en", "en", ADDED),
        ("hello", "こんにちは", "en", "ja", KEPT),
        ("4 4 4", "네 네 네", "ja", "ko", DROPPED),
        ("I have 1 cat", "我们一起走吧", "en", "zh", DROPPED),
        ("Wait 10 seconds", "十分に待って", "en", "ja", DROPPED),
        ("Count down 3, 2, 1", "세, 두, 하나", "en", "ko", SPELLED),
        ("I have 2 cats", "고양이가 두 마리 있어", "en", "ko", SPELLED),
        ("Chapter 7", "第七章", "en", "zh", SPELLED),
    ],
)
def test_classify(src, out, src_lang, out_lang, expected):
    assert classify(src, out, src_lang, out_lang) == expected


def test_a_set_comparison_would_have_passed_these():
    # Both outputs contain exactly the source's digit SET.
    assert set("444") == set("4")
    assert classify("4 4 4", "4", "en", "en") == DROPPED
    assert set("123") == set("321")
    assert classify("1 2 3", "3 2 1", "en", "en") == DROPPED


def test_a_dropped_repeat_is_counted_per_number():
    assert compare("4 4 4", "4", "en", "en")[DROPPED] == 2


def test_spoken_source_reads_number_words_only_when_asked():
    assert numbers("one one two", "en", words=False) == []
    assert [n.value for n in numbers("one one two", "en", words=True)] == [1, 1, 2]
    assert classify("one one two", "1 1 2", "en", "en", spoken_source=True) == SPELLED
    assert classify("one one two", "1 2", "en", "en", spoken_source=True) == DROPPED


def test_thousands_separators_and_fullwidth_digits_normalise():
    assert [n.digits for n in numbers("1,000 and １２")] == ["1000", "12"]
    assert [n.digits for n in numbers("1, 2, 3")] == ["1", "2", "3"]


def test_leading_zeros_keep_their_digits_and_carry_no_value():
    (num,) = numbers("0042")
    assert num.digits == "0042" and num.value is None


def test_merged_separates_joined_digits_from_lost_ones():
    assert merged("8 8 8 8", "8888")
    assert merged("9 9 9 9 9", "9 99 9 9")
    assert not merged("4 4 4", "4")
    assert not merged("hello", "hola")


def test_longest_run_flags_a_loop_and_not_prose():
    assert longest_run("はい" * 10) >= 4
    assert longest_run("I will be back in five minutes.") < 4


def test_leading_dash_only_when_the_source_has_none():
    assert leading_dash("Yeah.", "- Yeah.")
    assert not leading_dash("Yeah.", "Yeah.")
    assert not leading_dash("- Yeah.", "- Yeah.")


def test_corpus_is_well_formed():
    data = json.loads(
        (Path(__file__).resolve().parent.parent / "tools" / "mt_corpus.json")
        .read_text(encoding="utf-8")
    )
    assert {"en", "ja", "zh", "ko", "fr", "es", "de"} <= set(data)
    for groups in data.values():
        for texts in groups.values():
            assert texts and all(isinstance(t, str) and t.strip() for t in texts)
    for groups in data.values():
        assert set(groups) <= {
            "number", "repeat", "spoken", "interjection", "fragment", "sentence"
        }
