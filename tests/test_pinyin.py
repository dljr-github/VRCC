"""Pinyin annotation for Chinese translations: which entries get a reading
line, what it contains, and the fail-soft paths (pypinyin absent or raising).
"""

from __future__ import annotations

import builtins

import pytest

from vrcc.translate import pinyin


def test_chinese_translation_gains_a_reading_line():
    result = pinyin.annotate([("Chinese Simplified", "你好")])
    assert len(result) == 1
    name, text = result[0]
    assert name == "Chinese Simplified"
    lines = text.split("\n")
    assert lines[0] == "你好"
    assert lines[1] == "nǐ hǎo"


def test_traditional_chinese_is_annotated_too():
    [(_, text)] = pinyin.annotate([("Chinese Traditional", "謝謝")])
    assert text.split("\n")[1] == "xiè xiè"


def test_punctuation_attaches_to_the_preceding_word():
    [(_, text)] = pinyin.annotate([("Chinese Simplified", "你好，世界！")])
    assert text.split("\n")[1] == "nǐ hǎo， shì jiè！"


def test_non_chinese_entries_pass_through_untouched():
    entries = [("Japanese", "こんにちは"), ("English", "hello")]
    assert pinyin.annotate(entries) == entries


def test_mixed_targets_annotate_only_the_chinese_one():
    result = pinyin.annotate(
        [("Japanese", "ありがとう"), ("Chinese Simplified", "谢谢")]
    )
    assert result[0] == ("Japanese", "ありがとう")
    assert "\n" in result[1][1]


def test_empty_chinese_text_stays_empty():
    assert pinyin.annotate([("Chinese Simplified", "  ")]) == [
        ("Chinese Simplified", "  ")
    ]


def test_chinese_entry_without_han_characters_passes_through():
    # A reading of latin-only text would just duplicate it (also what the
    # pipeline tests' fake MT engines emit for Chinese targets).
    entries = [("Chinese Simplified", "OK, see you!")]
    assert pinyin.annotate(entries) == entries


def test_unknown_display_name_passes_through():
    # A name outside the language registry must never crash the MT worker.
    assert pinyin.annotate([("Klingon", "abc")]) == [("Klingon", "abc")]


def test_missing_pypinyin_degrades_to_plain_text(monkeypatch):
    real_import = builtins.__import__

    def no_pypinyin(name, *args, **kwargs):
        if name == "pypinyin":
            raise ImportError("no module named pypinyin")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_pypinyin)
    monkeypatch.setattr(pinyin, "_missing_logged", False)
    assert pinyin.annotate([("Chinese Simplified", "你好")]) == [
        ("Chinese Simplified", "你好")
    ]


def test_reading_failure_keeps_plain_text(monkeypatch):
    def boom(text):
        raise RuntimeError("pypinyin exploded")

    monkeypatch.setattr(pinyin, "_reading", boom)
    assert pinyin.annotate([("Chinese Simplified", "你好")]) == [
        ("Chinese Simplified", "你好")
    ]


def test_pipeline_config_gates_annotation():
    # The flag lives on TranslateConfig and is opt-in (default off).
    from vrcc.core.config import TranslateConfig

    assert TranslateConfig().pinyin is False
