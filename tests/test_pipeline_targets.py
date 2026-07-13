"""Tests for :mod:`vrcc.core.pipeline` -- MT target selection: source
resolution under "auto", skipping a target equal to the resolved source,
and what the chatbox receives when the original is hidden.
"""

from __future__ import annotations

import time

from vrcc.audio.segmenter import SegFinal, SegSpeculative
from vrcc.core.config import AppConfig, OscConfig, SttConfig, TranslateConfig
from vrcc.core.events import PhraseTranslated
from vrcc.core.languages import get as get_lang
from vrcc.osc.chatbox import format_message

from .conftest import FakeStt, collect, make_pipeline, make_result, running, sample


def _wait_until(predicate, timeout: float = 2.0, interval: float = 0.005) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return bool(predicate())


def test_auto_source_language_resolves_from_detected_whisper_code():
    # Target must differ from the detected language or the MT job filters
    # it out and the call under inspection never happens.
    cfg = AppConfig(
        stt=SttConfig(source_language="auto"),
        translate=TranslateConfig(targets=["English"]),
    )
    env = make_pipeline(config=cfg, stt=FakeStt(result=make_result(language="ja")))
    with running(env.pipeline):
        env.pipeline._on_seg_event(SegFinal(utterance_id=1, samples=sample()))
        assert _wait_until(lambda: len(env.mt.calls) == 1)
    _text, src, _targets = env.mt.calls[0]
    assert src == get_lang("Japanese")  # ja -> Japanese


def test_auto_detected_language_matching_a_target_is_skipped():
    cfg = AppConfig(
        stt=SttConfig(source_language="auto"),
        translate=TranslateConfig(targets=["English", "Japanese"]),
    )
    env = make_pipeline(config=cfg, stt=FakeStt(result=make_result(language="en")))
    translated = collect(env.bus, PhraseTranslated)
    with running(env.pipeline):
        env.pipeline._on_seg_event(SegFinal(utterance_id=1, samples=sample()))
        assert _wait_until(lambda: len(translated) == 1)
    _text, _src, targets = env.mt.calls[0]
    assert targets == [get_lang("Japanese")]
    assert translated[0].translations == (("Japanese", "Japanese:hello world"),)
    assert env.chatbox.submits[0] == ("hello world\nJapanese:hello world", 1)


def test_auto_detected_language_as_only_target_sends_original_without_mt():
    cfg = AppConfig(
        stt=SttConfig(source_language="auto"),
        translate=TranslateConfig(targets=["English"]),
    )
    env = make_pipeline(config=cfg, stt=FakeStt(result=make_result(language="en")))
    translated = collect(env.bus, PhraseTranslated)
    s = sample()
    with running(env.pipeline):
        env.pipeline._on_seg_event(SegSpeculative(utterance_id=1, samples=s))
        env.pipeline._on_seg_event(SegFinal(utterance_id=1, samples=s))
        assert _wait_until(lambda: len(env.chatbox.submits) == 1)
        assert _wait_until(lambda: len(translated) == 1)
        # The empty-targets branch must still resolve the MT-owned typing
        # entry: prune_orphans exempts it, so nothing else ever clears it.
        assert _wait_until(lambda: env.chatbox.typing[-1] is False)
    assert env.chatbox.typing[0] is True
    assert env.mt.calls == []
    assert translated[0].translations == ()
    assert env.chatbox.submits[0] == ("hello world", 1)


def test_hidden_original_still_reaches_source_language_readers():
    # include_original=False hides the original on the strength of every
    # target carrying a translation; the skipped source-matching target is
    # served by the original text, which must therefore stay in the message.
    cfg = AppConfig(
        stt=SttConfig(source_language="auto"),
        translate=TranslateConfig(targets=["English", "Japanese"]),
        osc=OscConfig(include_original=False),
    )
    env = make_pipeline(config=cfg, stt=FakeStt(result=make_result(language="en")))
    translated = collect(env.bus, PhraseTranslated)
    with running(env.pipeline):
        env.pipeline._on_seg_event(SegFinal(utterance_id=1, samples=sample()))
        assert _wait_until(lambda: len(env.chatbox.submits) == 1)
        assert _wait_until(lambda: len(translated) == 1)
    assert translated[0].translations == (("Japanese", "Japanese:hello world"),)
    original, submitted, uid = env.chatbox.messages[0]
    assert uid == 1
    assert submitted == [
        ("English", "hello world"),
        ("Japanese", "Japanese:hello world"),
    ]
    assert format_message(original, submitted, cfg.osc) == (
        "hello world\nJapanese:hello world"
    )


def test_hidden_original_with_only_the_source_target_sends_original():
    # Same setting, all targets filtered: consistent with the partial case
    # above, the original still lands in the chatbox.
    cfg = AppConfig(
        stt=SttConfig(source_language="auto"),
        translate=TranslateConfig(targets=["English"]),
        osc=OscConfig(include_original=False),
    )
    env = make_pipeline(config=cfg, stt=FakeStt(result=make_result(language="en")))
    with running(env.pipeline):
        env.pipeline._on_seg_event(SegFinal(utterance_id=1, samples=sample()))
        assert _wait_until(lambda: len(env.chatbox.submits) == 1)
    original, submitted, _uid = env.chatbox.messages[0]
    assert submitted == [("English", "hello world")]
    assert format_message(original, submitted, cfg.osc) == "hello world"


def test_auto_chinese_traditional_target_still_translates():
    # "auto" resolves whisper "zh" to Chinese Simplified, so only the
    # Simplified target is redundant; Traditional is a script conversion.
    cfg = AppConfig(
        stt=SttConfig(source_language="auto"),
        translate=TranslateConfig(targets=["Chinese Simplified", "Chinese Traditional"]),
    )
    env = make_pipeline(config=cfg, stt=FakeStt(result=make_result(language="zh")))
    with running(env.pipeline):
        env.pipeline._on_seg_event(SegFinal(utterance_id=1, samples=sample()))
        assert _wait_until(lambda: len(env.mt.calls) == 1)
    _text, src, targets = env.mt.calls[0]
    assert src == get_lang("Chinese Simplified")
    assert targets == [get_lang("Chinese Traditional")]


def test_explicit_source_matching_a_target_is_skipped_too():
    # The GUI excludes an explicit source from the target combos; a hand
    # edited config must still never translate a language into itself.
    cfg = AppConfig(
        stt=SttConfig(source_language="English"),
        translate=TranslateConfig(targets=["English", "Japanese"]),
    )
    env = make_pipeline(config=cfg)
    with running(env.pipeline):
        env.pipeline._on_seg_event(SegFinal(utterance_id=1, samples=sample()))
        assert _wait_until(lambda: len(env.mt.calls) == 1)
    _text, _src, targets = env.mt.calls[0]
    assert targets == [get_lang("Japanese")]
