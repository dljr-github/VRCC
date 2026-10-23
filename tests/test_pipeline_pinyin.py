"""Pinyin in the pipeline: the caption log always gets the reading line, the
chatbox only when the whole message still fits VRChat's limit.
"""

from __future__ import annotations

from vrcc.audio.segmenter import SegFinal
from vrcc.core.config import AppConfig, OscConfig, TranslateConfig
from vrcc.core.events import PhraseTranslated
from vrcc.osc.chatbox import CHATBOX_LIMIT

from .conftest import FakeMt, collect, make_pipeline, running, sample, wait_until

_LONG = "我今天下午要去超市买一些水果和蔬菜，你要不要跟我一起去？我们可以顺便喝杯咖啡。"


# 136 chars with its reading: fits alone, overflows behind "hello world\n".
_EDGE = "谢谢你今天帮我这么多忙，我真的很感激你，下次一定请你吃饭吧。"


def _run(text: str, pinyin: bool, osc: OscConfig | None = None):
    """(caption-log translations, chatbox translations) for one phrase."""
    cfg = AppConfig(
        translate=TranslateConfig(targets=["Chinese Simplified"], pinyin=pinyin),
        osc=osc or OscConfig(),
    )
    env = make_pipeline(config=cfg, mt=FakeMt([("Chinese Simplified", text)]))
    translated = collect(env.bus, PhraseTranslated)
    with running(env.pipeline):
        env.pipeline._on_seg_event(SegFinal(utterance_id=1, samples=sample()))
        assert wait_until(lambda: len(env.chatbox.messages) == 1)
        assert wait_until(lambda: len(translated) == 1)
    return translated[0].translations, env.chatbox.messages[0][1]


def test_short_translation_carries_pinyin_to_log_and_chatbox():
    log, chatbox = _run("你好", pinyin=True)
    assert log == (("Chinese Simplified", "你好\nnǐ hǎo"),)
    assert chatbox == list(log)


def test_reading_that_would_overflow_stays_out_of_the_chatbox():
    log, chatbox = _run(_LONG, pinyin=True)
    assert log[0][1].startswith(_LONG + "\n")  # caption log keeps the reading
    assert chatbox == [("Chinese Simplified", _LONG)]


def test_overflow_check_counts_the_original_only_when_it_is_shown():
    _log, shown = _run(_EDGE, pinyin=True)
    assert shown == [("Chinese Simplified", _EDGE)]
    _log, hidden = _run(_EDGE, pinyin=True, osc=OscConfig(include_original=False))
    assert hidden[0][1].startswith(_EDGE + "\n")
    assert len(hidden[0][1]) <= CHATBOX_LIMIT


def test_pinyin_off_leaves_both_untouched():
    log, chatbox = _run("你好", pinyin=False)
    assert log == (("Chinese Simplified", "你好"),)
    assert chatbox == [("Chinese Simplified", "你好")]
