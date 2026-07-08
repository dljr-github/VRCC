"""Tests for the OSC chatbox message shaping: ``format_message`` (joining
translations) and ``fit_chatbox`` (fitting text to the 144-char limit).
"""

import pytest

from vrcc.core.bus import EventBus
from vrcc.core.config import OscConfig
from vrcc.osc.chatbox import CHATBOX_LIMIT, ChatboxSender, fit_chatbox, format_message


def make_cfg(**overrides) -> OscConfig:
    return OscConfig(**overrides)


def make_idle_sender(cfg: OscConfig) -> ChatboxSender:
    """A `ChatboxSender` whose worker thread is never started -- enough to
    call `submit()` and inspect `._queue`, with no need for a fake clock."""
    return ChatboxSender(cfg, EventBus(), client_factory=lambda ip, port: object())


# -- format_message ----------------------------------------------------------


def test_format_message_include_original_joins_with_separator():
    cfg = make_cfg(include_original=True, translation_separator="\n")
    result = format_message("hello", [("JP", "konnichiwa"), ("FR", "bonjour")], cfg)
    assert result == "hello\nkonnichiwa\nbonjour"


def test_format_message_exclude_original_joins_translations_only():
    cfg = make_cfg(include_original=False, translation_separator=" | ")
    result = format_message("hello", [("JP", "konnichiwa"), ("FR", "bonjour")], cfg)
    assert result == "konnichiwa | bonjour"


def test_format_message_single_translation_exclude_original_is_just_the_text():
    cfg = make_cfg(include_original=False)
    result = format_message("hello", [("JP", "konnichiwa")], cfg)
    assert result == "konnichiwa"


def test_format_message_no_translations_returns_original_regardless_of_include_original():
    cfg_true = make_cfg(include_original=True)
    cfg_false = make_cfg(include_original=False)
    assert format_message("  hello  ", [], cfg_true) == "hello"
    assert format_message("  hello  ", [], cfg_false) == "hello"


def test_format_message_strips_only_the_overall_result():
    cfg = make_cfg(include_original=True, translation_separator="\n")
    # Interior whitespace from an individual piece is preserved; only the
    # leading/trailing whitespace of the final joined string is stripped.
    result = format_message("hello", [("JP", " konnichiwa ")], cfg)
    assert result == "hello\n konnichiwa"


# -- fit_chatbox ---------------------------------------------------------


def test_fit_chatbox_empty_text_returns_empty_list():
    assert fit_chatbox("", "truncate") == []
    assert fit_chatbox("", "split") == []
    assert fit_chatbox("", "send") == []


def test_fit_chatbox_send_mode_passes_through_unchanged_even_over_limit():
    text = "x" * 300
    assert fit_chatbox(text, "send") == [text]


def test_fit_chatbox_truncate_under_limit_is_unchanged():
    text = "short message"
    assert fit_chatbox(text, "truncate") == [text]


def test_fit_chatbox_truncate_at_exactly_the_limit_is_unchanged():
    text = "x" * CHATBOX_LIMIT
    assert fit_chatbox(text, "truncate") == [text]


def test_fit_chatbox_truncate_over_limit_gets_ellipsis():
    text = "x" * 300
    result = fit_chatbox(text, "truncate")
    assert len(result) == 1
    assert len(result[0]) == CHATBOX_LIMIT
    assert result[0] == "x" * (CHATBOX_LIMIT - 1) + "…"


def test_fit_chatbox_split_chunks_all_within_limit_and_preserve_words():
    words = [f"word{i}" for i in range(80)]
    text = " ".join(words)
    result = fit_chatbox(text, "split")
    assert len(result) > 1
    assert all(len(chunk) <= CHATBOX_LIMIT for chunk in result)
    # Every original word appears, in order, undamaged, across the chunks.
    assert " ".join(result).split() == words


def test_fit_chatbox_split_single_word_over_limit_is_hard_split():
    text = "a" * 300
    result = fit_chatbox(text, "split")
    assert len(result) == 3  # 144 + 144 + 12
    assert all(len(chunk) <= CHATBOX_LIMIT for chunk in result)
    assert "".join(result) == text


def test_fit_chatbox_split_mixed_words_and_one_oversized_word():
    text = "short words then " + ("z" * 300) + " and more words after"
    result = fit_chatbox(text, "split")
    assert all(len(chunk) <= CHATBOX_LIMIT for chunk in result)
    assert "".join(result).replace(" ", "") == text.replace(" ", "")


def test_fit_chatbox_unknown_mode_raises():
    with pytest.raises(ValueError):
        fit_chatbox("hello", "bogus")


# -- ChatboxSender.submit()'s per-chunk delay_after ---------------------


def test_submit_split_message_sets_delay_after_on_all_but_last_chunk():
    sender = make_idle_sender(make_cfg(overflow="split", split_delay_s=1.7))

    long_text = " ".join(f"word{i}" for i in range(60))
    sender.submit(long_text, 1)

    items = list(sender._queue)
    assert len(items) > 1  # sanity: fixture actually splits into several chunks
    delays = [item[3] for item in items]
    assert delays[:-1] == [1.7] * (len(delays) - 1)
    assert delays[-1] == 0.0


def test_submit_single_chunk_message_has_zero_delay_after():
    sender = make_idle_sender(make_cfg(overflow="send", split_delay_s=1.7))

    sender.submit("short message", 1)

    items = list(sender._queue)
    assert len(items) == 1
    assert items[0][3] == 0.0
