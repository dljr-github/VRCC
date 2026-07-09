"""Tests for the OSC chatbox message shaping: ``format_message`` (joining
translations), ``fit_chatbox`` (fitting text to the 144-char limit) and
``fit_message`` (per-language balanced splitting).
"""

import pytest

from vrcc.core.bus import EventBus
from vrcc.core.config import OscConfig
from vrcc.osc.chatbox import (
    CHATBOX_LIMIT,
    ChatboxSender,
    fit_chatbox,
    fit_message,
    format_message,
)


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


# -- fit_message -----------------------------------------------------------


def test_fit_message_within_limit_is_one_part_identical_to_format_message():
    cfg = make_cfg(overflow="split")
    translations = [("JP", "konnichiwa")]
    assert fit_message("hello", translations, cfg) == [
        format_message("hello", translations, cfg)
    ]


def test_fit_message_split_keeps_both_languages_in_every_part():
    cfg = make_cfg(overflow="split")
    caption = " ".join(f"cap{i}" for i in range(20))
    translated = " ".join(f"tr{i}" for i in range(26))
    joined = format_message(caption, [("FR", translated)], cfg)
    assert len(joined) > CHATBOX_LIMIT  # sanity: fixture actually overflows

    parts = fit_message(caption, [("FR", translated)], cfg)

    assert len(parts) == 2
    assert all(len(part) <= CHATBOX_LIMIT for part in parts)
    # Every part shows BOTH languages: a caption line and a translation line.
    lines = [part.split("\n") for part in parts]
    assert all(len(pair) == 2 and pair[0] and pair[1] for pair in lines)
    # Each language reconstructs exactly, in order, from its slices.
    assert " ".join(pair[0] for pair in lines).split() == caption.split()
    assert " ".join(pair[1] for pair in lines).split() == translated.split()
    # Slices are balanced, not front-loaded.
    sizes = [len(part) for part in parts]
    assert max(sizes) - min(sizes) <= 60


def test_fit_message_split_spaceless_translation_reconstructs_by_characters():
    cfg = make_cfg(overflow="split")
    caption = "This is a test caption"
    japanese = "あいうえおかきくけこ" * 20  # 200 chars, no spaces
    parts = fit_message(caption, [("JP", japanese)], cfg)

    assert len(parts) >= 2
    assert all(len(part) <= CHATBOX_LIMIT for part in parts)
    lines = [part.split("\n") for part in parts]
    assert all(len(pair) == 2 for pair in lines)
    assert " ".join(pair[0] for pair in lines).split() == caption.split()
    assert "".join(pair[1] for pair in lines) == japanese


def test_fit_message_exclude_original_shows_translations_only():
    cfg = make_cfg(overflow="split", include_original=False)
    japanese = "あいうえおかきくけこ" * 20
    parts = fit_message("secretcaption", [("JP", japanese)], cfg)

    assert len(parts) >= 2
    assert all(len(part) <= CHATBOX_LIMIT for part in parts)
    assert all("secretcaption" not in part for part in parts)
    assert "".join(parts) == japanese


def test_fit_message_split_omits_exhausted_caption_from_later_parts():
    cfg = make_cfg(overflow="split")
    caption = "Okay"
    translated = " ".join(f"long{i}" for i in range(60))
    parts = fit_message(caption, [("FR", translated)], cfg)

    assert len(parts) >= 3
    assert all(len(part) <= CHATBOX_LIMIT for part in parts)
    # The short caption runs out before the translation does: later parts
    # carry the translation alone, with no leading separator or blank line.
    assert "\n" not in parts[-1]
    for part in parts:
        assert part == part.strip()
        assert "\n\n" not in part
    # Both texts still reconstruct exactly across the parts.
    caption_lines = [p.split("\n")[0] for p in parts if "\n" in p]
    assert "".join(caption_lines) == caption
    translated_lines = [p.split("\n")[-1] for p in parts]
    assert " ".join(translated_lines).split() == translated.split()


def test_fit_message_split_never_chops_short_caption_words():
    # A caption with fewer words than the slice count must stay word-based
    # (empty trailing slices), not fall into the character path that would
    # show fragments like "the" / "re" as successive chatbox parts.
    cfg = make_cfg(overflow="split")
    caption = "Hi there"
    parts = fit_message(caption, [("JP", "あ" * 400)], cfg)

    assert len(parts) >= 3
    assert all(len(part) <= CHATBOX_LIMIT for part in parts)
    caption_lines = [p.split("\n")[0] for p in parts if "\n" in p]
    for line in caption_lines:
        assert all(word in caption.split() for word in line.split())
    assert " ".join(caption_lines).split() == caption.split()
    # The spaceless translation still reconstructs character-exactly.
    assert "".join(p.split("\n")[-1] for p in parts) == "あ" * 400


def test_fit_message_truncate_and_send_defer_to_fit_chatbox():
    translations = [("JP", "x" * 120)]
    for mode in ("truncate", "send"):
        cfg = make_cfg(overflow=mode)
        joined = format_message("y" * 100, translations, cfg)
        assert len(joined) > CHATBOX_LIMIT  # sanity: fixture actually overflows
        assert fit_message("y" * 100, translations, cfg) == fit_chatbox(joined, mode)


# -- ChatboxSender.submit_message() ------------------------------------------


def test_submit_message_split_queues_parts_with_delays_and_no_truncated_flag():
    cfg = make_cfg(overflow="split", split_delay_s=1.7)
    sender = make_idle_sender(cfg)
    caption = " ".join(f"cap{i}" for i in range(20))
    translations = [("FR", " ".join(f"tr{i}" for i in range(26)))]
    expected = fit_message(caption, translations, cfg)
    assert len(expected) > 1  # sanity: fixture actually splits

    sender.submit_message(caption, translations, 7)

    items = list(sender._queue)
    assert [item[0] for item in items] == expected
    assert all(item[1] == 7 for item in items)
    assert all(item[2] is False for item in items)  # split loses nothing
    delays = [item[3] for item in items]
    assert delays[:-1] == [1.7] * (len(delays) - 1)
    assert delays[-1] == 0.0


def test_submit_message_truncate_flags_over_limit_as_truncated():
    cfg = make_cfg(overflow="truncate")
    sender = make_idle_sender(cfg)
    translations = [("JP", "x" * 120)]

    sender.submit_message("y" * 100, translations, 3)

    items = list(sender._queue)
    assert len(items) == 1
    joined = format_message("y" * 100, translations, cfg)
    assert items[0][0] == fit_chatbox(joined, "truncate")[0]
    assert items[0][2] is True


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
