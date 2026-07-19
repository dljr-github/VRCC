from vrcc.core.sentences import ends_sentence


def test_terminal_punctuation_true():
    assert ends_sentence("I am going now.", 2)
    assert ends_sentence("Really?", 1)
    assert ends_sentence("Stop!", 1)
    assert ends_sentence("well then...", 2)


def test_cjk_terminal_punctuation_true():
    assert ends_sentence("もう行きます。", 1)
    assert ends_sentence("本当？", 1)


def test_trailing_quote_or_bracket_allowed():
    assert ends_sentence('He said "go."', 2)
    assert ends_sentence("(done.)", 1)


def test_no_terminal_punctuation_false():
    assert not ends_sentence("I am going", 2)
    assert not ends_sentence("and then we", 2)


def test_min_words_guard():
    # Abbreviation-like single token with a period must not count as a sentence.
    assert not ends_sentence("Mr.", 2)
    assert not ends_sentence("3.14", 2)
