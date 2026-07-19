from vrcc.core.sentences import ends_sentence


def test_terminal_punctuation_true():
    assert ends_sentence("I am going now.", 2)
    assert ends_sentence("Really?", 1)
    assert ends_sentence("Stop!", 1)
    assert ends_sentence("well then...", 2)


def test_cjk_terminal_punctuation_true():
    assert ends_sentence("もう行きます。", 1)
    assert ends_sentence("本当？", 1)


def test_cjk_terminal_punctuation_true_at_default_min_words():
    # A CJK sentence is one whitespace token; the word-count fragment guard
    # must be waived for space-less scripts, not just when min_words=1.
    assert ends_sentence("もう行きます。", 2)
    assert ends_sentence("这是一个句子。", 2)
    assert ends_sentence("本当ですか？", 2)


def test_hindi_danda_true_at_default_min_words():
    assert ends_sentence("यह एक वाक्य है।", 2)


def test_arabic_question_mark_true_at_default_min_words():
    assert ends_sentence("هل أنت بخير؟", 2)


def test_greek_question_mark_true_at_default_min_words():
    assert ends_sentence("Πώς είσαι;", 2)


def test_latin_semicolon_is_not_a_terminal():
    assert not ends_sentence("buy milk;", 2)


def test_trailing_quote_or_bracket_allowed():
    assert ends_sentence('He said "go."', 2)
    assert ends_sentence("(done.)", 1)


def test_trailing_guillemet_allowed():
    assert ends_sentence('He said «go.»', 2)


def test_no_terminal_punctuation_false():
    assert not ends_sentence("I am going", 2)
    assert not ends_sentence("and then we", 2)


def test_min_words_guard():
    # Abbreviation-like single token with a period must not count as a sentence.
    assert not ends_sentence("Mr.", 2)
    assert not ends_sentence("3.14", 2)
