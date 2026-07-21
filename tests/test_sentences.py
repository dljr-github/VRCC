from vrcc.core.sentences import ends_sentence, split_sentences, followed_complete_sentences


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


def test_default_min_words_holds_a_mid_sentence_fragment():
    # A comma pause mid-sentence can leave a short, punctuated fragment behind
    # ("Hello there.") that reads as a complete sentence at a low word-count
    # gate. Raising the default gate to 3 holds it until the full clause
    # arrives, while a genuine short clause still injects.
    assert ends_sentence("Hello there.", 2)
    assert not ends_sentence("Hello there.", 3)
    assert ends_sentence("Hello there, how are you doing today?", 3)


def test_split_sentences_basic():
    assert split_sentences("Hello there. How are you? I am fine.") == [
        "Hello there.", "How are you?", "I am fine."]


def test_split_sentences_trailing_fragment():
    assert split_sentences("Hello there. I am test") == ["Hello there.", "I am test"]


def test_split_sentences_empty():
    assert split_sentences("") == []
    assert split_sentences("   ") == []


def test_followed_complete_excludes_last_and_fragments():
    # "How are you?" is complete but LAST -> excluded; "I am test" is a fragment.
    assert followed_complete_sentences("Hello there. How are you?", 2) == ["Hello there."]
    assert followed_complete_sentences("Hello there. I am test", 2) == ["Hello there."]
    # both non-last completes returned
    assert followed_complete_sentences("A sentence here. Another one here. Frag", 2) == [
        "A sentence here.", "Another one here."]


def test_followed_complete_min_words_gate():
    # "Mr." is not a real sentence; even followed, it is not returned.
    assert followed_complete_sentences("Mr. Smith went home.", 2) == []


def test_followed_complete_spaceless_script():
    # CJK: the word-count gate is waived; the first (followed) sentence returns.
    out = followed_complete_sentences("こんにちは。元気ですか", 2)
    assert out == ["こんにちは。"]
