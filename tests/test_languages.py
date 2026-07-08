import pytest

from vrcc.core.languages import LANGUAGES, Language, get


def test_languages_has_around_30_entries():
    assert 25 <= len(LANGUAGES) <= 40


def test_english_is_first_for_gui_display_order():
    assert next(iter(LANGUAGES)) == "English"


def test_every_entry_has_non_empty_codes():
    for display, lang in LANGUAGES.items():
        assert isinstance(lang, Language)
        assert lang.display == display
        assert lang.whisper, f"{display} missing whisper code"
        assert lang.nllb, f"{display} missing nllb code"
        assert lang.m2m100, f"{display} missing m2m100 code"


def test_entries_are_frozen_dataclass_instances():
    lang = LANGUAGES["English"]
    with pytest.raises(Exception):
        lang.whisper = "xx"  # frozen dataclass must reject mutation


def test_get_japanese_returns_expected_codes():
    lang = get("Japanese")
    assert lang.whisper == "ja"
    assert lang.nllb == "jpn_Jpan"
    assert lang.m2m100 == "ja"


def test_get_matches_dict_lookup():
    for display in LANGUAGES:
        assert get(display) == LANGUAGES[display]


def test_get_unknown_name_raises_key_error():
    with pytest.raises(KeyError):
        get("Klingon")


def test_get_unknown_name_error_message_is_helpful():
    with pytest.raises(KeyError) as excinfo:
        get("Klingon")
    message = str(excinfo.value)
    assert "Klingon" in message


@pytest.mark.parametrize(
    "display,whisper,nllb,m2m100",
    [
        ("English", "en", "eng_Latn", "en"),
        ("Japanese", "ja", "jpn_Jpan", "ja"),
        ("Korean", "ko", "kor_Hang", "ko"),
        ("Chinese Simplified", "zh", "zho_Hans", "zh"),
        ("Chinese Traditional", "zh", "zho_Hant", "zh"),
        ("Spanish", "es", "spa_Latn", "es"),
        ("French", "fr", "fra_Latn", "fr"),
        ("German", "de", "deu_Latn", "de"),
        ("Portuguese", "pt", "por_Latn", "pt"),
        ("Russian", "ru", "rus_Cyrl", "ru"),
        ("Italian", "it", "ita_Latn", "it"),
        ("Indonesian", "id", "ind_Latn", "id"),
        ("Thai", "th", "tha_Thai", "th"),
        ("Vietnamese", "vi", "vie_Latn", "vi"),
        ("Arabic", "ar", "arb_Arab", "ar"),
        ("Hindi", "hi", "hin_Deva", "hi"),
        ("Turkish", "tr", "tur_Latn", "tr"),
        ("Polish", "pl", "pol_Latn", "pl"),
        ("Dutch", "nl", "nld_Latn", "nl"),
        ("Ukrainian", "uk", "ukr_Cyrl", "uk"),
        ("Filipino", "tl", "tgl_Latn", "tl"),
        ("Malay", "ms", "zsm_Latn", "ms"),
        ("Swedish", "sv", "swe_Latn", "sv"),
        ("Norwegian", "no", "nob_Latn", "no"),
        ("Danish", "da", "dan_Latn", "da"),
        ("Finnish", "fi", "fin_Latn", "fi"),
        ("Czech", "cs", "ces_Latn", "cs"),
        ("Greek", "el", "ell_Grek", "el"),
        ("Hebrew", "he", "heb_Hebr", "he"),
        ("Romanian", "ro", "ron_Latn", "ro"),
    ],
)
def test_seed_entries_match_brief(display, whisper, nllb, m2m100):
    lang = get(display)
    assert lang.whisper == whisper
    assert lang.nllb == nllb
    assert lang.m2m100 == m2m100
