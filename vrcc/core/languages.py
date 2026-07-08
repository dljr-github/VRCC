"""Static registry of supported languages and their model-specific codes.

Maps a display name to a `Language` with `whisper`/`nllb`/`m2m100` codes (the
`nllb` FLORES-200 token is fed verbatim to NLLB -- must match exactly).
Insertion order is GUI display order (English first). Zero Qt.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Language:
    display: str
    whisper: str
    nllb: str
    m2m100: str


def _lang(display: str, whisper: str, nllb: str, m2m100: str) -> Language:
    return Language(display=display, whisper=whisper, nllb=nllb, m2m100=m2m100)


LANGUAGES: dict[str, Language] = {
    lang.display: lang
    for lang in (
        _lang("English", "en", "eng_Latn", "en"),
        _lang("Japanese", "ja", "jpn_Jpan", "ja"),
        _lang("Korean", "ko", "kor_Hang", "ko"),
        _lang("Chinese Simplified", "zh", "zho_Hans", "zh"),
        _lang("Chinese Traditional", "zh", "zho_Hant", "zh"),
        _lang("Spanish", "es", "spa_Latn", "es"),
        _lang("French", "fr", "fra_Latn", "fr"),
        _lang("German", "de", "deu_Latn", "de"),
        _lang("Portuguese", "pt", "por_Latn", "pt"),
        _lang("Russian", "ru", "rus_Cyrl", "ru"),
        _lang("Italian", "it", "ita_Latn", "it"),
        _lang("Indonesian", "id", "ind_Latn", "id"),
        _lang("Thai", "th", "tha_Thai", "th"),
        _lang("Vietnamese", "vi", "vie_Latn", "vi"),
        _lang("Arabic", "ar", "arb_Arab", "ar"),
        _lang("Hindi", "hi", "hin_Deva", "hi"),
        _lang("Turkish", "tr", "tur_Latn", "tr"),
        _lang("Polish", "pl", "pol_Latn", "pl"),
        _lang("Dutch", "nl", "nld_Latn", "nl"),
        _lang("Ukrainian", "uk", "ukr_Cyrl", "uk"),
        _lang("Filipino", "tl", "tgl_Latn", "tl"),
        _lang("Malay", "ms", "zsm_Latn", "ms"),
        _lang("Swedish", "sv", "swe_Latn", "sv"),
        _lang("Norwegian", "no", "nob_Latn", "no"),
        _lang("Danish", "da", "dan_Latn", "da"),
        _lang("Finnish", "fi", "fin_Latn", "fi"),
        _lang("Czech", "cs", "ces_Latn", "cs"),
        _lang("Greek", "el", "ell_Grek", "el"),
        _lang("Hebrew", "he", "heb_Hebr", "he"),
        _lang("Romanian", "ro", "ron_Latn", "ro"),
    )
}


def get(display: str) -> Language:
    """Look up a `Language` by display name; raises `KeyError` (naming the bad
    value and pointing at `LANGUAGES.keys()`) if unknown.
    """
    try:
        return LANGUAGES[display]
    except KeyError:
        raise KeyError(
            f"Unknown language display name: {display!r}. "
            f"Valid names come from vrcc.core.languages.LANGUAGES.keys()."
        ) from None
