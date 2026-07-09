"""Interface translations: a tiny gettext-style string catalog. Zero Qt.

`tr("English source text")` returns that string's translation in the current
UI language, set once at startup by :func:`set_language` (the language is a
restart-applied setting, like the theme). English source strings are the
catalog keys; a missing key or catalog falls back to the English text, so
`tr` never raises and English needs no catalog file.

Catalogs are JSON files next to this module (``ja.json``, ``pt-BR.json``,
...) mapping the exact source string to its translation. Placeholders use
``str.format`` names (``"Downloading {model}"``) and must survive
translation verbatim (guarded by tests/test_i18n.py).

`tr_noop` marks strings built at module import time (constant dicts,
tooltips) for extraction without translating them yet -- call `tr()` on the
value at the point of use, after startup has set the language.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger("vrcc.i18n")

# UI language code -> native display name (the Settings picker shows these).
# Codes are BCP-47-ish; "en" is the source language and has no catalog file.
UI_LANGUAGES: dict[str, str] = {
    "en": "English",
    "ja": "日本語",
    "ko": "한국어",
    "zh-Hans": "简体中文",
    "zh-Hant": "繁體中文",
    "es": "Español",
    "fr": "Français",
    "de": "Deutsch",
    "it": "Italiano",
    "pt-BR": "Português (Brasil)",
    "ru": "Русский",
    "uk": "Українська",
    "pl": "Polski",
    "nl": "Nederlands",
    "tr": "Türkçe",
    "id": "Bahasa Indonesia",
    "vi": "Tiếng Việt",
    "th": "ไทย",
}

# Region/script spellings that don't reduce to a plain language-code match.
_LOCALE_OVERRIDES = {
    "zh": "zh-Hans",
    "zh-cn": "zh-Hans",
    "zh-sg": "zh-Hans",
    "zh-hans": "zh-Hans",
    "zh-tw": "zh-Hant",
    "zh-hk": "zh-Hant",
    "zh-mo": "zh-Hant",
    "zh-hant": "zh-Hant",
    "pt": "pt-BR",
    "pt-br": "pt-BR",
    "pt-pt": "pt-BR",
}

_current_language = "en"
_catalog: dict[str, str] = {}

# Case-insensitive UI-language lookup, derived once from the module constant.
_BY_LOWER = {code.lower(): code for code in UI_LANGUAGES}

# Errors str.format may raise on a mangled placeholder: a missing named field
# (KeyError), missing positional (IndexError), bad format spec (ValueError), or
# a field whose attribute/subscript access is invalid for the given value
# (AttributeError/TypeError). tr() catches all of them to honor "never raises".
_FORMAT_ERRORS = (KeyError, IndexError, ValueError, AttributeError, TypeError)


def match_locale(name: str | None) -> str | None:
    """Map a locale name ("ja_JP.UTF-8", "zh_TW", "pt") to a supported UI
    language code, or ``None`` if nothing matches."""
    if not name:
        return None
    norm = name.replace("_", "-").split(".")[0].strip().lower()
    if not norm:
        return None
    while norm:
        if norm in _LOCALE_OVERRIDES:
            return _LOCALE_OVERRIDES[norm]
        if norm in _BY_LOWER:
            return _BY_LOWER[norm]
        if "-" not in norm:
            break
        norm = norm.rsplit("-", 1)[0]
    return None


def resolve_ui_language(
    configured: str, system_locale: str | list[str] | None = None
) -> str:
    """Turn the ``gui.ui_language`` config value plus the OS locale into a
    supported code. ``"auto"`` (or anything unrecognized) follows the system
    setting; English when that isn't supported either.

    ``system_locale`` may be a single locale name or the OS's ordered
    display-language preference list (``QLocale.system().uiLanguages()``);
    the first supported entry wins, so a user whose preferences are
    ["gd-GB", "fr-FR"] gets French rather than English."""
    if configured != "auto":
        matched = match_locale(configured)
        if matched is not None:
            return matched
        logger.warning("gui.ui_language %r is not a supported UI language; using auto", configured)
    candidates = [system_locale] if isinstance(system_locale, str) else (system_locale or [])
    for name in candidates:
        matched = match_locale(name)
        if matched is not None:
            return matched
    return "en"


def _load_catalog(code: str) -> dict[str, str]:
    """Read ``<code>.json`` next to this module, tolerantly: a missing or
    malformed catalog (or non-string entries) degrades to English."""
    path = Path(__file__).resolve().parent / f"{code}.json"
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        logger.warning("no catalog for UI language %r; falling back to English", code)
        return {}
    except (OSError, ValueError):
        # ValueError covers json.JSONDecodeError and a non-UTF-8 file's
        # UnicodeDecodeError (both subclasses); a corrupt catalog must degrade
        # to English, never crash startup.
        logger.warning("could not read catalog %s; falling back to English", path, exc_info=True)
        return {}
    if not isinstance(raw, dict):
        logger.warning("catalog %s is not a JSON object; falling back to English", path)
        return {}
    return {k: v for k, v in raw.items() if isinstance(k, str) and isinstance(v, str)}


def set_language(code: str) -> None:
    """Switch `tr` to ``code``, loading its catalog. Called once at startup
    (before any GUI construction); unknown codes degrade to English."""
    global _current_language, _catalog
    if code not in UI_LANGUAGES:
        logger.warning("unknown UI language %r; using English", code)
        code = "en"
    _current_language = code
    _catalog = {} if code == "en" else _load_catalog(code)


def current_language() -> str:
    return _current_language


def tr(text: str, **kwargs) -> str:
    """Translate ``text`` (an English source string) into the current UI
    language, formatting any ``{name}`` placeholders with ``kwargs``.

    Never raises: unknown strings fall back to the English source, and a
    translation whose placeholders were mangled falls back to formatting the
    English source instead.
    """
    translated = _catalog.get(text, text)
    if not kwargs:
        return translated
    try:
        return translated.format(**kwargs)
    except _FORMAT_ERRORS:
        if translated is not text:
            logger.warning(
                "catalog entry for %r has broken placeholders in %r; using English",
                text,
                _current_language,
            )
            try:
                return text.format(**kwargs)
            except _FORMAT_ERRORS:
                pass
        logger.warning("could not format %r with %r", text, sorted(kwargs))
        return translated


def tr_noop(text: str) -> str:
    """Mark ``text`` for catalog extraction without translating it here.

    Use on strings evaluated at module import time (constant dicts, module-
    level tooltips), where the UI language isn't set yet -- then pass the
    value through :func:`tr` at the point of use.
    """
    return text
