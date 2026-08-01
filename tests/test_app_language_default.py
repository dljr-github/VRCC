"""First-launch OS caption-language default (``vrcc.app``): a fresh config
(no file on disk, ``ConfigStore.missing_on_load``) gets ``stt.source_language``
pre-selected from the OS display-language preference via
:func:`vrcc.core.languages.match_caption_language`; an unmatched preference
list keeps the "English" default and an existing config is never rewritten.
"""

from __future__ import annotations

from pathlib import Path

from vrcc.app import _default_source_language
from vrcc.core.config import ConfigStore, default_paths


def _store(tmp_path: Path) -> ConfigStore:
    return ConfigStore(default_paths(portable=True, app_dir=tmp_path).config_file)


def test_matches_os_display_language(tmp_path):
    store = _store(tmp_path)
    store.load()
    _default_source_language(store.config, ["de-DE", "en-US"])
    assert store.config.stt.source_language == "German"


def test_walks_preference_order(tmp_path):
    store = _store(tmp_path)
    store.load()
    # An uncovered first preference falls through to the next, not to English.
    _default_source_language(store.config, ["gd-GB", "fr-FR", "en-US"])
    assert store.config.stt.source_language == "French"


def test_unmatched_preference_keeps_english(tmp_path):
    store = _store(tmp_path)
    store.load()
    _default_source_language(store.config, ["gd-GB", "C"])
    assert store.config.stt.source_language == "English"
    _default_source_language(store.config, [])
    assert store.config.stt.source_language == "English"


def test_existing_config_never_gets_the_default(tmp_path):
    # run() gates the default on ConfigStore.missing_on_load; mirroring that
    # gate here proves a saved config keeps the user's own choice.
    store = _store(tmp_path)
    store.config.stt.source_language = "Japanese"
    store.save_now()

    reloaded = _store(tmp_path)
    reloaded.load()
    if reloaded.missing_on_load:
        _default_source_language(reloaded.config, ["de-DE"])
    assert reloaded.config.stt.source_language == "Japanese"
