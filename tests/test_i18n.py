"""UI-language plumbing: locale resolution, tr() semantics, catalog
integrity (every catalog fully translates exactly the extracted source
strings, placeholders intact), and the restart-applied language flow
(Settings picker writes config; a non-English language changes built text).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

import vrcc.i18n as i18n
from vrcc.i18n import (
    UI_LANGUAGES,
    match_locale,
    resolve_ui_language,
    set_language,
    tr,
    tr_noop,
)
from vrcc.i18n.extract import extract_source_strings, placeholder_tokens

_I18N_DIR = Path(i18n.__file__).resolve().parent
_CATALOGS = sorted(_I18N_DIR.glob("*.json"))


@pytest.fixture(autouse=True)
def _english_after_each_test():
    # tr() reads module state; never leak a language into other tests.
    yield
    set_language("en")


# -- locale resolution -------------------------------------------------------


@pytest.mark.parametrize(
    "locale,expected",
    [
        ("ja_JP", "ja"),
        ("ja_JP.UTF-8", "ja"),
        ("ko_KR", "ko"),
        ("zh_CN", "zh-Hans"),
        ("zh_SG", "zh-Hans"),
        ("zh_TW", "zh-Hant"),
        ("zh_HK", "zh-Hant"),
        ("pt_BR", "pt-BR"),
        ("pt_PT", "pt-BR"),
        ("en_US", "en"),
        ("de_AT", "de"),
        ("C", None),
        ("", None),
        (None, None),
        ("xx_YY", None),
    ],
)
def test_match_locale(locale, expected):
    assert match_locale(locale) == expected


def test_resolve_explicit_setting_wins_over_system_locale():
    assert resolve_ui_language("ko", "ja_JP") == "ko"
    # Tolerant spellings of an explicit setting still resolve.
    assert resolve_ui_language("KO_kr", "ja_JP") == "ko"


def test_resolve_auto_follows_system_locale_else_english():
    assert resolve_ui_language("auto", "ja_JP") == "ja"
    assert resolve_ui_language("auto", "C") == "en"
    assert resolve_ui_language("auto", None) == "en"


def test_resolve_unknown_setting_degrades_to_auto():
    assert resolve_ui_language("klingon", "fr_FR") == "fr"
    assert resolve_ui_language("klingon", None) == "en"


def test_resolve_auto_walks_display_language_preference_in_order():
    # QLocale.system().uiLanguages() shapes: BCP-47, ordered by preference.
    # The first SUPPORTED entry wins -- an unsupported first choice must fall
    # through to the next preference, not to English.
    assert resolve_ui_language("auto", ["gd-GB", "fr-FR", "en-US"]) == "fr"
    assert resolve_ui_language("auto", ["ja-JP", "ja", "en-US"]) == "ja"
    assert resolve_ui_language("auto", ["en-US", "ja-JP"]) == "en"
    # Script-qualified Chinese as Windows/Qt report it.
    assert resolve_ui_language("auto", ["zh-Hans-CN", "zh-CN"]) == "zh-Hans"
    assert resolve_ui_language("auto", ["zh-Hant-TW", "zh-TW"]) == "zh-Hant"
    assert resolve_ui_language("auto", []) == "en"
    assert resolve_ui_language("auto", ["C", "POSIX"]) == "en"
    # An explicit setting still beats the whole preference list.
    assert resolve_ui_language("ko", ["ja-JP", "en-US"]) == "ko"


# -- tr() semantics ----------------------------------------------------------


def test_tr_is_identity_in_english():
    assert tr("Start captioning") == "Start captioning"
    assert tr("Downloading {model_id}: {pct}%", model_id="m", pct=7) == "Downloading m: 7%"
    assert tr_noop("Marked") == "Marked"


def test_set_language_translates_and_falls_back(monkeypatch):
    # Stub the catalog loader so the test owns the content without touching the
    # real, version-controlled ja.json (a mid-test crash or xdist parallelism
    # would otherwise corrupt the shipped file for other tests).
    monkeypatch.setattr(
        i18n,
        "_load_catalog",
        lambda code: {"Hello": "こんにちは", "Hi {name}": "やあ {mangled}"},
    )
    set_language("ja")
    assert tr("Hello") == "こんにちは"
    # Untranslated strings fall back to the English source.
    assert tr("Not in the catalog") == "Not in the catalog"
    # Mangled placeholders in a translation fall back to formatted English.
    assert tr("Hi {name}", name="X") == "Hi X"


def test_unknown_language_degrades_to_english():
    set_language("xx")
    assert i18n.current_language() == "en"
    assert tr("Start captioning") == "Start captioning"


# -- catalog integrity -------------------------------------------------------


def test_every_ui_language_has_a_catalog():
    expected = {f"{code}.json" for code in UI_LANGUAGES if code != "en"}
    present = {p.name for p in _CATALOGS}
    assert expected <= present, f"missing catalogs: {sorted(expected - present)}"
    assert present <= expected, f"stray catalogs: {sorted(present - expected)}"


@pytest.mark.parametrize("path", _CATALOGS, ids=lambda p: p.stem)
def test_catalog_matches_extracted_source_strings(path):
    source = set(extract_source_strings())
    assert source, "extractor found no tr()/tr_noop() strings under vrcc/"
    catalog = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(catalog, dict)
    keys = set(catalog)
    missing = source - keys
    stale = keys - source
    assert not missing, f"{path.name} lacks translations for: {sorted(missing)[:10]}"
    assert not stale, f"{path.name} has keys no source string uses: {sorted(stale)[:10]}"


def test_registry_labels_are_marked_for_translation():
    # whisper_display_name()/firstrun render WhisperSpec.label via tr(), but the
    # labels live in the Qt-free registry, so model_labels._WHISPER_LABEL_MARKERS
    # mirrors them for the extractor. Guard that mirror: a label the extractor
    # can't see never reaches a catalog and ships untranslated with green tests.
    from vrcc.stt.registry import WHISPER_MODELS

    source = set(extract_source_strings())
    labels = {spec.label for spec in WHISPER_MODELS.values()}
    missing = labels - source
    assert not missing, (
        "whisper labels not marked for translation (add to "
        f"_WHISPER_LABEL_MARKERS in vrcc/gui/model_labels.py): {sorted(missing)}"
    )


@pytest.mark.parametrize("path", _CATALOGS, ids=lambda p: p.stem)
def test_catalog_values_are_nonempty_with_intact_placeholders(path):
    catalog = json.loads(path.read_text(encoding="utf-8"))
    for key, value in catalog.items():
        assert isinstance(value, str) and value.strip(), f"{path.name}: empty value for {key!r}"
        # Tokens include the format spec ({seconds:.1f}) and Qt %-tokens (%p%),
        # so a translation that drops a precision spec or a progress token fails.
        assert placeholder_tokens(value) == placeholder_tokens(key), (
            f"{path.name}: placeholder mismatch for {key!r}: {value!r}"
        )
        # Inline markup (the caption log's explicit line break) must survive.
        if "<br/>" in key:
            assert "<br/>" in value, f"{path.name}: lost <br/> in {key!r}"


def test_apply_ui_language_resolves_and_activates():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    from vrcc.i18n.qt import apply_ui_language

    app = QApplication.instance() or QApplication([])
    assert apply_ui_language(app, "ja") == "ja"
    assert i18n.current_language() == "ja"
    # Unknown setting degrades via auto -> the OS locale or English; either
    # way it must land on a supported code and never raise.
    assert apply_ui_language(app, "klingon") in UI_LANGUAGES


def test_qt_base_translators_are_removed_on_language_switch(monkeypatch):
    # Live language switches reuse apply_ui_language; without removal the Qt
    # base translators stack, and ja -> en keeps Japanese Yes/No/Cancel
    # buttons for the rest of the session.
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtCore import QCoreApplication
    from PySide6.QtWidgets import QApplication

    import vrcc.i18n.qt as qt_mod
    from vrcc.i18n.qt import apply_ui_language

    app = QApplication.instance() or QApplication([])
    removed = []
    real_remove = QCoreApplication.removeTranslator

    def record_remove(translator):
        removed.append(translator)
        return real_remove(translator)

    monkeypatch.setattr(QCoreApplication, "removeTranslator", record_remove)

    apply_ui_language(app, "ja")
    installed = list(qt_mod._QT_TRANSLATORS)
    if not installed:
        pytest.skip("this PySide6 build ships no qtbase ja catalog")

    apply_ui_language(app, "ja")
    assert len(qt_mod._QT_TRANSLATORS) == 1  # re-applying replaces, never stacks

    apply_ui_language(app, "en")
    assert qt_mod._QT_TRANSLATORS == []
    # Every translator we ever installed went through removeTranslator.
    assert installed[0] in removed


# -- end-to-end: language reaches widgets ------------------------------------


def _ja_catalog() -> dict[str, str]:
    return json.loads((_I18N_DIR / "ja.json").read_text(encoding="utf-8"))


def test_main_window_builds_in_japanese(tmp_path):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    from vrcc.core.bus import EventBus
    from vrcc.core.config import ConfigStore
    from vrcc.gui.bridge import BusBridge

    QApplication.instance() or QApplication([])
    ja = _ja_catalog()
    expected = ja["Start captioning"]
    assert expected != "Start captioning"  # ja must actually translate it

    set_language("ja")
    try:
        # Import after set_language is NOT required (no import-time tr()),
        # which is exactly what this test should prove -- import first:
        from vrcc.gui.main_window import MainWindow

        # A fresh path (not os.devnull): the store's debounced save writes
        # <path>.json.tmp, and "nul.json.tmp" would land in the repo root.
        store = ConfigStore(tmp_path / "config.json")
        bridge = BusBridge(EventBus())

        class _Pipeline:
            captioning_enabled = False

            def set_captioning(self, value):
                pass

        window = MainWindow(bridge, store, _Pipeline(), lambda: None, lambda: None)
        try:
            assert window._captioning_btn.text() == expected
            assert window.windowTitle() == "VRCC"  # the brand stays
        finally:
            window.close()
            window.deleteLater()
            bridge.detach()
    finally:
        set_language("en")


def test_settings_language_picker_writes_config(tmp_path):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    from vrcc.core.config import ConfigStore
    from vrcc.gui.settings import SettingsDialog

    QApplication.instance() or QApplication([])
    store = ConfigStore(tmp_path / "config.json")
    store.load()
    dlg = SettingsDialog(store)
    try:
        combo = dlg._ui_language_combo
        # "Auto" first, then every supported language, by data code.
        codes = [combo.itemData(i) for i in range(combo.count())]
        assert codes[0] == "auto"
        assert codes[1:] == list(UI_LANGUAGES)
        # Native names label the entries so anyone can find their language.
        assert combo.itemText(codes.index("ja")) == "日本語"

        combo.setCurrentIndex(codes.index("ja"))
        assert store.config.gui.ui_language == "ja"
        # The restart banner is gone: a language change now rebuilds the window
        # on dialog close instead of deferring to a restart.
        assert not hasattr(dlg, "_restart_banner")
    finally:
        dlg.close()
        dlg.deleteLater()
