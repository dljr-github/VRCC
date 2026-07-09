"""Qt-side i18n glue: load Qt's own translations for the chosen UI language.

Separate from ``vrcc.i18n`` so the catalog machinery stays Qt-free; only
:func:`vrcc.app.run` imports this, after the QApplication exists.
"""

from __future__ import annotations

import logging

logger = logging.getLogger("vrcc.i18n.qt")

# Qt owns no reference to an installed QTranslator; keep ours alive here or
# the built-in dialog buttons snap back to English mid-session.
_QT_TRANSLATORS: list = []

# Our UI language codes vs the locale names Qt's bundled catalogs use. Only
# genuine exceptions belong here; codes like "pt-BR" already yield "pt_BR" via
# the default "-"->"_" transform below.
_QT_LOCALE_NAMES = {"zh-Hans": "zh_CN", "zh-Hant": "zh_TW"}


def apply_ui_language(app, configured: str) -> str:
    """Resolve ``configured`` (``"auto"`` follows the OS locale) into a
    supported UI language, activate it for :func:`vrcc.i18n.tr`, and install
    Qt's own translations. Returns the resolved code. Called once by
    :func:`vrcc.app.run`, before any widget is built (the language is
    restart-applied, like the theme)."""
    from PySide6.QtCore import QLocale

    from vrcc.i18n import resolve_ui_language, set_language

    system = QLocale.system()
    # uiLanguages() is the user's ordered DISPLAY-language preference (what
    # Windows calls the display language); name() is the regional-format
    # locale, which can legitimately differ — a Japanese display language
    # with English (US) number/date formats must still get a Japanese UI.
    preferred = list(system.uiLanguages()) or [system.name()]
    code = resolve_ui_language(configured, preferred)
    set_language(code)
    logger.info("UI language: %s", code)
    install_qt_translations(app, code)
    return code


def install_qt_translations(app, ui_language: str) -> None:
    """Load Qt's base translations (standard-button texts like Yes/No/Cancel)
    for ``ui_language``. Best-effort: a missing catalog leaves those buttons
    English while the app's own strings stay translated. Never raises."""
    if ui_language == "en":
        return
    try:
        from PySide6.QtCore import QLibraryInfo, QLocale, QTranslator

        locale = QLocale(_QT_LOCALE_NAMES.get(ui_language, ui_language.replace("-", "_")))
        translations_dir = QLibraryInfo.path(QLibraryInfo.LibraryPath.TranslationsPath)
        translator = QTranslator()
        if translator.load(locale, "qtbase", "_", translations_dir):
            app.installTranslator(translator)
            _QT_TRANSLATORS.append(translator)
        else:
            logger.info("no Qt base translations for %s", ui_language)
    except Exception:  # noqa: BLE001 -- missing Qt translations must never block startup
        logger.warning("could not install Qt translations", exc_info=True)
