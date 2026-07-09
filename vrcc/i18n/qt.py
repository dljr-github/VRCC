"""Qt-side i18n glue: load Qt's own translations for the chosen UI language.

Separate from ``vrcc.i18n`` so the catalog machinery stays Qt-free; only
:func:`vrcc.app.run` imports this, after the QApplication exists.
"""

from __future__ import annotations

import logging

logger = logging.getLogger("vrcc.i18n.qt")

# Qt owns no reference to an installed QTranslator; keep ours alive here or
# the built-in dialog buttons snap back to English mid-session. The next
# install_qt_translations call removes these before installing its own, so a
# live language switch never stacks catalogs.
_QT_TRANSLATORS: list = []

# Our UI language codes vs the locale names Qt's bundled catalogs use. Only
# genuine exceptions belong here; codes like "pt-BR" already yield "pt_BR" via
# the default "-"->"_" transform below.
_QT_LOCALE_NAMES = {"zh-Hans": "zh_CN", "zh-Hant": "zh_TW"}


def system_locale_preference() -> list[str]:
    """The user's ordered DISPLAY-language preference: ``uiLanguages()`` (what
    Windows calls the display language), falling back to the regional-format
    ``name()``, which can legitimately differ (a Japanese display language
    with English (US) number/date formats must still read as Japanese). Shared
    by :func:`apply_ui_language` and the first-launch caption-language default
    so both read the same OS signal."""
    from PySide6.QtCore import QLocale

    system = QLocale.system()
    return list(system.uiLanguages()) or [system.name()]


def apply_ui_language(app, configured: str) -> str:
    """Resolve ``configured`` (``"auto"`` follows the OS locale) into a
    supported UI language, activate it for :func:`vrcc.i18n.tr`, and install
    Qt's own translations. Returns the resolved code. Called once by
    :func:`vrcc.app.run`, before any widget is built (the language is
    restart-applied, like the theme)."""
    from vrcc.i18n import resolve_ui_language, set_language

    code = resolve_ui_language(configured, system_locale_preference())
    set_language(code)
    logger.info("UI language: %s", code)
    install_qt_translations(app, code)
    return code


def install_qt_translations(app, ui_language: str) -> None:
    """Load Qt's base translations (standard-button texts like Yes/No/Cancel)
    for ``ui_language``, replacing whatever a previous call installed: without
    the removal, a live switch back to English keeps the old catalog active
    for the rest of the session. Best-effort: a missing catalog leaves those
    buttons English while the app's own strings stay translated. Never raises."""
    try:
        from PySide6.QtCore import QCoreApplication, QLibraryInfo, QLocale, QTranslator

        while _QT_TRANSLATORS:
            QCoreApplication.removeTranslator(_QT_TRANSLATORS.pop())
        if ui_language == "en":
            return
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
