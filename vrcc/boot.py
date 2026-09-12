"""Bring Qt up first, show what is loading, then hand off to the app.

Distinct from :mod:`vrcc.core.startup`, which holds model-readiness helpers that
run()'s body calls. This module is what runs BEFORE run() exists.

Every heavy import in :mod:`vrcc.app`'s module scope happens after QApplication
here, so the window that reports progress exists before the slow part rather than
after it. The paths, the logging, the config store and the QApplication are built
once here and passed into run(), which skips building its own: setup_logging adds
a fresh file handler on every call, and two ConfigStore objects would leave run()
closing over a different one than the panel was themed from.
"""

from __future__ import annotations

import importlib
import logging

from PySide6.QtWidgets import QApplication

from vrcc.core.config import ConfigStore, default_paths
from vrcc.core.logs import setup_logging
from vrcc.core.progress import LogProgress, NoProgress

logger = logging.getLogger("vrcc.boot")

# vrcc.app's own module-scope import order. The speech group reaches
# onnxruntime through vrcc/audio/vad.py, and app.py carries a measured claim
# that setup_cuda_dlls is cheap only because onnxruntime is already loaded by
# then -- reordering this table would reorder that cost too.
_IMPORT_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("audio", ("vrcc.audio.source",)),
    ("hardware", ("vrcc.core.hardware",)),
    ("speech", ("vrcc.core.engine_stack",)),
    ("downloads", ("vrcc.download.manager",)),
    ("interface", (
        "vrcc.gui.main_window",
        "vrcc.gui.firstrun",
        "vrcc.gui.settings",
        "vrcc.gui.models_dialog",
        "vrcc.gui.bridge",
    )),
)


def _import_module(name: str):
    """A seam: tests replace this to prove the walk survives a broken group
    without actually breaking an import."""
    return importlib.import_module(name)


def _walk_imports(progress) -> None:
    """Name each phase before paying for it, so a launch that dies inside an
    import leaves the failing step as the log's last line. One broken group
    must not stop the app from starting: run() reports engine failures
    through its own UI, which never gets the chance if boot dies first."""
    for key, module_names in _IMPORT_GROUPS:
        try:
            progress.start(key)
            for name in module_names:
                _import_module(name)
        except Exception:  # noqa: BLE001 -- a reporter or an import failing must not sink the launch
            logger.warning("boot: %s group failed to import", key, exc_info=True)


def _run_app(**kwargs) -> int:
    """Hand off to the composition root. A seam so tests never build the real
    engine stack, the real window, or a real Qt event loop."""
    from vrcc.app import run

    return run(**kwargs)


class _Both:
    """Reports a phase change to the panel and the log together, and pumps
    the event loop so the panel actually paints it. The import walk runs
    synchronously on the GUI thread, and nothing under vrcc/ pumps events
    anywhere else -- without this call the user watches a blank rectangle for
    the whole walk, which is worse than no panel at all."""

    def __init__(self, panel, log: LogProgress, app: QApplication) -> None:
        self._panel = panel
        self._log = log
        self._app = app

    def start(self, key: str) -> None:
        self._panel.start(key)
        self._log.start(key)
        self._app.processEvents()

    def close(self) -> None:
        self._panel.close()
        self._log.close()


def boot(portable: bool = False, verbose: bool = False, guard=None) -> int:
    """Bring up logging, config and Qt, show progress through the heavy
    imports, then start the app. Returns the process exit code."""
    paths = default_paths(portable)
    setup_logging(paths.logs_dir, verbose)
    logger.info("VRCC starting (portable=%s)", portable)

    store = ConfigStore(paths.config_file)
    store.load()
    for warning in store.load_warnings:
        logger.warning("config: %s", warning)

    # A launch that cannot draw a progress bar must still start the app, so
    # everything from here through building the panel falls back to a
    # log-only reporter rather than aborting the launch.
    progress = NoProgress()
    try:
        app = QApplication.instance() or QApplication([])

        from vrcc.gui.style import apply_font_scale, apply_theme_guarded
        from vrcc.i18n.qt import apply_ui_language

        apply_ui_language(app, store.config.gui.ui_language)
        apply_theme_guarded(app, store.config.gui.theme, store.config.gui.font_scale)
        apply_font_scale(app, store.config.gui.font_scale)

        from vrcc.gui.boot_panel import BootPanel

        panel = BootPanel(store.config.gui.theme)
        panel.show()
        app.processEvents()
        progress = _Both(panel, LogProgress(), app)
    except Exception:  # noqa: BLE001 -- see the comment above: the walk must still run
        logger.warning("boot: could not build the progress panel", exc_info=True)

    _walk_imports(progress)

    return _run_app(
        portable=portable,
        verbose=verbose,
        guard=guard,
        paths=paths,
        store=store,
        progress=progress,
    )
