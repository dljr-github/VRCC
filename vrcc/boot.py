"""Read config, bring Qt up, show what is loading, then hand off to the app.

Distinct from :mod:`vrcc.core.startup`, which holds model-readiness helpers that
run()'s body calls. This module is what runs BEFORE run() exists.

Config is read before QApplication exists, not after: the panel needs the
user's language and theme to build itself already right, and only
ConfigStore.load() can supply those. Reading it first is deliberate, not an
oversight -- ``python -X importtime`` puts the cumulative cost of ``from
vrcc.core.config import ConfigStore, default_paths`` at roughly 176,000 to
184,000 microseconds over three runs on this machine, the largest single
cost paid before the panel can appear. Every other heavy import in
:mod:`vrcc.app`'s module scope happens after QApplication and behind the
panel built here, so the window that reports progress exists before the slow
part rather than after it. The paths, the logging, the config store and the
QApplication are built once here and passed into run(), which skips building
its own: setup_logging adds a fresh file handler on every call, and two
ConfigStore objects would leave run() closing over a different one than the
panel was themed from.
"""

from __future__ import annotations

import importlib
import logging

from PySide6.QtWidgets import QApplication

from vrcc.core.config import ConfigStore, default_paths
from vrcc.core.logs import setup_logging
from vrcc.core.progress import LogProgress

logger = logging.getLogger("vrcc.boot")

# Mirrors vrcc.app's own module-scope import order (audio at its line 14,
# hardware at 15, speech's engine_stack at 18, downloads at 34), so whichever
# phase here is charged for a shared import is the one a reader scanning
# that file would expect. That is all this order buys: vrcc.app imports the
# same names at its own module scope regardless of what order this tuple
# lists them in, and Python runs module-scope imports before any function
# body, so onnxruntime is loaded before run() reaches setup_cuda_dlls()
# no matter how this table is ordered.
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
    """Name each phase before paying for it, then import it inside a try
    that logs and moves on to the next phase rather than re-raising.

    A failing group's "group failed to import" warning is written to the
    log at the moment it fails. The launch does not survive the failure
    regardless: vrcc.app imports these same modules again, most of them at
    its own module scope and the window's GUI modules inside run()'s body,
    so the process still exits either while :func:`_run_app` imports
    vrcc.app or once run() begins. What the catch here buys is where the
    traceback goes: logged here, it reaches the run log; left to propagate,
    it would only reach stderr, which ``cli._ensure_std_streams`` points at
    the null device in a windowed build. Catching here is how a broken
    group gets recorded at all.
    """
    for key, module_names in _IMPORT_GROUPS:
        try:
            progress.start(key)
            for name in module_names:
                _import_module(name)
        except Exception:  # noqa: BLE001 -- a reporter or an import failing must not sink the walk
            logger.warning("boot: %s group failed to import", key, exc_info=True)


def _close_native_splash() -> None:
    """Dismiss the bootloader's splash once Qt has something on screen.

    ``pyi_splash`` is injected by PyInstaller and exists only inside a frozen
    build, so a source run finds nothing to close and must stay quiet about
    it. Nothing this function does may propagate: boot() has no recovery for
    a failure here beyond falling back to a bare log reporter and abandoning
    the panel it already built, which is worse than a splash left on screen.
    """
    try:
        import pyi_splash
    except ImportError:
        return
    except Exception:  # noqa: BLE001 -- an import failure must not sink the boot either
        logger.debug("could not import the bootloader splash module", exc_info=True)
        return
    try:
        pyi_splash.close()
    except Exception:  # noqa: BLE001 -- a stuck splash must not stop the launch
        logger.debug("could not close the bootloader splash", exc_info=True)


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
    the whole walk, which is worse than no panel at all.

    ``close()`` is idempotent: run() calls it once if the first-run wizard
    opens and again once the main window is ready, and the second call must
    not write a second "boot steps complete" line to a log someone is
    reading to troubleshoot.
    """

    def __init__(self, panel, log: LogProgress, app: QApplication) -> None:
        self._panel = panel
        self._log = log
        self._app = app
        self._closed = False

    def start(self, key: str) -> None:
        self._panel.start(key)
        self._log.start(key)
        self._app.processEvents()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
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
    progress = LogProgress()
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
        _close_native_splash()
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
