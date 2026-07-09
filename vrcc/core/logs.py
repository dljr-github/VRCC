"""Per-run log files: configure the root logger, retire the old runs.

One file per run, named for its start time and written at DEBUG regardless of
``--verbose``, so a bug report means attaching the newest file instead of
reconstructing which slice of a rotating log covers the session that went
wrong. Only the newest few runs are kept. Zero Qt.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

logger = logging.getLogger("vrcc.core.logs")

LOG_KEEP = 5
_LOG_GLOB = "vrcc*.log*"

# Libraries whose DEBUG output would bury VRCC's own in the file (per-frame
# discovery chatter, HTTP retries, lock acquisitions).
_QUIET_LOGGERS = {
    "asyncio": logging.INFO,
    "zeroconf": logging.INFO,
    "urllib3": logging.INFO,
    "huggingface_hub": logging.INFO,
    "filelock": logging.WARNING,
}


def prune_logs(logs_dir: Path, keep: int = LOG_KEEP) -> None:
    """Delete all but the ``keep`` newest log files in ``logs_dir``. Never
    raises: losing an old log must not stop the app from launching."""
    try:
        files = sorted(
            (p for p in logs_dir.glob(_LOG_GLOB) if p.is_file()),
            key=lambda p: (p.stat().st_mtime, p.name),
            reverse=True,
        )
        for stale in files[keep:]:
            stale.unlink(missing_ok=True)
    except OSError:
        logger.debug("could not prune old logs in %s", logs_dir, exc_info=True)


def setup_logging(logs_dir: Path, verbose: bool) -> None:
    """Log this run to its own file in ``logs_dir`` at DEBUG, keeping only the
    newest few runs, plus a console handler when ``verbose``.

    The file always gets DEBUG (that is what makes a bug report useful);
    ``verbose`` only decides whether the same records also reach the console.
    """
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    for name, level in _QUIET_LOGGERS.items():
        logging.getLogger(name).setLevel(level)

    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s")

    try:
        logs_dir.mkdir(parents=True, exist_ok=True)
        # Prune before opening so this run's file is never a deletion target.
        prune_logs(logs_dir, LOG_KEEP - 1)
        file_handler = logging.FileHandler(
            logs_dir / f"vrcc-{time.strftime('%Y%m%d-%H%M%S')}.log",
            encoding="utf-8",
        )
        file_handler.setFormatter(fmt)
        file_handler.setLevel(logging.DEBUG)
        root.addHandler(file_handler)
    except OSError:
        # Never let a log-file problem stop the app from launching.
        logger.warning(
            "could not open log file in %s; continuing without file logging",
            logs_dir,
            exc_info=True,
        )

    # Console only when asked for: the windowed exe has no stderr to write to.
    if verbose:
        console = logging.StreamHandler()
        console.setFormatter(fmt)
        root.addHandler(console)
