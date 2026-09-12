"""The ordered phases a launch reports, and the reporter that logs them.

Qt-free so :mod:`vrcc.boot` can report a phase before any widget exists, and so
the table can be tested with no display.

The phases are named after what the import actually does, and they follow
:mod:`vrcc.app`'s own module-scope import order because that order decides which
step pays for a shared dependency. Nothing below about 10 ms gets a step: the
run-to-run spread of a whole launch is larger than that, so a shorter step would
report noise.
"""

from __future__ import annotations

import logging

from vrcc.i18n import tr, tr_noop

logger = logging.getLogger("vrcc.core.progress")

PHASES = (
    ("audio", tr_noop("Setting up audio")),
    ("hardware", tr_noop("Checking your graphics card")),
    ("speech", tr_noop("Loading speech recognition")),
    ("downloads", tr_noop("Preparing model downloads")),
    ("interface", tr_noop("Building the window")),
)

_LABELS = dict(PHASES)


def phase_labels() -> tuple[str, ...]:
    """Translated labels in phase order.

    Read through tr() here rather than at import time: the catalog is empty until
    apply_ui_language runs, and this module is imported well before that.
    """
    return tuple(tr(label) for _, label in PHASES)


def phase_label(key: str) -> str:
    """Translated label for one phase key, read at call time like phase_labels().

    A key outside the table falls back to itself instead of raising: a launch
    must never die because a caller named a phase that does not exist, and the
    raw key still reads as something rather than nothing in a log or a panel.
    tr() then passes an unrecognised key through unchanged, same as it does for
    any string that is not a catalog entry.
    """
    return tr(_LABELS.get(key, key))


class LogProgress:
    """Writes each phase to the run log. The only reporter in a packaged launch
    that leaves evidence behind, so per-phase timings can be read back out of a
    user's log without new tooling."""

    def __init__(self) -> None:
        self.steps: list[str] = []

    def start(self, key: str) -> None:
        self.steps.append(key)
        logger.info("boot step: %s", key)

    def close(self) -> None:
        logger.info("boot steps complete: %s", len(self.steps))
