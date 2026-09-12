"""The ordered phases a launch reports, and the two ways of reporting them.

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


class NoProgress:
    """Accepts the same calls and does nothing, so a caller never needs to test
    whether it has a reporter."""

    def start(self, key: str) -> None:
        pass

    def close(self) -> None:
        pass
