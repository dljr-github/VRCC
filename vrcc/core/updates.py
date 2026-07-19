"""Check GitHub releases for a newer VRCC and publish the result. Qt-free.

Notify only: a positive result carries the release page URL for the GUI to
open. The check runs on a daemon thread; any network failure is silent on the
launch check (an offline app must not nag) and reported only on a manual check.
"""

from __future__ import annotations

import json
import logging
import threading
import urllib.request

from vrcc.core.bus import EventBus
from vrcc.core.events import UpdateCheckResult

logger = logging.getLogger("vrcc.core.updates")

_API = "https://api.github.com/repos/{repo}/releases/latest"
_TIMEOUT_S = 6.0


def _parse(version: str) -> tuple[int, ...] | None:
    v = version.strip().lstrip("vV")
    if not v:
        return None
    parts = v.split(".")
    try:
        return tuple(int(p) for p in parts)
    except ValueError:
        return None


def is_newer(latest: str, current: str) -> bool:
    """Whether ``latest`` is a strictly higher version than ``current``.
    Tolerant: an unparseable ``latest`` is treated as not newer."""
    lo, cur = _parse(latest), _parse(current)
    if lo is None or cur is None:
        return False
    return lo > cur


def _fetch_latest(repo: str) -> tuple[str, str]:
    """Return (tag_name, html_url) of the latest release. Raises on failure."""
    req = urllib.request.Request(
        _API.format(repo=repo),
        headers={"User-Agent": "VRCC-update-check", "Accept": "application/vnd.github+json"},
    )
    with urllib.request.urlopen(req, timeout=_TIMEOUT_S) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data.get("tag_name", ""), data.get("html_url", "")


class UpdateChecker:
    def __init__(self, bus: EventBus, current_version: str, repo: str = "dljr-github/VRCC") -> None:
        self._bus = bus
        self._current = current_version
        self._repo = repo

    def check(self, announce_no_update: bool = False) -> None:
        """Run one check on a daemon thread."""
        t = threading.Thread(
            target=self._run, args=(announce_no_update,), name="VrccUpdateCheck", daemon=True
        )
        t.start()

    def _run(self, announce_no_update: bool) -> None:
        try:
            tag, url = _fetch_latest(self._repo)
        except Exception as exc:  # noqa: BLE001 -- offline is normal; never crash
            logger.debug("update check failed", exc_info=True)
            if announce_no_update:
                self._bus.publish(UpdateCheckResult(available=False, error=str(exc) or "check failed"))
            return
        if is_newer(tag, self._current):
            latest = tag.strip().lstrip("vV")
            self._bus.publish(UpdateCheckResult(available=True, latest=latest, url=url))
        elif announce_no_update:
            self._bus.publish(UpdateCheckResult(available=False))
