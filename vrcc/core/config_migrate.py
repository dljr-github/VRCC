"""Schema migrations for a stored `AppConfig`.

Each one adopts a new default for a value an older build wrote, and each is
guarded on the stored schema version so it runs once. Split out of
:mod:`vrcc.core.config` to keep that module under the line cap.

Defaults are read off the instance's own model class rather than imported, so
this module never imports the one that imports it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vrcc.core.config import AppConfig


# The only two MT beam widths the mode bundles ever wrote. A stored config
# holding one of them most likely recorded the mode rather than a decision
# about translation, but the two are indistinguishable: 1 was also the shipping
# default and both are inside the Advanced page's range, so a user who picked
# one by hand is migrated along with everyone else. That is the intended trade,
# because greedy MT decoding fabricates content and the alternative is leaving
# every pre-schema-2 install on it.
_PROFILE_WRITTEN_MT_BEAMS = (1, 3)


def _migrate_profile_written_mt_beam(config: AppConfig, stored_version: int) -> None:
    """Adopt the current MT beam default for a config the mode control wrote.

    Until schema 2 the Speed/Quality bundles set ``translate.beam_size`` to 1 or
    3. Because the stored file carries every field, leaving those values alone
    would keep existing users on the greedy decoding this default moved away
    from, and the fix would reach new installs only. Any width outside
    :data:`_PROFILE_WRITTEN_MT_BEAMS` survives untouched; the two inside it are
    migrated even when the user chose them, for the reason recorded there.
    Setting either again from the Advanced page sticks, since this runs once.
    """
    if stored_version >= 2:
        return
    if config.translate.beam_size in _PROFILE_WRITTEN_MT_BEAMS:
        config.translate.beam_size = type(config.translate)().beam_size


def _migrate_default_overflow(config: AppConfig, stored_version: int) -> None:
    """Adopt "auto" for a config carrying the pre-schema-3 shipping default.

    "split" is what every untouched install stored, so it records the default
    far more often than a decision and the two are indistinguishable in the
    file. Same trade as the MT beam above: whoever did choose split is
    migrated too, and choosing it again sticks. "truncate" and "send" were
    only reachable by hand and survive untouched.
    """
    if stored_version >= 3:
        return
    if config.osc.overflow == "split":
        config.osc.overflow = type(config.osc)().overflow


def apply_migrations(config: "AppConfig", stored_version: int) -> None:
    """Run every schema migration against `config`, oldest first."""
    _migrate_profile_written_mt_beam(config, stored_version)
    _migrate_default_overflow(config, stored_version)
