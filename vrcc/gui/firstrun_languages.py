"""The first-run wizard's "which languages do you speak" step.

Split out of :mod:`vrcc.gui.firstrun` for the 500-line cap, and because this is
the one part of the wizard with real logic rather than layout: the answer feeds
the model recommendation (a restricted model leads only when it covers every
language spoken) and then has to be collapsed back into the single
``stt.source_language`` the engines actually run off.

Functions take the wizard, the way :mod:`vrcc.gui.settings_pages` takes the
settings dialog.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QListWidget, QListWidgetItem

from vrcc.core.languages import LANGUAGES
from vrcc.stt.registry import WHISPER_MODELS

_AUTO = "auto"

# ~4.5 rows: enough to see the pre-ticked language plus its neighbours without
# the picker dominating a wizard that also has to fit the device row, the plan
# summary and two progress bars.
_PICKER_HEIGHT_PX = 112


def build_spoken_picker(wizard) -> QListWidget:
    """The checkable language list, pre-ticked from config.

    The ``itemChanged`` hookup happens *after* pre-ticking, so the handler only
    ever fires on a real user edit and needs no loading guard.
    """
    picker = QListWidget()
    picker.setSelectionMode(QListWidget.SelectionMode.NoSelection)
    picker.setFixedHeight(round(_PICKER_HEIGHT_PX * wizard._scale))
    for display in LANGUAGES:
        item = QListWidgetItem(display)
        item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
        item.setCheckState(Qt.CheckState.Unchecked)
        picker.addItem(item)

    _preselect(picker, wizard._store.config)
    picker.itemChanged.connect(lambda _item: on_spoken_changed(wizard))
    return picker


def _preselect(picker: QListWidget, cfg) -> None:
    """Tick what the config already says the user speaks: their stored
    multi-select if they have one, else the single source language app.run
    seeded from the OS locale. A stored "auto" ticks nothing."""
    wanted = set(cfg.stt.spoken_languages or [cfg.stt.source_language])
    for i in range(picker.count()):
        item = picker.item(i)
        if item.text() in wanted:
            item.setCheckState(Qt.CheckState.Checked)


def checked_spoken(wizard) -> list[str]:
    """Ticked display names, in the registry's display order."""
    picker = wizard._spoken_list
    return [
        picker.item(i).text()
        for i in range(picker.count())
        if picker.item(i).checkState() == Qt.CheckState.Checked
    ]


def on_spoken_changed(wizard) -> None:
    """Commit a tick, re-plan, then re-derive the source language.

    Order matters: the recommendation depends on the languages, and the source
    language depends on what got recommended.
    """
    wizard._store.config.stt.spoken_languages = checked_spoken(wizard)
    wizard._refresh_plan()
    apply_source_language(wizard)
    wizard._store.save_soon()


def apply_source_language(wizard) -> None:
    """Derive the single ``stt.source_language`` the engines run off from the
    multi-select.

    One language pins it. Several means "auto" -- but only when the recommended
    model can actually serve auto: a model that detects the language without
    reporting it (Parakeet) tags every result "en", which Settings greys out
    while translation is on and which would mislabel the translator's source.
    There one of the ticked languages has to be named, because a source the
    user can see and correct beats one that is silently wrong.

    Nothing ticked leaves the stored value alone: the user declined to say, and
    the recommendation falls back to being language-blind.
    """
    cfg = wizard._store.config
    checked = checked_spoken(wizard)
    if not checked:
        return
    if len(checked) == 1:
        cfg.stt.source_language = checked[0]
        return

    spec = WHISPER_MODELS.get(wizard.recommended_whisper)
    serves_auto = (
        spec is not None
        and spec.auto_language
        and (spec.reports_language or not wizard._translation_enabled())
    )
    if serves_auto:
        cfg.stt.source_language = _AUTO
        return
    # Keep whichever language is already the source if the user still speaks
    # it; falling to the first tick would silently re-point the source at
    # whatever happens to sort first in the picker.
    if cfg.stt.source_language not in checked:
        cfg.stt.source_language = checked[0]
