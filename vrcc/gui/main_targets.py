"""The "You speak X, they read Y" pills: add, remove, and keep them honest.

Split out of :mod:`vrcc.gui.main_window` for the 500-line cap, and because the
rule these functions share is easy to lose among window plumbing: what a pill
SHOWS is what the engine will produce, while what config STORES is what the user
asked for. Those differ under a model that renders two languages with one
control token (m2m100 and madlad do it to the Chinese scripts), and collapsing
the two would either mislabel the caption log or quietly rewrite the user's
choice the first time they touched an unrelated pill.

So `_target_intent[slot]` is the answer of record and the combo is a view of it.
Every check here (duplicate, equal to the source) runs on the shown name,
because two pills that decode identically are two identical chatbox lines
regardless of what they are called.

Takes the window as its first argument, the same shape
:mod:`vrcc.gui.settings_pages` uses for the settings dialog.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from vrcc.gui import mt_prompts
from vrcc.gui.widgets import set_combo_text
from vrcc.i18n import tr

if TYPE_CHECKING:
    from vrcc.gui.main_window import MainWindow


def load_targets(w: MainWindow, cfg) -> None:
    """Fill the pills from config, recording what each slot was ASKED for.

    A target the model cannot render distinctly shows as the language it
    collapses onto, and its entry is greyed, so the promise the combo makes is
    one the engine can keep. Display only: config keeps what the user asked
    for, so a model that can tell the scripts apart honours it again instead of
    having lost it. Runs on every reload_from_config, which is what picks up a
    model swap made in Settings.

    Paired in one pass rather than zipped from two lists: resolving can DROP an
    entry, and indexing a shortened display list against the stored one hands
    slot i the name of a different target, which the next edit then writes back
    as the user's choice.
    """
    pairs = mt_prompts.target_pairs(
        cfg.translate.model, cfg.translate.targets, cfg.stt.source_language
    )
    for slot, combo in enumerate(w._target_combos):
        mt_prompts.grey_collapsed_targets(combo, cfg.translate.model)
        check = w._target_checks[slot]
        if slot < len(pairs):
            asked, display = pairs[slot]
            w._target_intent[slot] = asked
            set_combo_text(combo, display)
            if check is not None:
                check.setChecked(True)
        else:
            w._target_intent[slot] = combo.currentText()
            if check is not None:
                check.setChecked(False)


def add_target(w: MainWindow, slots: int) -> None:
    """Enable the lowest hidden slot; its checkbox drives the rebuild."""
    for slot in range(1, slots):
        check = w._target_checks[slot]
        if check is not None and not check.isChecked():
            seed_free_target(w, slot)
            check.setChecked(True)
            break
    sync_target_visibility(w, slots)


def seed_free_target(w: MainWindow, slot: int) -> None:
    """Point a slot about to be shown at a language the rebuild will keep.

    Every slot is built on the registry's first entry, which on a stock install
    is also the source language, so a new pill would be dropped by
    :func:`rebuild_targets` the moment it appeared: the user sees a target that
    never reaches config and never gets translated. Anything already spoken or
    targeted is skipped for the same reason, and a greyed entry is skipped
    because the active model cannot render it distinctly.
    """
    taken = {w._store.config.stt.source_language}
    for other, combo in enumerate(w._target_combos):
        check = w._target_checks[other]
        if other != slot and (check is None or check.isChecked()):
            taken.add(combo.currentText())
    combo = w._target_combos[slot]
    if combo.currentText() not in taken:
        return
    model = combo.model()
    for i in range(combo.count()):
        item = model.item(i)
        if combo.itemText(i) not in taken and (item is None or item.isEnabled()):
            set_combo_text(combo, combo.itemText(i))
            return


def remove_target(w: MainWindow, slot: int, slots: int) -> None:
    check = w._target_checks[slot]
    if check is not None:
        check.setChecked(False)
    sync_target_visibility(w, slots)


def sync_target_visibility(w: MainWindow, slots: int) -> None:
    """Show a slot's pill iff its (hidden) checkbox is on; offer "+ Language"
    only while a slot is still free."""
    any_free = False
    for slot in range(1, slots):
        check = w._target_checks[slot]
        cont = w._target_conts[slot]
        if check is None or cont is None:
            continue
        cont.setVisible(check.isChecked())
        if not check.isChecked():
            any_free = True
    w._add_target_btn.setVisible(any_free)


def rebuild_targets(w: MainWindow, slots: int) -> None:
    """Write the enabled slots back to config, dropping what cannot stand.

    Dedupe across slots (first-occurrence order) and drop any target equal to
    the source: translating a language into itself just re-sends the original.
    Both checks run on the name the combo SHOWS, which is what the engine will
    actually produce, while config keeps the name the user ASKED for.

    Only name equality, deliberately. Sharing a control token with the source is
    NOT a reason to refuse: measured on the real m2m100, Chinese Traditional in
    with Chinese Simplified out returns converted Simplified
    ("我這個學期要去臺灣" -> "这个学期,我要去台湾"), not an echo of the source.
    Refusing that pair deleted a target the engine delivers.
    """
    if w._loading or w._rebuilding:
        return
    w._rebuilding = True
    note: str | None = None
    try:
        source = w._store.config.stt.source_language
        targets: list[str] = []
        seen: set[str] = set()
        for slot, combo in enumerate(w._target_combos):
            check = w._target_checks[slot]
            if not (check is None or check.isChecked()):
                continue
            was = combo.currentText()
            if was in seen or was == source:
                # A target the rebuild would drop must not stay on screen
                # claiming to be a language nothing will ever translate into.
                # Say which one moved and why, or the change looks like a bug.
                seed_free_target(w, slot)
                w._target_intent[slot] = combo.currentText()
                if was == source:
                    note = tr(
                        "You speak {lang}, so that target changed to {other}.",
                        lang=was, other=combo.currentText(),
                    )
                else:
                    note = tr(
                        "{lang} was already a target, so this one changed to {other}.",
                        lang=was, other=combo.currentText(),
                    )
                if combo.currentText() in seen or combo.currentText() == source:
                    continue
            seen.add(combo.currentText())
            targets.append(w._target_intent[slot])
        w._store.config.translate.targets = targets[:slots]
        w._store.save_soon()
    finally:
        w._rebuilding = False
    if note is not None:
        w._flash_status(note)
