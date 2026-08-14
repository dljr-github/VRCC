"""The "which languages do you speak" picker and the logic behind it.

Split out of :mod:`vrcc.gui.firstrun` for the 500-line cap, and because this is
the one part of the wizard with real logic rather than layout: the answer feeds
the model recommendation (a restricted model leads only when it covers every
language spoken) and then has to be collapsed back into the single
``stt.source_language`` the engines actually run off.

Two layers. :func:`build_picker`, :func:`checked_in` and
:func:`resolve_source_language` know nothing about who is hosting them, so the
Models window can offer the same control after first run (without it, a user
who never answered the wizard is stuck on the language-blind recommendation
with no way to correct it). The wrappers below them take the wizard, the way
:mod:`vrcc.gui.settings_pages` takes the settings dialog.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QListWidget, QListWidgetItem

from vrcc.core.languages import LANGUAGES
from vrcc.gui import model_prompts
from vrcc.i18n import tr
from vrcc.stt.registry import WHISPER_MODELS

__all__ = [
    "build_picker", "checked_in", "build_spoken_picker", "checked_spoken",
    "on_spoken_changed", "apply_source_language", "resolve_source_language",
    "resync_spoken", "retarget_off_source", "warn_if_source_unservable",
]

_AUTO = "auto"

# ~4.5 rows: enough to see the pre-ticked language plus its neighbours without
# the picker dominating a wizard that also has to fit the device row, the plan
# summary and two progress bars.
_PICKER_HEIGHT_PX = 112


class _RowToggleList(QListWidget):
    """A checkable list where clicking anywhere on a row toggles it.

    Qt's hit area is the check indicator alone, about 14 px of a row several
    hundred px wide, and NoSelection means a click on the language name gives no
    feedback at all: not a toggle, not even a highlight. Since ticking a
    language is the gate on both proceed buttons, a click that silently does
    nothing reads as the wizard being broken.

    Left clicks are handled here and NOT passed to Qt, so a click on the
    indicator toggles once rather than twice.
    """

    def mousePressEvent(self, event):  # noqa: N802 -- Qt override
        item = self.itemAt(event.position().toPoint())
        if item is not None and event.button() == Qt.MouseButton.LeftButton:
            checked = item.checkState() == Qt.CheckState.Checked
            item.setCheckState(
                Qt.CheckState.Unchecked if checked else Qt.CheckState.Checked
            )
            return
        super().mousePressEvent(event)


def build_picker(scale: float, cfg, on_change) -> QListWidget:
    """The checkable language list, pre-ticked from ``cfg``.

    The ``itemChanged`` hookup happens *after* pre-ticking, so ``on_change``
    only ever fires on a real user edit and needs no loading guard.
    """
    picker = _RowToggleList()
    picker.setSelectionMode(QListWidget.SelectionMode.NoSelection)
    picker.setFixedHeight(round(_PICKER_HEIGHT_PX * scale))
    for display in LANGUAGES:
        item = QListWidgetItem(display)
        item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
        item.setCheckState(Qt.CheckState.Unchecked)
        picker.addItem(item)

    _preselect(picker, cfg)
    _scroll_to_first_checked(picker)
    picker.itemChanged.connect(lambda _item: on_change())
    return picker


def _scroll_to_first_checked(picker: QListWidget) -> None:
    """Bring the pre-ticked language into view.

    The list shows about 4.5 of 30 rows, so for most locales the answer the
    wizard already filled in sits below the fold with the scrollbar at zero. The
    user sees an all-unticked list under a hint telling them to pick one, and
    reasonably ticks a second language they did not want.
    """
    for i in range(picker.count()):
        if picker.item(i).checkState() == Qt.CheckState.Checked:
            picker.scrollToItem(
                picker.item(i), QListWidget.ScrollHint.PositionAtCenter
            )
            return


def checked_in(picker: QListWidget) -> list[str]:
    """Ticked display names, in the registry's display order."""
    return [
        picker.item(i).text()
        for i in range(picker.count())
        if picker.item(i).checkState() == Qt.CheckState.Checked
    ]


def build_spoken_picker(wizard) -> QListWidget:
    return build_picker(
        wizard._scale, wizard._store.config, lambda: on_spoken_changed(wizard)
    )


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
    return checked_in(wizard._spoken_list)


def on_spoken_changed(wizard) -> None:
    """Commit a tick, re-plan, then re-derive the source language.

    Order matters: the recommendation depends on the languages, and the source
    language depends on what got recommended. Unticking the last language is
    the exception, because there the source has to be cleared first or the plan
    below still ranks for the language just removed.
    """
    checked = checked_spoken(wizard)
    wizard._store.config.stt.spoken_languages = checked
    if not checked:
        apply_source_language(wizard)
    wizard._refresh_plan()
    apply_source_language(wizard)
    # After the source is resolved, not during _refresh_plan: the target can
    # only be checked against a source that has caught up with this tick.
    retarget_off_source(wizard)
    wizard._store.save_soon()


def apply_source_language(wizard) -> None:
    """Re-derive the wizard's ``stt.source_language`` from its ticks.

    Nothing ticked clears it to "auto", and that is what makes the next plan
    language-blind: ``spoken_whisper_codes`` falls back to this field, so the
    answer the user just removed would otherwise keep steering the ranking and
    the summary would go on advertising a model chosen for it. A model that
    cannot honour "auto" keeps the stored value, for the reason on
    :func:`_serves_auto`, and the plan stays as it was.
    """
    checked = checked_spoken(wizard)
    cfg = wizard._store.config
    translating = wizard._translation_enabled()
    if not checked:
        spec = WHISPER_MODELS.get(wizard.recommended_whisper)
        if _serves_auto(spec, translating):
            cfg.stt.source_language = _AUTO
        return
    resolve_source_language(
        cfg, checked, wizard.recommended_whisper, translating
    )


def _serves(spec, display: str) -> bool:
    """Whether ``spec`` can transcribe the language behind a display name.

    Through :func:`vrcc.gui.model_prompts.covers` because this decides which
    entries may be written as the source and that decides which ones Settings
    and the main window leave enabled; two readings of "can transcribe" would
    let one surface offer what the other refuses. An unknown model id restricts
    nothing there, so it restricts nothing here.
    """
    lang = LANGUAGES.get(display)
    if lang is None:
        return False
    return spec is None or model_prompts.covers(spec, lang.whisper)


def _serves_auto(spec, translation_enabled: bool) -> bool:
    """Whether ``"auto"`` is a source the model can honour: it has to detect the
    language, and (while translation is on) report which one it heard. A model
    that detects without reporting (Parakeet) tags every result "en", which
    Settings greys out and which would mislabel the translator's source."""
    return (
        spec is not None
        and spec.auto_language
        and (spec.reports_language or not translation_enabled)
    )


def resolve_source_language(
    cfg, checked: list[str], whisper_id: str, translation_enabled: bool
) -> None:
    """Derive the single ``stt.source_language`` the engines run off from the
    multi-select, in place on ``cfg``.

    One language pins it. Several means "auto" -- but only when the model can
    actually serve auto (:func:`_serves_auto`). Where it cannot, one of the
    ticked languages has to be named, because a source the user can see and
    correct beats one that is silently wrong.

    ``whisper_id`` is whichever model will be running once the caller is done:
    the wizard's recommendation, which it is about to install, or the active
    ``cfg.stt.model`` for a surface that only reports. A language that model
    cannot transcribe is never written, because Settings and the main window
    grey exactly those entries out, and a source they refuse to offer must not
    be reachable from here either.

    Nothing ticked leaves the stored value alone: declining to say is not the
    same as saying "auto", and a surface that only reports on a running config
    must not drop the language the engines are captioning in. The wizard, which
    has no running config and whose whole screen is a plan, clears it instead
    (:func:`apply_source_language`).
    """
    if not checked:
        return
    spec = WHISPER_MODELS.get(whisper_id)
    servable = [display for display in checked if _serves(spec, display)]
    if not servable:
        return
    if len(servable) == 1:
        cfg.stt.source_language = servable[0]
        return

    if _serves_auto(spec, translation_enabled):
        cfg.stt.source_language = _AUTO
        return
    # Keep whichever language is already the source if the user still speaks
    # it; falling to the first tick would silently re-point the source at
    # whatever happens to sort first in the picker.
    if cfg.stt.source_language not in servable:
        cfg.stt.source_language = servable[0]


def retarget_off_source(wizard) -> None:
    """Move "They read" off a language the user already speaks.

    The shipped default target is Japanese and so is the pre-tick on a Japanese
    system, so the likeliest first run of all (open VRCC, press the primary
    button) would otherwise finish with source == target. The pipeline drops a
    source-equal target, so the wizard would promise translation and deliver
    none, with nothing on screen to explain it.

    Called from the wizard's _refresh_plan, which fires on every tick and device
    change, so it also covers a user who picks their language after seeing the
    plan.
    """
    source = wizard._store.config.stt.source_language
    spoken = set(checked_spoken(wizard))
    combo = wizard._target_combo
    current = combo.currentText()
    # With several languages ticked the source is "auto" and every ticked
    # language is dead as a target for its own utterances, not just one.
    if current != source and not (source == _AUTO and current in spoken):
        return
    model = combo.model()
    for i in range(combo.count()):
        name = combo.itemText(i)
        item = model.item(i)
        if name == source or name in spoken:
            continue
        if item is None or item.isEnabled():
            wizard._set_combo_text(combo, name)
            return


def warn_if_source_unservable(wizard, whisper_id: str) -> bool:
    """True (and says so) when the voice model about to start cannot transcribe
    the language the wizard would store as the source.

    :func:`resolve_source_language` deliberately leaves the stored value alone
    when NOTHING the user ticked is servable. That is the one case that must not
    be allowed to start: captions would come out wrong in silence, and the main
    window's rescue nudge has nothing better on disk to offer.
    """
    from PySide6.QtWidgets import QMessageBox

    from vrcc.gui.model_labels import whisper_display_name

    spec = WHISPER_MODELS.get(whisper_id)
    lang = LANGUAGES.get(wizard._store.config.stt.source_language)
    if spec is None or lang is None or model_prompts.covers(spec, lang.whisper):
        return False
    QMessageBox.information(
        wizard,
        tr("Almost there"),
        tr(
            "{name} cannot transcribe this language. Choose another voice "
            "model first.",
            name=whisper_display_name(whisper_id),
        ),
    )
    return True


def resync_spoken(wizard) -> None:
    """Re-tick the wizard's picker from config and re-plan.

    The wizard can hand its own config store to the Models window, which offers
    the same picker. Without this the wizard's widgets keep the answer from
    before that window opened, and _apply_recommendation writes those stale
    ticks back over the newer one.
    """
    picker = wizard._spoken_list
    picker.blockSignals(True)
    try:
        # Through _preselect, not a second copy of its rule: an empty
        # spoken_languages means "never answered", and reading it as "unticked
        # everything" would drop the pre-tick the user was about to accept and
        # leave the wizard with no proceed button enabled.
        for i in range(picker.count()):
            picker.item(i).setCheckState(Qt.CheckState.Unchecked)
        _preselect(picker, wizard._store.config)
    finally:
        picker.blockSignals(False)
    wizard._refresh_plan()
