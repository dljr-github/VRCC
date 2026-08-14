"""The "They read" pills must show exactly what the engine will translate into.

Three ways that promise used to break: a pill equal to the source (dropped from
config while still on screen), two pills showing one language (stored once), and
a new slot opening on the registry's first entry, which on a stock install is
also the source. All three left the user looking at a target nothing would ever
be translated into, with nothing on screen to say so.

The fourth is the inverse: what config STORES is what the user asked for, which
under m2m100 is not always what the combo SHOWS. Reading the display back into
config converted their choice the first time they touched an unrelated pill.
"""

from __future__ import annotations

import os
import pathlib
import tempfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication

from vrcc.core.bus import EventBus
from vrcc.core.config import ConfigStore, default_paths
from vrcc.gui.bridge import BusBridge
from vrcc.gui.main_window import MainWindow


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


class _Pipe:
    captioning_enabled = True

    def submit_typed(self, *a):
        return True

    def set_captioning(self, *a):
        pass


def _window(qapp, **fields):
    store = ConfigStore(
        default_paths(
            portable=True, app_dir=pathlib.Path(tempfile.mkdtemp())
        ).config_file
    )
    for dotted, value in fields.items():
        section, field = dotted.split(".")
        setattr(getattr(store.config, section), field, value)
    bridge = BusBridge(EventBus())
    return store, bridge, MainWindow(bridge, store, _Pipe(), lambda: None, lambda: None)


def _pills(window) -> list[str]:
    return [
        combo.currentText()
        for slot, combo in enumerate(window._target_combos)
        if window._target_checks[slot] is None or window._target_checks[slot].isChecked()
    ]


def test_every_visible_pill_is_a_target_that_was_stored(qapp):
    # The invariant the other tests are specific cases of.
    store, bridge, w = _window(qapp, **{"translate.targets": ["Japanese"]})
    try:
        w._add_target()
        w._add_target()
        assert _pills(w) == list(store.config.translate.targets)
        assert len(set(_pills(w))) == len(_pills(w))
    finally:
        w.close()
        bridge.detach()


def test_adding_a_language_does_not_open_on_the_source(qapp):
    # Slots are built on LANGUAGES' first entry, which is English, and the
    # stock source_language is English too: the new pill was dropped by the
    # rebuild the instant it appeared.
    store, bridge, w = _window(qapp, **{"translate.targets": ["Japanese"]})
    try:
        assert store.config.stt.source_language == "English"
        w._add_target()
        assert _pills(w)[-1] != "English"
        assert _pills(w)[-1] in store.config.translate.targets
    finally:
        w.close()
        bridge.detach()


def test_speaking_a_language_moves_it_off_the_target_pills(qapp):
    store, bridge, w = _window(qapp, **{"translate.targets": ["Japanese"]})
    try:
        w._source_combo.setCurrentText("Japanese")

        assert "Japanese" not in _pills(w)
        assert _pills(w) == list(store.config.translate.targets)
        assert store.config.translate.targets, "a live target must survive"
    finally:
        w.close()
        bridge.detach()


def test_two_pills_cannot_show_the_same_language(qapp):
    store, bridge, w = _window(
        qapp, **{"translate.targets": ["Japanese", "Korean"]}
    )
    try:
        w._target_combos[1].setCurrentText("Japanese")

        assert _pills(w).count("Japanese") == 1
        assert _pills(w) == list(store.config.translate.targets)
    finally:
        w.close()
        bridge.detach()


def test_a_collapsed_target_survives_an_unrelated_edit(qapp):
    """Under m2m100 the combo shows Chinese Simplified for a stored Chinese
    Traditional. Touching a different pill must not write the shown name back:
    the user gets their choice again on a model that can tell the two apart."""
    store, bridge, w = _window(
        qapp,
        **{
            "translate.model": "m2m100-418M-int8",
            "translate.targets": ["Chinese Traditional"],
        },
    )
    try:
        assert w._target_combos[0].currentText() == "Chinese Simplified"

        w._add_target()  # an edit somewhere else entirely

        assert store.config.translate.targets[0] == "Chinese Traditional"

        store.config.translate.model = "nllb-600M-int8"
        w.reload_from_config()
        assert w._target_combos[0].currentText() == "Chinese Traditional"
    finally:
        w.close()
        bridge.detach()
