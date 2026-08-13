"""Targets an MT family cannot tell apart.

m2m100 and madlad render both Chinese scripts with one control token, so a
Chinese Traditional request reaches their decoder as the Simplified one. Two
consequences the code has to handle: the same decode must not run twice, and
the UI must not offer a target the engine cannot deliver.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication, QComboBox

from vrcc.core.languages import LANGUAGES, get
from vrcc.gui import mt_prompts
from vrcc.translate.registry import (
    MT_MODELS,
    _collapse_map,
    collapses_onto,
    distinct_targets,
    lang_token,
)


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def test_which_families_collapse_which_languages():
    # The measurement, pinned. A language or family added later that collapses
    # something fails here rather than shipping a silently wrong target.
    assert _collapse_map("nllb") == {}
    assert _collapse_map("m2m100") == {"Chinese Traditional": "Chinese Simplified"}
    assert _collapse_map("madlad") == {"Chinese Traditional": "Chinese Simplified"}


def test_collapse_agrees_with_the_token_the_engine_sends():
    # The map is derived from lang_token rather than declared, so the two
    # cannot drift. This asserts the property that makes that worth doing.
    for family in ("nllb", "m2m100", "madlad"):
        for display, lang in LANGUAGES.items():
            other = collapses_onto(family, display)
            if other is not None:
                assert lang_token(family, lang) == lang_token(family, LANGUAGES[other])


def test_distinct_targets_drops_the_duplicate_decode():
    targets = [get("Chinese Simplified"), get("Chinese Traditional"), get("Japanese")]

    assert [t.display for t in distinct_targets("m2m100", targets)] == [
        "Chinese Simplified", "Japanese",
    ]
    # nllb has a token for each, so nothing is dropped.
    assert len(distinct_targets("nllb", targets)) == 3


def test_distinct_targets_names_the_script_it_will_produce():
    # Keeping the entry as asked would label Simplified output "Chinese
    # Traditional" in the caption log. Same resolution usable_targets applies to
    # the stored list, so the two surfaces agree.
    targets = [get("Chinese Traditional"), get("Chinese Simplified")]
    assert [t.display for t in distinct_targets("m2m100", targets)] == [
        "Chinese Simplified",
    ]
    assert [t.display for t in distinct_targets("m2m100", [get("Chinese Traditional")])] == [
        "Chinese Simplified",
    ]
    # nllb renders both, so nothing is resolved away.
    assert [t.display for t in distinct_targets("nllb", targets)] == [
        "Chinese Traditional", "Chinese Simplified",
    ]


def test_usable_targets_removes_the_duplicate():
    stored = ["Chinese Simplified", "Chinese Traditional", "Japanese"]

    assert mt_prompts.usable_targets("m2m100-418M-int8", stored) == [
        "Chinese Simplified", "Japanese",
    ]
    assert mt_prompts.usable_targets("nllb-600M-int8", stored) == stored


def test_a_sole_collapsed_target_is_relabelled_not_removed():
    # Regression, and the reason substitution beats dropping. A Taiwanese user
    # on the default model has exactly one target; dropping it left them with
    # no translation at all, and the greying stopped them choosing it back.
    assert mt_prompts.usable_targets("m2m100-418M-int8", ["Chinese Traditional"]) == [
        "Chinese Simplified",
    ]
    assert mt_prompts.usable_targets("m2m100-418M-int8", ["Chinese Traditional", "Japanese"]) == [
        "Chinese Simplified", "Japanese",
    ]


@pytest.mark.parametrize("model_id", list(MT_MODELS))
def test_usable_targets_never_empties_a_non_empty_list(model_id):
    for stored in (
        ["Chinese Traditional"],
        ["Chinese Simplified"],
        ["Chinese Traditional", "Chinese Simplified"],
        ["Japanese"],
    ):
        assert mt_prompts.usable_targets(model_id, stored), (model_id, stored)


def test_an_unknown_model_id_restricts_nothing():
    # A hand-edited config must not lose targets to a model we cannot inspect.
    assert mt_prompts.collapsed_target("not-a-model", "Chinese Traditional") is None
    assert mt_prompts.usable_targets("not-a-model", ["Chinese Traditional"]) == [
        "Chinese Traditional",
    ]


def _combo(qapp):
    combo = QComboBox()
    combo.addItems(list(LANGUAGES.keys()))
    return combo


def test_greying_disables_only_the_collapsed_entry(qapp):
    combo = _combo(qapp)

    mt_prompts.grey_collapsed_targets(combo, "m2m100-418M-int8")

    model = combo.model()
    disabled = [
        combo.itemText(i) for i in range(combo.count())
        if not model.item(i).isEnabled()
    ]
    assert disabled == ["Chinese Traditional"]


def test_greying_names_both_languages_in_the_tooltip(qapp):
    combo = _combo(qapp)

    mt_prompts.grey_collapsed_targets(combo, "m2m100-418M-int8")

    model = combo.model()
    tip = next(
        model.item(i).toolTip() for i in range(combo.count())
        if combo.itemText(i) == "Chinese Traditional"
    )
    assert "Chinese Traditional" in tip and "Chinese Simplified" in tip


def test_greying_re_enables_under_a_family_that_can_tell_them_apart(qapp):
    # The way out has to be reachable: switching the model must restore the
    # entry, or the user is stuck with no route back.
    combo = _combo(qapp)

    mt_prompts.grey_collapsed_targets(combo, "m2m100-418M-int8")
    mt_prompts.grey_collapsed_targets(combo, "nllb-600M-int8")

    model = combo.model()
    assert all(model.item(i).isEnabled() for i in range(combo.count()))
    assert all(model.item(i).toolTip() == "" for i in range(combo.count()))


def test_every_registered_mt_model_is_inspectable():
    # collapsed_target reads spec.family; a spec whose family lang_token does
    # not know would raise rather than grey.
    for model_id in MT_MODELS:
        assert mt_prompts.collapsed_target(model_id, "Japanese") is None


# -- a target that resolves onto the source -----------------------------------


def test_a_target_resolving_onto_the_source_is_dropped():
    """Chinese Traditional beside a Chinese Simplified source is distinct by
    name, but m2m100 renders both with one token, so it would decode the source
    into itself and echo the input back. The check has to happen after
    resolution, which is why it lives beside the resolution."""
    src = get("Chinese Simplified")
    targets = [get("Chinese Traditional"), get("Japanese")]

    assert [t.display for t in distinct_targets("m2m100", targets, src)] == ["Japanese"]
    # nllb tells the two apart, so it is a real target there.
    assert [t.display for t in distinct_targets("nllb", targets, src)] == [
        "Chinese Traditional", "Japanese",
    ]


def test_no_source_given_keeps_everything():
    targets = [get("Chinese Traditional"), get("Japanese")]
    assert len(distinct_targets("m2m100", targets)) == 2


def test_the_stored_choice_survives_a_collapsing_model(tmp_path):
    """The substitution is display only. A user whose model can no longer tell
    the scripts apart must get their target back when they switch to one that
    can, rather than having had it overwritten."""
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from vrcc.core.bus import EventBus
    from vrcc.core.config import ConfigStore, default_paths
    from vrcc.gui.bridge import BusBridge
    from vrcc.gui.main_window import MainWindow

    QApplication.instance() or QApplication([])

    class _Pipe:
        captioning_enabled = True

        def submit_typed(self, *a):
            return True

        def set_captioning(self, *a):
            pass

    store = ConfigStore(default_paths(portable=True, app_dir=tmp_path).config_file)
    store.config.translate.model = "nllb-600M-int8"
    store.config.translate.targets = ["Chinese Traditional", "Japanese"]
    bridge = BusBridge(EventBus())
    window = MainWindow(bridge, store, _Pipe(), lambda: None, lambda: None)
    try:
        store.config.translate.model = "m2m100-418M-int8"
        window.reload_from_config()

        assert store.config.translate.targets == ["Chinese Traditional", "Japanese"]
        assert window._target_combos[0].currentText() == "Chinese Simplified"

        store.config.translate.model = "nllb-600M-int8"
        window.reload_from_config()

        assert window._target_combos[0].currentText() == "Chinese Traditional"
    finally:
        window.close()
        bridge.detach()
