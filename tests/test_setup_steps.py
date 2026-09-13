"""The setup check evaluator: every row in every reachable state, four traps
in the bus events that back it, and a copy guard over every tr_noop string
the module defines.
"""

from __future__ import annotations

import ast
import re
from dataclasses import replace
from pathlib import Path

import pytest

from vrcc.gui.setup_steps import (
    REQUIRED,
    ROWS,
    SetupFacts,
    evaluate,
    required_passed,
    row_text,
)
from vrcc.i18n.extract import extract_from_source

_MODULE_PATH = Path(__file__).resolve().parent.parent / "vrcc" / "gui" / "setup_steps.py"


# -- shape --------------------------------------------------------------


def test_rows_are_exactly_the_six_ids_in_order():
    assert ROWS == ("captioning", "model", "voice", "vrchat", "chatbox", "heard")


def test_required_is_the_five_ids_without_heard():
    assert REQUIRED == frozenset({"captioning", "model", "voice", "vrchat", "chatbox"})
    assert "heard" not in REQUIRED


def test_bare_facts_is_nothing_known_yet():
    facts = SetupFacts()
    states = evaluate(facts)
    assert set(states) == set(ROWS)
    assert states["model"] == "pending"
    assert states["voice"] == "pending"
    assert states["vrchat"] == "pending"
    assert states["chatbox"] == "pending"
    assert states["heard"] == "pending"
    assert required_passed(facts) is False


# -- captioning -----------------------------------------------------------


def test_captioning_off_is_attention_not_an_error_state():
    assert evaluate(SetupFacts(captioning=False))["captioning"] == "attention"


def test_captioning_on_passes():
    assert evaluate(SetupFacts(captioning=True))["captioning"] == "pass"


# -- model ------------------------------------------------------------------


def test_model_row_ignores_the_translator():
    """EngineStateChanged carries both engines on one signal, so a row keyed on
    state alone goes green when only the translator loaded."""
    facts = SetupFacts(engine_states={"mt": "ready"})
    assert evaluate(facts)["model"] != "pass"
    facts = SetupFacts(engine_states={"stt": "ready", "mt": "failed"})
    assert evaluate(facts)["model"] == "pass"


@pytest.mark.parametrize("state", ["loading", "fallback_cpu"])
def test_model_transient_states_are_pending_not_attention(state):
    """fallback_cpu is always followed by another ready; flagging it as a
    failure would flash a false alarm mid-load."""
    assert evaluate(SetupFacts(engine_states={"stt": state}))["model"] == "pending"


def test_model_failed_is_attention():
    assert evaluate(SetupFacts(engine_states={"stt": "failed"}))["model"] == "attention"


# -- voice --------------------------------------------------------------


def test_voice_row_is_not_satisfied_by_typed_text():
    """Typed text publishes PhraseRecognized with a negative utterance_id,
    bypassing captioning and mute gating. Without the filter a user passes this
    row by typing into the compose box without ever speaking. captioning=True
    puts the ladder past its first branch so this actually reaches the
    mic_seen check the test is named for, rather than returning early."""
    facts = SetupFacts(captioning=True, mic_seen=True, spoken_utterance=False)
    assert evaluate(facts)["voice"] != "pass"


def test_voice_pending_before_captioning_is_even_on():
    facts = SetupFacts(captioning=False, mic_seen=False, spoken_utterance=False)
    assert evaluate(facts)["voice"] == "pending"


def test_voice_pending_while_sound_is_coming_through():
    facts = SetupFacts(captioning=True, mic_seen=True, spoken_utterance=False)
    assert evaluate(facts)["voice"] == "pending"


def test_voice_attention_when_captioning_is_on_and_mic_looks_silent():
    facts = SetupFacts(captioning=True, mic_seen=False, spoken_utterance=False)
    assert evaluate(facts)["voice"] == "attention"


def test_voice_passes_on_a_spoken_utterance():
    facts = SetupFacts(captioning=True, mic_seen=True, spoken_utterance=True)
    assert evaluate(facts)["voice"] == "pass"


# -- vrchat -------------------------------------------------------------


def test_vrchat_not_found_is_pending_never_attention():
    assert evaluate(SetupFacts(vrchat_found=False))["vrchat"] == "pending"


def test_vrchat_found_passes():
    assert evaluate(SetupFacts(vrchat_found=True))["vrchat"] == "pass"


# -- chatbox --------------------------------------------------------------


def test_chatbox_row_is_not_satisfied_by_typed_text():
    facts = SetupFacts(spoken_chatbox_send=False, any_chatbox_send=True)
    assert evaluate(facts)["chatbox"] != "pass"


def test_chatbox_passes_on_a_spoken_send():
    facts = SetupFacts(spoken_chatbox_send=True)
    assert evaluate(facts)["chatbox"] == "pass"


def test_chatbox_drops_out_of_required_when_sending_is_off():
    facts = SetupFacts(
        captioning=True,
        engine_states={"stt": "ready"},
        spoken_utterance=True,
        vrchat_found=True,
        send_enabled=False,
    )
    assert required_passed(facts) is True
    # Still not lied about: the row itself stays pending, just not required.
    assert evaluate(facts)["chatbox"] == "pending"


def test_chatbox_still_required_when_sending_is_on():
    facts = SetupFacts(
        captioning=True,
        engine_states={"stt": "ready"},
        spoken_utterance=True,
        vrchat_found=True,
        send_enabled=True,
    )
    assert required_passed(facts) is False


# -- heard ------------------------------------------------------------------


def test_heard_never_gates_required():
    facts = SetupFacts(
        captioning=True,
        engine_states={"stt": "ready"},
        spoken_utterance=True,
        vrchat_found=True,
        spoken_chatbox_send=True,
    )
    assert required_passed(facts) is True
    assert evaluate(facts)["heard"] != "pass"


def test_heard_passes_when_seen():
    assert evaluate(SetupFacts(heard_seen=True))["heard"] == "pass"


# -- everything required together -------------------------------------------


def test_required_passed_true_only_once_every_required_row_passes():
    facts = SetupFacts(
        captioning=True,
        engine_states={"stt": "ready"},
        spoken_utterance=True,
        vrchat_found=True,
        spoken_chatbox_send=True,
    )
    assert required_passed(facts) is True
    for field_name in ("captioning", "spoken_utterance", "vrchat_found", "spoken_chatbox_send"):
        broken = replace(facts, **{field_name: False})
        assert required_passed(broken) is False
    broken_model = replace(facts, engine_states={})
    assert required_passed(broken_model) is False


# -- row_text -----------------------------------------------------------


_ALL_FACTS_COMBINATIONS = [
    SetupFacts(),
    SetupFacts(captioning=True),
    SetupFacts(engine_states={"stt": "loading"}),
    SetupFacts(engine_states={"stt": "ready"}),
    SetupFacts(engine_states={"stt": "failed"}),
    SetupFacts(captioning=True, mic_seen=True),
    SetupFacts(captioning=True, spoken_utterance=True),
    SetupFacts(vrchat_found=True),
    SetupFacts(spoken_chatbox_send=True),
    SetupFacts(heard_seen=True),
]


def test_row_text_is_nonempty_for_every_state_actually_produced():
    for facts in _ALL_FACTS_COMBINATIONS:
        states = evaluate(facts)
        for row_id, state in states.items():
            headline, detail = row_text(row_id, state)
            assert headline, (row_id, state)
            assert detail, (row_id, state)


def test_row_text_falls_back_instead_of_raising_on_an_unreachable_state():
    # "vrchat" never produces "attention"; row_text must not crash if asked.
    headline, detail = row_text("vrchat", "attention")
    assert headline
    assert detail


def test_row_text_falls_back_on_an_unknown_row_id():
    headline, detail = row_text("nonsense", "pending")
    assert headline == "nonsense"
    assert detail == ""


# -- Qt-free ------------------------------------------------------------


def test_module_imports_no_qt():
    tree = ast.parse(_MODULE_PATH.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module or ""]
        else:
            continue
        for name in names:
            assert "PySide" not in name
            assert "Qt" not in name


# -- copy guard -----------------------------------------------------------

_BANNED_DASH_CHARS = ("—", "–", "―")  # em dash, en dash, horizontal bar
# Stems, not whole words. OSC is fire-and-forget UDP with no delivery ack and
# an mDNS advert proves only that a service announced itself, so no row may
# claim a caption got anywhere. A whole-word list let "reachable", "linked"
# and every other inflection straight through; the leading \b stays so the
# stem has to start a word ("recognition" and "picked" are not overclaims).
_BANNED_STEMS = re.compile(
    r"\b(arriv|connect|confirm|deliver|link|reach|receiv)", re.IGNORECASE
)


def test_the_copy_guard_catches_the_words_it_exists_for():
    """The guard itself, since a pattern that matches nothing would let every
    string below pass. Inflections, not just the bare stems."""
    for overclaim in (
        "It arrived in VRChat.",
        "Connected to VRChat.",
        "We confirmed VRChat got it.",
        "Delivered to the chatbox.",
        "Linked to VRChat.",
        "VRChat is reachable.",
        "VRChat reached us.",
        "Received by VRChat.",
    ):
        assert _BANNED_STEMS.search(overclaim), overclaim


def test_copy_guard_over_every_tr_noop_string():
    source = _MODULE_PATH.read_text(encoding="utf-8")
    strings = [text for text, _ in extract_from_source(source, str(_MODULE_PATH))]
    assert len(strings) >= 20, "expected the row labels and per-state detail strings"
    for text in strings:
        assert not any(ch in text for ch in _BANNED_DASH_CHARS), text
        assert not _BANNED_STEMS.search(text), text
