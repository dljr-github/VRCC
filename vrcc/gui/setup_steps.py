"""The setup check panel's rows and the pure evaluator behind them.

Qt-free on purpose: the panel sits beside the main window, but the widget
stays thin. The evidence rules that decide whether a row ticks green live
here where they can be tested with no display and read without a debugger.

Each row lands on one of three states. ``pending`` means the evidence hasn't
shown up yet but may on its own (loading, waiting for VRChat to appear,
waiting for the user to speak). ``attention`` means the row needs something
from the user, either because a real failure was reported (the voice model
failed to load) or because the plain next step is theirs to take (turn on
captioning, check the microphone). ``pass`` means the bus already reported the
evidence this row exists to check for.

Every fact here is real bus evidence, not a guess. Three things about the bus
make that harder than it looks: ``EngineStateChanged`` carries the translator
on the same signal as the voice model (translate/engine.py:96,110-114,
forwarded on one Qt signal at bridge.py:82), so the "model" row reads only
the "stt" key. ``PhraseRecognized``
and ``ChatboxSent`` both fire for typed text sent from the compose box
(pipeline_typed.py:61-69, with a negative utterance id from
pipeline.py:461-469), so "voice" and "chatbox" both need a caller that already
filtered typed sends out before setting ``spoken_utterance`` /
``spoken_chatbox_send``. And captioning has no bus event at all
(``Pipeline.set_captioning`` at pipeline.py:246-249 publishes nothing), so the
"captioning" row is a plain boolean the caller reads off the toggle widget.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from vrcc.i18n import tr, tr_noop

ROWS: tuple[str, ...] = ("captioning", "model", "voice", "vrchat", "chatbox", "heard")

# "heard" is evidence-only and never blocks anything: it has no send path of
# its own to prove, just a meter reacting to whatever the speakers play.
REQUIRED: frozenset[str] = frozenset({"captioning", "model", "voice", "vrchat", "chatbox"})


@dataclass
class SetupFacts:
    """A snapshot of everything the six rows need. Every field defaults to
    the "nothing has happened yet" value, so a bare ``SetupFacts()`` is a
    valid, fully pending state rather than a partially-built one."""

    captioning: bool = False

    # Keyed "stt" / "mt", values "loading" | "ready" | "failed" | "fallback_cpu",
    # mirroring EngineStateChanged verbatim. A single bool here would be the
    # "model" row's exact trap: the translator publishes on the same event, so
    # the row keys off engine_states["stt"] and nothing else.
    engine_states: dict[str, str] = field(default_factory=dict)

    # MicLevel with rms > 0: the mic is picking up sound, said or not.
    mic_seen: bool = False
    # PhraseRecognized for a SPOKEN utterance (utterance_id > 0). The caller
    # is responsible for that filter; this field must never be set from a
    # typed Send.
    spoken_utterance: bool = False

    vrchat_found: bool = False

    # ChatboxSent for a spoken utterance, same filter as spoken_utterance.
    spoken_chatbox_send: bool = False
    # ChatboxSent for ANY utterance, typed included. Kept separate (and read
    # by nothing below) so a caller, or a test, cannot satisfy the row by
    # wiring this one up by mistake; only the spoken-filtered field counts.
    any_chatbox_send: bool = False
    # config.osc.send_to_vrchat. Defaults True to match config.py:97; when
    # False nothing is ever queued (pipeline_send.py:24-26), so "chatbox"
    # drops out of the required set entirely rather than block forever.
    send_enabled: bool = True

    heard_seen: bool = False


def _captioning_state(facts: SetupFacts) -> str:
    # No bus event carries this; do not key it on Pipeline._should_caption
    # (pipeline.py:406-411), which also reads False mid model-swap and while
    # mute sync holds captions, and would untick this row on every VRChat mute.
    return "pass" if facts.captioning else "attention"


def _model_state(facts: SetupFacts) -> str:
    state = facts.engine_states.get("stt")
    if state == "ready":
        return "pass"
    if state == "failed":
        return "attention"
    # None (never reported yet), "loading", or "fallback_cpu". A CPU rebuild
    # after an unusable CUDA device is always followed by another "ready"
    # publish, so treating it as a failure would flash a false attention
    # state mid-load.
    return "pending"


def _voice_state(facts: SetupFacts) -> str:
    if facts.spoken_utterance:
        return "pass"
    if not facts.captioning:
        return "pending"  # nothing to check until captioning is even on
    if facts.mic_seen:
        return "pending"  # sound is coming through, just no recognized phrase yet
    return "attention"  # captioning is on and the mic looks silent


def _vrchat_state(facts: SetupFacts) -> str:
    # The detector publishes False the instant it starts (vrchat_detect.py:49),
    # so this reads "pending" within a second of every healthy launch. That is
    # expected, not a fault, which is why there is no attention state here.
    return "pass" if facts.vrchat_found else "pending"


def _chatbox_state(facts: SetupFacts) -> str:
    # A send failure returns silently (chatbox.py:367-375) and OSC has no ack
    # at all (events.py:50-52), so there is no signal an attention state could
    # be built from: absence of a spoken send is indistinguishable from a
    # send that failed. Both read as "pending".
    return "pass" if facts.spoken_chatbox_send else "pending"


def _heard_state(facts: SetupFacts) -> str:
    return "pass" if facts.heard_seen else "pending"


_ROW_EVALUATORS = {
    "captioning": _captioning_state,
    "model": _model_state,
    "voice": _voice_state,
    "vrchat": _vrchat_state,
    "chatbox": _chatbox_state,
    "heard": _heard_state,
}


def evaluate(facts: SetupFacts) -> dict[str, str]:
    """Every row id mapped to "pass", "attention" or "pending"."""
    return {row_id: _ROW_EVALUATORS[row_id](facts) for row_id in ROWS}


def required_passed(facts: SetupFacts) -> bool:
    """Whether every row that currently matters has passed.

    "heard" never counts (see REQUIRED). "chatbox" drops out too, but only
    when sending to VRChat is off: nothing will ever be queued in that case
    (pipeline_send.py:24-26), so requiring it would block forever on a
    setting the user chose on purpose.
    """
    states = evaluate(facts)
    for row_id in REQUIRED:
        if row_id == "chatbox" and not facts.send_enabled:
            continue
        if states[row_id] != "pass":
            return False
    return True


# Row labels: one per row id, stable across every state that row can reach.
_HEADLINES: dict[str, str] = {
    "captioning": tr_noop("Captioning"),
    "model": tr_noop("Speech recognition"),
    "voice": tr_noop("Your voice"),
    "vrchat": tr_noop("VRChat"),
    "chatbox": tr_noop("Chatbox"),
    "heard": tr_noop("What VRCC hears"),
}

# One detail sentence per row per state it can actually reach; a row with no
# attention state (vrchat, chatbox, heard) simply has no entry for one.
_DETAILS: dict[str, dict[str, str]] = {
    "captioning": {
        "attention": tr_noop("Turn it on so VRChat can see your words."),
        "pass": tr_noop("Speech turns into chat text while this stays on."),
    },
    "model": {
        "pending": tr_noop(
            "Your voice model is loading. This can take a minute the first time."
        ),
        "attention": tr_noop(
            "The voice model did not load. Pick a different one in Settings, or restart VRCC."
        ),
        "pass": tr_noop("Your voice model finished loading and is ready."),
    },
    "voice": {
        # True whether or not the mic has already picked up sound: this text
        # covers both "captioning is off" and "captioning is on, sound seen,
        # no phrase yet", and must not go stale in either case.
        "pending": tr_noop(
            "With captioning on, say something out loud. "
            "This ticks once we hear a full sentence."
        ),
        "attention": tr_noop(
            "Captioning is on, but nothing is coming through your microphone. "
            "Check your input device."
        ),
        # No confidence or audio-quality field feeds this row, so the copy
        # claims only that a phrase was recognized, not how well.
        "pass": tr_noop("We heard you speak."),
    },
    "vrchat": {
        "pending": tr_noop("Not found yet. It may still be loading, so keep this window open."),
        # An mDNS advert is what was actually seen, not a live link to VRChat
        # (vrchat_detect.py's own docstring calls it a proxy); "advertised"
        # names that evidence instead of overstating it.
        "pass": tr_noop("Found on the network. Its OSC service is advertised here."),
    },
    "chatbox": {
        "pending": tr_noop("Nothing sent yet. Speak a full sentence and we will try sending it."),
        # Says sent, never arrived: OSC is fire-and-forget UDP with no
        # delivery ack (events.py:50-52) and a failed write returns silently
        # (chatbox.py:367-375), so this can only claim the write happened,
        # never that VRChat picked it up.
        "pass": tr_noop("Sent over OSC. There is no way to check that VRChat picked it up."),
    },
    "heard": {
        "pending": tr_noop("Nothing picked up from your speakers yet."),
        "pass": tr_noop(
            "Picking up sound from your speakers. This one is optional and never blocks the rest."
        ),
    },
}


def row_text(row_id: str, state: str) -> tuple[str, str]:
    """The row's label and a state-specific detail sentence, both read
    through tr() here rather than resolved at import time: this module
    loads before set_language runs, and there is no retranslate path, so
    text resolved too early would stay English for the life of the process.
    """
    headline = _HEADLINES.get(row_id, row_id)
    details = _DETAILS.get(row_id, {})
    detail = details.get(state)
    if detail is None:
        # A (row, state) pair that row never actually produces. Fall back to
        # whatever this row does define rather than raising: a caller must
        # never crash the panel over an impossible combination.
        detail = next(iter(details.values()), "")
    return tr(headline), tr(detail)
