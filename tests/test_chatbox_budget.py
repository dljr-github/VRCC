"""How a too-long chatbox message is shaped, and who decides.

"auto" reads the message and answers for it; the three explicit modes are
overrides and are taken at their word. Where the answer is to shorten, the
limit is divided between the targets rather than spent front to back on the
joined string, which used to leave whichever target came last with nothing.
These pin that division, the markers it leaves, and the mode each length
resolves to.
"""

import unicodedata

from vrcc.core.bus import EventBus
from vrcc.core.config import OscConfig
from vrcc.osc.chatbox import ChatboxSender
from vrcc.osc.chatbox_format import (
    CHATBOX_LIMIT,
    _MAX_REPEAT_PARTS,
    fit_message,
    format_message,
    resolve_overflow,
)

# Sized so any two of them already fill the 144-char limit: whichever target
# is last then starts past the cut and arrives with nothing.
JA = "七時ごろ噴水の近くで待ち合わせて、それから一緒に新しいお店まで歩いていくのはどうかな"
ES = (
    "Estaba pensando que podriamos vernos cerca de la fuente sobre las siete "
    "y luego ir andando juntos"
)
DE = (
    "Ich habe mir ueberlegt, dass wir uns gegen sieben Uhr in der Naehe des "
    "Brunnens treffen koennten und dann zusammen hinueberlaufen"
)
ORIGINAL = "So I was thinking we could meet up near the fountain"


def make_idle_sender(cfg: OscConfig) -> ChatboxSender:
    """A `ChatboxSender` whose worker thread is never started -- enough to
    call `submit_message()` and inspect `._queue`."""
    return ChatboxSender(cfg, EventBus(), client_factory=lambda ip, port: object())


def test_every_target_arrives_when_three_overflow_together():
    # The last target in order was cut away entirely: the 144-char limit is
    # spent front to back on the joined string.
    cfg = OscConfig(overflow="truncate")
    parts = fit_message(ORIGINAL, [("JP", JA), ("ES", ES), ("DE", DE)], cfg)

    assert len(parts) == 1
    assert len(parts[0]) <= CHATBOX_LIMIT
    lines = [line for line in parts[0].split("\n") if line]
    assert len(lines) == 3, lines
    assert lines[0].startswith(JA[:6])
    assert lines[1].startswith(ES[:6])
    assert lines[2].startswith(DE[:6])


def test_order_does_not_decide_who_survives():
    # Whichever target sat last lost everything, so the same three must
    # survive under every ordering.
    cfg = OscConfig(overflow="truncate")
    targets = [("JP", JA), ("ES", ES), ("DE", DE)]
    for rotation in range(3):
        rotated = targets[rotation:] + targets[:rotation]
        parts = fit_message(ORIGINAL, rotated, cfg)
        lines = [line for line in parts[0].split("\n") if line]
        assert len(lines) == 3, (rotation, lines)
        for (_, text), line in zip(rotated, lines):
            assert line.startswith(text[:6]), (rotation, line)


def test_a_short_target_is_never_shortened():
    # Water-filling: a target inside its share keeps all of it and releases
    # the surplus, so only the long ones pay.
    cfg = OscConfig(overflow="truncate")
    short = "はい"
    parts = fit_message(
        ORIGINAL, [("JP", short), ("A", "a" * 120), ("B", "b" * 120)], cfg
    )

    lines = [line for line in parts[0].split("\n") if line]
    assert short in lines
    assert len(parts[0]) <= CHATBOX_LIMIT
    a_line = next(line for line in lines if line.startswith("a"))
    b_line = next(line for line in lines if line.startswith("b"))
    assert abs(len(a_line) - len(b_line)) <= 1


def test_sharing_spends_the_room_it_has():
    # An equal division is not enough on its own: capping every target at some
    # small constant divides evenly too and wastes most of the box. The point
    # of the fix is that what fits is used.
    cfg = OscConfig(overflow="truncate", include_original=False)
    parts = fit_message(ORIGINAL, [("A", "a" * 120), ("B", "b" * 120)], cfg)

    assert len(parts[0]) >= CHATBOX_LIMIT - 2
    lines = [line for line in parts[0].split("\n") if line]
    assert len(lines) == 2
    for line in lines:
        assert len(line) >= (CHATBOX_LIMIT - 1) // 2 - 1, lines


def test_a_shortened_target_says_so():
    # Without the marker a reader cannot tell a cut line from a short one.
    cfg = OscConfig(overflow="truncate", include_original=False)
    parts = fit_message(ORIGINAL, [("A", "a" * 120), ("B", "b" * 120)], cfg)

    lines = [line for line in parts[0].split("\n") if line]
    assert len(lines) == 2
    for line in lines:
        assert line.endswith("…"), lines


def test_a_cut_lands_on_a_word_boundary():
    # Two more characters are not worth a word sawn in half. Spaceless
    # scripts have no boundary to find and keep the character cut.
    cfg = OscConfig(overflow="truncate", include_original=False)
    parts = fit_message(ORIGINAL, [("ES", ES), ("DE", DE)], cfg)

    for line, source in zip(parts[0].split("\n"), (ES, DE)):
        body = line.rstrip("…")
        assert not body.endswith(" "), body
        rest = source[len(body):]
        assert rest == "" or rest.startswith(" "), (body[-24:], rest[:12])


def test_a_cut_keeps_a_combining_mark_with_its_base():
    # Thai tone marks are the case that reaches linebreak.safe_cut: a boundary
    # landing on one backs up so the base and its mark travel together or not
    # at all.
    cfg = OscConfig(overflow="truncate", include_original=False)
    thai = "ก" * 70 + "้" + "ก" * 60
    parts = fit_message(ORIGINAL, [("TH", thai), ("X", "x" * 131)], cfg)

    line = next(line for line in parts[0].split("\n") if line.startswith("ก"))
    body = line.rstrip("…")
    assert thai.startswith(body)
    assert unicodedata.combining(thai[len(body)]) == 0


def test_send_mode_still_sends_the_whole_thing():
    # "Send full (may be cut off in VRChat)" promises the untouched text.
    for include_original in (True, False):
        cfg = OscConfig(overflow="send", include_original=include_original)
        parts = fit_message(ORIGINAL, [("ES", ES), ("DE", DE)], cfg)

        assert len(parts) == 1, include_original
        assert ES in parts[0], include_original
        assert DE in parts[0], include_original


def test_send_mode_does_not_shorten_the_original():
    # Budgeting the original away silently overrode the mode: picking "Send
    # full" is a choice to hand VRChat the whole string and let it do the
    # cutting, including when the original is the long part.
    cfg = OscConfig(overflow="send")
    original = "x" * 145
    translations = [("JP", "テスト")]
    raw = format_message(original, translations, cfg)
    assert len(raw) > CHATBOX_LIMIT  # sanity: the fixture overflows

    assert fit_message(original, translations, cfg) == [raw]


def test_send_mode_is_not_badged_as_shortened():
    # The caption log paints the truncated flag as "sent / shortened to fit".
    # Send hands VRChat the whole string, so claiming VRCC shortened it points
    # the reader at the wrong culprit for a missing translation.
    cfg = OscConfig(overflow="send")
    sender = make_idle_sender(cfg)
    translations = [("JP", "t" * 60)]
    original = "o" * 100
    assert len(format_message(original, translations, cfg)) > CHATBOX_LIMIT

    sender.submit_message(original, translations, 1)

    (chunks, _utterance_id, truncated) = list(sender._queue)[0][:3]
    assert truncated is False
    assert chunks == format_message(original, translations, cfg)


def test_single_target_is_unchanged():
    # Nothing to share against, so this stays the plain flat cut.
    cfg = OscConfig(overflow="truncate", include_original=False)
    parts = fit_message(ORIGINAL, [("DE", "d" * 200)], cfg)

    assert len(parts) == 1
    assert len(parts[0]) == CHATBOX_LIMIT
    assert parts[0].endswith("…")


def test_auto_keeps_a_message_that_already_fits_in_one_part():
    assert resolve_overflow("x" * CHATBOX_LIMIT, "auto") == "split"
    cfg = OscConfig(overflow="auto", include_original=False)
    assert fit_message("", [("ES", "short enough")], cfg) == ["short enough"]


def test_auto_sends_parts_while_they_will_all_be_read():
    # Splitting loses nothing, so it wins for as long as the queue drains
    # before a new utterance clears it.
    text = "x" * (CHATBOX_LIMIT * _MAX_REPEAT_PARTS)
    assert resolve_overflow(text, "auto") == "split"

    cfg = OscConfig(overflow="auto", include_original=False)
    parts = fit_message("", [("ES", " ".join(["word"] * 70))], cfg)
    assert len(parts) > 1
    assert all(len(part) <= CHATBOX_LIMIT for part in parts)


def test_auto_shortens_a_message_too_long_to_arrive_as_parts():
    # Past this the tail is never read, so a queue of parts costs the reader
    # the wait and delivers a fragment anyway.
    text = "x" * (CHATBOX_LIMIT * _MAX_REPEAT_PARTS + 1)
    assert resolve_overflow(text, "auto") == "truncate"

    cfg = OscConfig(overflow="auto", include_original=False)
    parts = fit_message("", [("ES", " ".join(["word"] * 200))], cfg)
    assert len(parts) == 1
    assert len(parts[0]) <= CHATBOX_LIMIT


def test_auto_badges_a_caption_only_when_it_shortened_one():
    cfg = OscConfig(overflow="auto", include_original=False)
    for text, expected in ((" ".join(["word"] * 70), False),
                           (" ".join(["word"] * 200), True)):
        sender = make_idle_sender(cfg)
        sender.submit_message("", [("ES", text)], 1)
        flags = {item[2] for item in sender._queue}
        assert flags == {expected}, (len(text), flags)


def test_an_explicit_mode_is_never_second_guessed():
    # Auto is a default, not a policy imposed on someone who chose.
    for mode in ("split", "truncate", "send"):
        for size in (10, CHATBOX_LIMIT * 9):
            assert resolve_overflow("x" * size, mode) == mode
