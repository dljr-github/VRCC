"""How an over-limit message is carved into parts.

Split from ``test_chatbox_overflow`` for the file-length cap. These cover which
part each language's share lands in, which is a different question from whether
a message overflows at all.
"""

from __future__ import annotations

from tests.test_chatbox_overflow import make_cfg
from vrcc.osc.chatbox import fit_message
from vrcc.osc.chatbox_format import CHATBOX_LIMIT


def test_fit_message_split_carves_a_thai_translation_across_every_part():
    """Thai is written without spaces, so .split() returns one token that the
    word packer accepts whole and drops into a single slice, leaving every
    other part without a word of it. The same reasoning as CJK, which the
    character branch already handles."""
    cfg = make_cfg(overflow="split")
    original = " ".join(f"word{i}" for i in range(55))
    thai = (
        "สวัสดีครับทุกคนวันนี้อากาศดีมากผมกำลังทดสอบระบบ"
        "คำบรรยายภาษาไทยอยู่ครับหวังว่าจะใช้งานได้ดี"
    )

    parts = fit_message(original, [("TH", thai)], cfg)

    assert len(parts) >= 2
    assert all(len(part) <= CHATBOX_LIMIT for part in parts)
    assert all(any(ch in part for ch in thai) for part in parts), (
        "every part must carry a share of the Thai translation"
    )
    carried = "".join(part.split("\n")[-1] for part in parts)
    assert carried == thai, "the slices must reproduce the translation exactly"


def test_fit_message_split_leaves_a_sliced_translation_in_the_final_part():
    """Parts drain one at a time and a new utterance clears the queue, so the
    last part is the one still on screen once the user stops talking. A
    translation with fewer words than parts cannot both lead and persist, and
    persisting is the half worth having: a reader who is still being talked
    over never reaches the later parts either way. A translation long enough to
    fill every slice does both, which is what
    test_fit_message_split_puts_an_unrepeatable_translation_in_the_first_part
    pins."""
    cfg = make_cfg(overflow="split")
    original = " ".join(f"word{i}" for i in range(50))
    translation = " ".join("T" * 40 for _ in range(3))  # too long to repeat

    parts = fit_message(original, [("X", translation)], cfg)

    assert len(parts) >= 3
    assert all(len(part) <= CHATBOX_LIMIT for part in parts)
    assert "T" * 40 not in parts[0], "a 3-word translation cannot fill 4 parts"
    assert "T" * 40 in parts[-1], "the persisting part must carry the translation"


def test_end_anchoring_never_costs_a_repeat():
    """Anchoring changes which original slice a translation shares a part with,
    so a part count that only fits because of it would be chosen over a larger
    one that repeats the translation outright. Repeating measured better, so
    the part count is judged before anchoring is applied."""
    cfg = make_cfg(overflow="split")
    original = " ".join(f"word{i}" for i in range(60))
    translation = "これはテストです"

    parts = fit_message(original, [("JP", translation)], cfg)

    assert all(translation in part for part in parts), (
        "a translation short enough to repeat must still be in every part"
    )
