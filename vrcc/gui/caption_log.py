"""Structured, per-utterance caption log with delivery feedback.

Each utterance is one row that updates in place through the pipeline:
recognized -> translated -> sent (with latency) / not sent. CaptionModel is
pure (no Qt) so the row state machine is unit-tested directly; the view is a
thin QTextBrowser that re-renders from it.
"""

from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass, field

# Status values (also the render key).
TRANSLATING = "translating"
QUEUED = "queued"
SENT = "sent"
TRUNCATED = "truncated"
NOT_SENT = "not_sent"

_TERMINAL = frozenset({SENT, TRUNCATED, NOT_SENT})


@dataclass
class CaptionRow:
    key: int
    utterance_id: int
    time_label: str
    original: str
    translations: list[tuple[str, str]] = field(default_factory=list)
    status: str = TRANSLATING
    latency_ms: int | None = None


class CaptionModel:
    """Ordered, capped set of caption rows keyed by utterance.

    recognized() always starts a fresh row (so a reused utterance_id -- typed
    messages all use id 0 -- never overwrites an older entry); translated()/sent()
    update the most recent row for that id.
    """

    def __init__(self, cap: int = 200, clock=time.monotonic, time_label=None) -> None:
        self._rows: "OrderedDict[int, CaptionRow]" = OrderedDict()
        self._by_utt: dict[int, int] = {}
        self._recv: dict[int, float] = {}  # row key -> monotonic recv time
        self._next_key = 0
        self._cap = cap
        self._clock = clock
        self._time_label = time_label or (lambda: time.strftime("%H:%M"))

    def recognized(
        self, utterance_id: int, text: str, *, translate_enabled: bool, send_enabled: bool
    ) -> None:
        key = self._next_key
        self._next_key += 1
        if translate_enabled:
            status = TRANSLATING
        else:
            status = QUEUED if send_enabled else NOT_SENT
        self._rows[key] = CaptionRow(
            key=key,
            utterance_id=utterance_id,
            time_label=self._time_label(),
            original=text,
            status=status,
        )
        self._by_utt[utterance_id] = key
        self._recv[key] = self._clock()
        self._trim()

    def translated(
        self, utterance_id: int, translations, *, send_enabled: bool
    ) -> None:
        row = self._current_row(utterance_id)
        if row is None:
            return
        row.translations = list(translations)
        if row.status not in _TERMINAL:
            row.status = QUEUED if send_enabled else NOT_SENT

    def sent(self, utterance_id: int, truncated: bool) -> None:
        row = self._current_row(utterance_id)
        if row is None:
            return
        row.status = TRUNCATED if truncated else SENT
        recv = self._recv.get(row.key)
        if recv is not None:
            row.latency_ms = int((self._clock() - recv) * 1000)

    def rows(self) -> list[CaptionRow]:
        return list(self._rows.values())

    def clear(self) -> None:
        self._rows.clear()
        self._by_utt.clear()
        self._recv.clear()

    # -- internals ---------------------------------------------------------

    def _current_row(self, utterance_id: int) -> CaptionRow | None:
        key = self._by_utt.get(utterance_id)
        if key is None:
            return None
        return self._rows.get(key)

    def _trim(self) -> None:
        while len(self._rows) > self._cap:
            key, _ = self._rows.popitem(last=False)
            self._recv.pop(key, None)
            # Drop any utterance->key mapping that pointed at the evicted row.
            for utt, k in list(self._by_utt.items()):
                if k == key:
                    del self._by_utt[utt]


# -- status rendering (shared by the view and tests) ------------------------

# Default palette when no theme colors are passed, so single-arg callers keep the dark look.
_DEFAULT_COLORS = {
    "text": "#e6e9f0", "muted": "#98a2b3", "accent": "#3ea6ff",
    "good": "#2ecc71", "warn": "#e0a33e", "bad": "#e5544b", "border": "#2a2e3a",
}


def _latency_inline(row: CaptionRow, c: dict) -> str:
    # Plain single-line statuses ("sent", "queued", ...) fit the latency on
    # the SAME line with a middot between the two words -- only a middot at a
    # line EDGE reads as a stray mark, not one sitting between two words.
    if row.latency_ms is None:
        return ""
    return f' · <span style="color:{c["muted"]};">{row.latency_ms / 1000:.1f}s</span>'


def _latency_block(row: CaptionRow, c: dict) -> str:
    # The truncated status is already multi-line ("sent" / "shortened to
    # fit"), so latency keeps its own trailing line rather than crowding onto
    # "shortened to fit". No leading separator: a middot at a line edge reads
    # as a stray mark.
    if row.latency_ms is None:
        return ""
    return f'<br/><span style="color:{c["muted"]};">{row.latency_ms / 1000:.1f}s</span>'


def status_markup(
    row: CaptionRow, colors: dict | None = None, scale: float = 1.0
) -> tuple[str, str]:
    """(text, css color) for a row's status marker.

    ``scale`` is accepted for signature symmetry with the other renderers; the
    status text itself carries no font-size (that lives on the render cell)."""
    c = {**_DEFAULT_COLORS, **(colors or {})}
    if row.status == SENT:
        return (f"sent{_latency_inline(row, c)}", c["good"])
    if row.status == TRUNCATED:
        # Still a successful send (just clipped to VRChat's 144-char limit), so it
        # keeps "sent" wording. The explicit <br/> keeps "shortened to fit" whole:
        # Qt ignores nowrap/nbsp, so placing the wrap ourselves is the only
        # reliable way. This is the ONLY status that stays multi-line -- plain
        # sent/not_sent/queued/translating render on one line.
        return (f"sent<br/>shortened to fit{_latency_block(row, c)}", c["warn"])
    if row.status == NOT_SENT:
        return ("not sent", c["bad"])
    if row.status == QUEUED:
        return ("queued", c["muted"])
    return ("translating…", c["muted"])


def _esc(text: str) -> str:
    return (
        text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    )


# Fixed column widths. Qt honors the HTML width *attribute*; CSS width on a
# cell is silently ignored. Fixed px (not %) so a narrow window can't squeeze
# the status column; 190 keeps "shortened to fit" on one line even at ~2x scale.
_GUTTER_W = 64
_STATUS_W = 190


def render_rows_html(
    rows: list[CaptionRow], colors: dict | None = None, scale: float = 1.0
) -> str:
    """Full-document HTML for the caption log (pure, unit-testable).

    Three-column table per entry (timestamp gutter, caption + translations,
    right-aligned status). Status gets its own cell because Qt's rich-text engine
    ignores white-space:nowrap; keeping it out of the caption flow is the only
    reliable way to prevent orphaned fragments. scale grows widths + font-sizes.
    """
    c = {**_DEFAULT_COLORS, **(colors or {})}
    gutter_w = round(_GUTTER_W * scale)
    status_w = round(_STATUS_W * scale)
    fs = round(11 * scale)
    blocks = []
    for row in rows:
        marker, color = status_markup(row, c, scale)
        trans = "".join(
            f'<div style="color:{c["muted"]}; margin-top:4px; '
            f'border-left: 2px solid {c["border"]}; padding-left:8px;">{_esc(text)}</div>'
            for _lang, text in row.translations
        )
        blocks.append(
            f'<div style="margin-bottom:10px;">'
            f'<table width="100%" cellspacing="0" cellpadding="0" '
            f'style="border-collapse:collapse;">'
            f"<tr>"
            # font-size must live on inner spans: Qt ignores it on <td>.
            f'<td width="{gutter_w}" style="vertical-align:top;">'
            f'<span style="color:{c["muted"]}; font-size:{fs}px;">{row.time_label}</span>'
            f"</td>"
            f'<td style="vertical-align:top;">'
            f'<span style="font-weight: normal; color:{c["text"]};">{_esc(row.original)}</span>'
            f"{trans}"
            f"</td>"
            f'<td width="{status_w}" style="vertical-align:top; text-align:right; '
            f'padding-left:10px;">'
            f'<span style="font-size:{fs}px; color:{color};">{marker}</span>'
            f"</td>"
            f"</tr>"
            f"</table>"
            f"</div>"
        )
    return "".join(blocks)


def empty_state_html(
    message: str, sub: str = "", colors: dict | None = None, scale: float = 1.0
) -> str:
    """Centered placeholder shown when the log has no rows yet.

    ``message`` is the readable headline; ``sub`` (optional) is a muted
    second line with extra context (e.g. "usually takes a few seconds").
    ``scale`` grows both font-sizes in step with the text-size preset.
    """
    c = {**_DEFAULT_COLORS, **(colors or {})}
    sub_fs = round(12 * scale)
    msg_fs = round(14 * scale)
    sub_html = (
        f'<div style="color:{c["muted"]}; font-size:{sub_fs}px; margin-top:6px;">{_esc(sub)}</div>'
        if sub
        else ""
    )
    return (
        f'<div style="text-align:center; margin-top:64px;">'
        f'<div style="color:{c["text"]}; font-size:{msg_fs}px;">{_esc(message)}</div>'
        f"{sub_html}"
        f"</div>"
    )
