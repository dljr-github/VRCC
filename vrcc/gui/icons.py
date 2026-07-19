"""Inline SVG icon builders and user-facing copy assets shared across the GUI.

Single home for every hand-drawn SVG string used as a button/label icon (no
external asset files) -- rendered via QSvgRenderer elsewhere (see
``vrcc.gui.widgets.svg_pixmap``). Colors are explicit hexes: QSvgRenderer
can't resolve "currentColor". Also holds copy maps (friendly error text,
tr_noop-marked for translation) that read naturally as icon/asset-adjacent
constants.
"""

from __future__ import annotations

from vrcc.i18n import tr_noop


def mic_svg(color: str) -> str:
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" '
        f'viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" '
        'stroke-linecap="round" stroke-linejoin="round">'
        '<rect x="9" y="2" width="6" height="12" rx="3"/>'
        '<path d="M5 10a7 7 0 0 0 14 0"/><line x1="12" y1="19" x2="12" y2="22"/></svg>'
    )


def arrow_svg(color: str) -> str:
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" '
        f'viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" '
        'stroke-linecap="round" stroke-linejoin="round">'
        '<line x1="4" y1="12" x2="20" y2="12"/><polyline points="14 6 20 12 14 18"/></svg>'
    )


# Inline line icons (no external assets), rendered via QSvgRenderer. Colors are
# explicit hexes: QSvgRenderer can't resolve "currentColor".
def gear_svg(color: str) -> str:
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" '
        f'viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" '
        'stroke-linecap="round" stroke-linejoin="round">'
        '<circle cx="12" cy="12" r="3"/>'
        '<path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83'
        'l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0'
        'v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1'
        '-2.83-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 '
        '0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 '
        '0 1 1 2.83-2.83l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3'
        'a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06'
        'a2 2 0 1 1 2.83 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 '
        '1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"/></svg>'
    )


def dots_svg(color: str) -> str:
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" '
        f'viewBox="0 0 24 24" fill="{color}">'
        '<circle cx="5" cy="12" r="2"/><circle cx="12" cy="12" r="2"/>'
        '<circle cx="19" cy="12" r="2"/></svg>'
    )


def x_svg(color: str) -> str:
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" '
        f'viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" '
        'stroke-linecap="round" stroke-linejoin="round">'
        '<line x1="6" y1="6" x2="18" y2="18"/><line x1="6" y1="18" x2="18" y2="6"/></svg>'
    )


# Human sentences for known AppError codes; the status bar shows these, the log
# keeps the raw code+message (see MainWindow._on_app_error). Unknown codes
# render raw.
FRIENDLY_ERRORS = {
    "PIPELINE_NOT_RUNNING": tr_noop("Engines are still loading. Try again in a moment."),
    "MIC_OPEN_FAILED": tr_noop("Could not open the microphone. Check Settings > Audio."),
    "MUTE_SYNC_REQUIRES_LOCALHOST": tr_noop(
        "Mute sync is off: it only works when OSC points at this machine (127.0.0.1)."
    ),
    "MUTE_SYNC_MDNS_FAILED": tr_noop(
        "Mute sync may not work: VRChat could not be notified (network discovery failed)."
    ),
    "SOURCE_LANG_UNSUPPORTED": tr_noop(
        "Your spoken language could not be matched, sending your words without translation."
    ),
    "DRIVER_TOO_OLD": tr_noop(
        "Your NVIDIA driver is too old for GPU mode; running on CPU. Update to driver 570+."
    ),
    "CHATBOX_SEND_FAILED": tr_noop(
        "Could not send to the VRChat chatbox. Is VRChat running?"
    ),
    "ENGINE_LOAD_FAILED": tr_noop(
        "An engine failed to load. Open Models to re-download, then restart VRCC."
    ),
    "MODEL_SWITCH_FAILED": tr_noop(
        "Switching models failed. The previous model is still in use."
    ),
    "SEGMENTER_FAILED": tr_noop(
        "Audio processing hit an error. Captioning may have stopped; restart VRCC"
        " if it doesn't recover."
    ),
    "STT_JOB_FAILED": tr_noop("Transcription failed for the last utterance."),
    "MT_JOB_FAILED": tr_noop("Translation failed. The original text was sent instead."),
    "HANDLER_ERROR": tr_noop("An internal error occurred. See the log file for details."),
}
