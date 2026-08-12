"""The "caption what I hear" controls on the Simple settings page.

Split from ``settings_audio`` because this is the one audio control that is not
about the microphone: it captures the speakers, so other people in VRChat can be
read as well as heard.

On/off lives on the main window, next to the captioning toggle, because it is
something you reach for mid-conversation. What is left here is the setup you
choose once: which output to listen to, and which language to read it in.

Two things the UI has to be honest about, because neither is guessable and both
will otherwise read as bugs. What gets captured is the whole output device, not
VRChat's voice channel, so world audio and music are transcribed too. And it is
a second transcription stream sharing one voice model, so on a machine without
a graphics card it competes with captioning the user's own speech.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtWidgets import QComboBox, QLabel

from vrcc.core import recommend
from vrcc.core.languages import LANGUAGES
from vrcc.gui.widgets import fill_spoken_languages, no_wheel
from vrcc.i18n import tr, tr_noop

if TYPE_CHECKING:
    from PySide6.QtWidgets import QFormLayout

    from vrcc.gui.settings import SettingsDialog

_DEFAULT_SPEAKER = ""

_HEAR_TIP = tr_noop(
    "Read what other people are saying, translated into your language. This "
    "captures everything your speakers play, including game and world audio, "
    "and it is only shown in this window."
)
# Shown on a machine with no graphics card, where the second stream shares one
# voice model with the user's own captions. Not conditional on the feature
# being on: it is the reason someone might decide not to turn it on.
_HEAR_CPU_WARNING = tr_noop(
    "Without a graphics card this shares the voice model with your own "
    "captions, so both will be slower."
)
_SPEAKER_LABEL = tr_noop("Listen to")
_TARGET_LABEL = tr_noop("Show it in")
# Empty value: follow the spoken language rather than pin one. The wording says
# what it resolves to, because "Auto" alone would not say into which language.
_TARGET_AUTO = tr_noop("The language I speak")
_SPEAKER_DEFAULT = tr_noop("Default speakers")
_NO_SPEAKERS = tr_noop("No speakers found")


def build_heard_controls(dlg: "SettingsDialog", form: "QFormLayout") -> None:
    """Add the speaker picker and the language its captions are shown in."""
    combo = no_wheel(QComboBox())
    dlg._hear_device_combo = combo
    _fill_speakers(combo, dlg._cfg.audio.hear_others_device)

    target = no_wheel(QComboBox())
    dlg._hear_target_combo = target
    fill_spoken_languages(target, tr(_TARGET_AUTO), "", LANGUAGES.keys())
    at = target.findData(dlg._cfg.audio.hear_others_language)
    target.setCurrentIndex(at if at >= 0 else 0)

    def on_device(_index: int) -> None:
        if dlg._loading:
            return
        dlg._cfg.audio.hear_others_device = combo.currentData() or _DEFAULT_SPEAKER
        dlg._changed()

    def on_target(_index: int) -> None:
        if dlg._loading:
            return
        dlg._cfg.audio.hear_others_language = target.currentData() or ""
        dlg._changed()

    combo.currentIndexChanged.connect(on_device)
    target.currentIndexChanged.connect(on_target)

    dlg._hear_note = QLabel(tr(_HEAR_CPU_WARNING))
    dlg._hear_note.setWordWrap(True)
    dlg._hear_note.setStyleSheet(dlg._muted_style)

    combo.setToolTip(tr(_HEAR_TIP))
    form.addRow(tr(_SPEAKER_LABEL), combo)
    form.addRow(tr(_TARGET_LABEL), target)
    form.addRow("", dlg._hear_note)
    dlg._hear_note.setVisible(_on_cpu(dlg))


def _on_cpu(dlg: "SettingsDialog") -> bool:
    """Whether this machine has no graphics card the engines can use.

    Through the recommender's own tier rather than a second reading of the
    hardware, so the warning and the model recommendation cannot disagree about
    what this PC is.
    """
    try:
        return recommend.detect_tier(dlg._cfg.stt.device_index) == "cpu"
    except Exception:
        return False


def _fill_speakers(combo: QComboBox, selected: str) -> None:
    """Fill with the capturable outputs, default first.

    Names carry in the data role because that is what config stores and what
    soundcard resolves; an index would not survive plugging in a headset.
    """
    from vrcc.audio.loopback import loopback_devices

    combo.clear()
    combo.addItem(tr(_SPEAKER_DEFAULT), _DEFAULT_SPEAKER)
    devices = loopback_devices()
    for name in devices:
        combo.addItem(name, name)
    if not devices:
        combo.addItem(tr(_NO_SPEAKERS), _DEFAULT_SPEAKER)
        combo.model().item(combo.count() - 1).setEnabled(False)
    index = combo.findData(selected)
    if index >= 0:
        combo.setCurrentIndex(index)
