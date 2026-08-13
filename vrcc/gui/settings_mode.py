"""The Speed / Quality Mode control on the Simple settings page.

Split out of ``settings_pages`` to hold that file under the source cap, and
kept together because the three things here answer each other: the control
applies a whole ``apply_profile`` bundle, the description has to say what that
bundle overwrites, and the greying covers the models the bundle cannot tune.
Imports from ``settings`` are type-only.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtWidgets import QLabel

from vrcc.core import calibrate, recommend
from vrcc.core.hardware import resolved_device
from vrcc.gui.model_labels import whisper_display_name
from vrcc.gui.widgets import SegmentedControl
from vrcc.i18n import tr, tr_noop
from vrcc.stt.registry import WHISPER_MODELS

if TYPE_CHECKING:
    from PySide6.QtWidgets import QFormLayout

    from vrcc.gui.settings import SettingsDialog

# Plain-language Speed/Quality explanation (Mode tooltip + visible description).
_MODE_TOOLTIP = tr_noop(
    "Speed shows captions almost instantly. Quality is more accurate and "
    "clips fewer words off the ends of sentences, but each caption takes a "
    "little longer."
)
# Shown to everyone; the prompt in confirm_mode_overwrite fires on top of it
# only when the flip would actually replace a hand-set value, which is what
# lets it be a modal without taxing someone flipping to compare. Names the page
# each field lives on, because "Search width" labels two different rows and the
# profile writes only the voice one.
_MODE_DESC = tr_noop(
    "Speed shows captions almost instantly; Quality is more accurate and "
    "clips fewer words, but each caption takes a little longer. Switching "
    "mode also rewrites the pause timings on the Advanced page and Search "
    "width on the Voice recognition page. Nothing else is touched."
)
# Replaces _MODE_TOOLTIP while the active voice model decodes greedily: the
# onnxruntime backends ignore beam size, which is the profile's only
# caption-quality lever now that the timings are its other half.
_MODE_LOCKED_TOOLTIP = tr_noop(
    "{name} always decodes at full accuracy, so Speed / Quality "
    "does not change its captions."
)
# Appended to _MODE_DESC for a whisper model the benchmarks have an opinion on
# (recommend.recommended_profile); mapped from its "quality"/"latency" verdict.
_MODE_RECOMMEND_QUALITY = tr_noop(
    "Quality is recommended for this model: more accurate, and barely slower."
)
_MODE_RECOMMEND_SPEED = tr_noop(
    "Speed is recommended for this model: Quality is no more accurate here."
)


# The bundle fields, under the label each one carries where the user tuned it.
# Same literals as the row builders, so the prompt names a control they can
# find rather than a config key.
_PROFILE_FIELD_LABELS: dict[tuple[str, str], str] = {
    ("vad", "speculative_silence_ms"): tr_noop("Wait before an early caption (ms)"),
    ("vad", "finalize_silence_ms"): tr_noop("Wait before finishing a caption (ms)"),
    ("vad", "pre_roll_ms"): tr_noop("Keep audio before you start (ms)"),
    ("stt", "beam_size"): tr_noop("Search width"),
}

_MODE_OVERWRITE_TITLE = tr_noop("Replace your own settings?")
_MODE_OVERWRITE_BODY = tr_noop(
    "Switching mode replaces these settings you changed yourself:\n\n{fields}"
)


def overwritten_labels(config) -> list[str]:
    """Display labels for the hand-set fields a mode flip would replace."""
    from vrcc.core.config import profile_overrides

    return [
        tr(_PROFILE_FIELD_LABELS[key])
        for key in profile_overrides(config)
        if key in _PROFILE_FIELD_LABELS
    ]


def confirm_mode_overwrite(parent, config) -> bool:
    """Whether a mode flip may proceed, asking only when it would replace a
    value the user set by hand."""
    from PySide6.QtWidgets import QMessageBox

    labels = overwritten_labels(config)
    if not labels:
        return True
    answer = QMessageBox.question(
        parent,
        tr(_MODE_OVERWRITE_TITLE),
        tr(_MODE_OVERWRITE_BODY, fields="\n".join(f"- {name}" for name in labels)),
        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        QMessageBox.StandardButton.No,
    )
    return answer == QMessageBox.StandardButton.Yes


def on_mode_changed(dlg: "SettingsDialog", value: str) -> None:
    """Apply the picked profile, asking first when that would replace a value
    the user set by hand. Declining puts the control back, since a control
    showing a mode that never applied is worse than no control."""
    if not confirm_mode_overwrite(dlg, dlg._cfg):
        dlg._loading = True
        try:
            dlg._mode.set_value("Speed" if value == "Quality" else "Quality")
        finally:
            dlg._loading = False
        return
    dlg._apply_profile("quality" if value == "Quality" else "latency")


def _mode_recommendation(dlg: "SettingsDialog") -> str:
    """Muted Speed/Quality recommendation for the active whisper model on its
    resolved device, or "" when the recommender has no opinion (onnx-asr, or an
    unmeasured model/device)."""
    device = resolved_device(
        dlg._cfg.stt.device, dlg._cfg.stt.device_index, dlg._cfg.stt.model
    )
    # Same machine factor the reset applies the mode with: judging the beam
    # against the reference machine's clock here would advise Quality beside
    # the Speed that button just chose.
    profile = recommend.recommended_profile(
        dlg._cfg.stt.model, device, calibrate.stored_factor(dlg._cfg)
    )
    if profile == "quality":
        return tr(_MODE_RECOMMEND_QUALITY)
    if profile == "latency":
        return tr(_MODE_RECOMMEND_SPEED)
    return ""


def build_mode_control(dlg: "SettingsDialog", form: "QFormLayout") -> None:
    """Add the Mode control (Speed <-> Quality, mapped to apply_profile) with
    its tooltip and the description row under it, and bind
    ``dlg._update_mode_for_model``."""
    dlg._mode = SegmentedControl(
        [("Speed", tr("Speed")), ("Quality", tr("Quality"))],
        "Quality" if dlg._cfg.gui.profile == "quality" else "Speed",
    )
    dlg._mode.setToolTip(tr(_MODE_TOOLTIP))
    dlg._mode.changed.connect(dlg._on_mode_changed)
    form.addRow(tr("Mode"), dlg._mode)

    dlg._mode_desc = QLabel(tr(_MODE_DESC))
    dlg._mode_desc.setWordWrap(True)
    dlg._mode_desc.setStyleSheet(dlg._muted_style)
    form.addRow("", dlg._mode_desc)

    def update_mode_for_model():
        # The onnxruntime-backed models decode greedily, so the profile's beam
        # width cannot tune their captions: grey the control in place. The
        # stored profile is untouched (its VAD parts still apply, and the
        # Advanced knobs stay usable). The visible description must not
        # advertise a trade-off the locked control cannot deliver, so it swaps
        # to the locked explanation and back.
        spec = WHISPER_MODELS.get(dlg._cfg.stt.model)
        locked = spec is not None and spec.runs_on_onnxruntime
        dlg._mode.setEnabled(not locked)
        if locked:
            locked_text = tr(
                _MODE_LOCKED_TOOLTIP, name=whisper_display_name(spec.id)
            )
            dlg._mode.setToolTip(locked_text)
            dlg._mode_desc.setText(locked_text)
            return
        dlg._mode.setToolTip(tr(_MODE_TOOLTIP))
        text = tr(_MODE_DESC)
        recommendation = _mode_recommendation(dlg)
        if recommendation:
            text = text + "\n" + recommendation
        dlg._mode_desc.setText(text)

    dlg._update_mode_for_model = update_mode_for_model
    # Re-evaluate the recommendation when the Mode toggles (device / model
    # triggers fire from their own combos).
    dlg._mode.changed.connect(lambda _v: update_mode_for_model())
    update_mode_for_model()
