"""Heavier Settings pages: VRChat connection and Advanced / power-user tuning.
Each ``build_*_page(dlg)`` returns the tab widget and reuses ``dlg``'s
bind/spin helpers (settings imports this module, never the reverse). The raw
CTranslate2 kwargs tables live in ``settings_kwargs``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)

from vrcc.core.hardware import device_names, resolved_device
from vrcc.gui import model_prompts, settings_kwargs, settings_reset
from vrcc.gui.widgets import no_wheel
from vrcc.i18n import tr, tr_noop

if TYPE_CHECKING:
    from vrcc.gui.settings import SettingsDialog

_AUTO = "auto"

# The stored card is not in this machine's list: a portable install carried to
# a PC with fewer GPUs. Naming it beats silently showing Auto, which claims a
# setting the config does not hold and makes re-picking Auto a no-op.
_MISSING_GPU = tr_noop("GPU {index}: not found on this PC")

# Chatbox separator: the default is a newline, which a line edit renders as
# nothing at all, so the box looks empty and a keystroke silently replaces it.
_SEPARATOR_TIP = tr_noop(
    "Text placed between the original and the translation. "
    "Type \\n for a new line."
)


def _auto_label(dlg: "SettingsDialog") -> QLabel:
    """Muted, hidden-until-Auto label sitting under a device combo."""
    label = QLabel()
    label.setStyleSheet(dlg._muted_style)
    label.setWordWrap(True)
    label.setVisible(False)
    return label


def _device_choices():
    choices = [(tr("Auto"), _AUTO, 0), (tr("CPU"), "cpu", 0)]
    try:
        names = device_names()
    except Exception:  # noqa: BLE001
        names = []
    for i, name in enumerate(names):
        choices.append((tr("GPU {index}: {name}", index=i, name=name), "cuda", i))
    return choices


def _select_stored_device(combo: QComboBox, section) -> None:
    """Point ``combo`` at the stored (device, index). A value this machine
    cannot offer gets an entry of its own rather than leaving the combo on
    Auto: showing Auto would hide what the config actually says, keep the
    "Auto is using your ..." hint hidden, and make picking Auto a no-op
    (currentIndexChanged never fires when the index is already 0)."""
    current = (section.device, section.device_index)
    for i in range(combo.count()):
        if combo.itemData(i) == current:
            combo.setCurrentIndex(i)
            return
    if section.device == "cuda":
        label = tr(_MISSING_GPU, index=section.device_index)
    else:
        # Unreachable from the UI (auto/cpu always match): a hand-edited
        # config, shown verbatim so it is at least recognisable.
        label = section.device
    combo.addItem(label, current)
    combo.setCurrentIndex(combo.count() - 1)


def _make_device_combo(dlg: "SettingsDialog", section) -> QComboBox:
    combo = no_wheel(QComboBox())
    for label, device, index in _device_choices():
        combo.addItem(label, (device, index))
    _select_stored_device(combo, section)

    def on_change(_i):
        if dlg._loading:
            return
        device, index = combo.currentData()
        section.device = device
        section.device_index = index
        if device == "cuda" and section is dlg._cfg.stt:
            model_prompts.maybe_prefer_cpu(dlg, dlg._cfg.stt.model)
        dlg._changed()
    combo.currentIndexChanged.connect(on_change)
    return combo


def _supported_compute_types(device: str, index: int):
    try:
        import ctranslate2

        return sorted(ctranslate2.get_supported_compute_types(device, index))
    except Exception:  # noqa: BLE001
        return []


def _union_compute_values() -> list[str]:
    values = [_AUTO]
    for device, index in (("cpu", 0), ("cuda", 0)):
        for ct in _supported_compute_types(device, index):
            if ct not in values:
                values.append(ct)
    return values


def _compute_values(dlg: "SettingsDialog", section) -> list[str]:
    """Precisions the device THIS section runs on can actually do. Offering
    the union of CPU and CUDA let a processor user pick float16, which
    CTranslate2 cannot build there. Falls back to the union when the probe
    yields nothing (no ctranslate2, or a device it refuses to inspect), so a
    failed probe leaves a usable control rather than Auto alone."""
    model_id = dlg._cfg.stt.model if section is dlg._cfg.stt else None
    device = resolved_device(section.device, section.device_index, model_id)
    values = [_AUTO] + [
        ct
        for ct in _supported_compute_types(device, section.device_index)
        if ct != _AUTO
    ]
    return values if len(values) > 1 else _union_compute_values()


def _make_compute_combo(dlg: "SettingsDialog", section) -> QComboBox:
    combo = no_wheel(QComboBox())
    values = _compute_values(dlg, section)
    combo.addItems(values)
    # A stored value outside the list stays offered at build time: opening
    # Settings must not rewrite a config the user never touched here.
    if section.compute_type not in values:
        combo.addItem(section.compute_type)
    combo.setCurrentText(section.compute_type)
    dlg._bind_text_combo(combo, section, "compute_type")
    return combo


def refresh_compute_combo(dlg: "SettingsDialog", combo: QComboBox, section) -> None:
    """Re-offer the precisions the newly selected device supports. A stored
    value the new device cannot run falls back to Auto: that write follows a
    device change the user just made, and leaving float16 pinned on the
    processor would only fail later, at engine build."""
    values = _compute_values(dlg, section)
    if [combo.itemText(i) for i in range(combo.count())] == values:
        return
    was_loading = dlg._loading
    dlg._loading = True
    try:
        combo.clear()
        combo.addItems(values)
        if section.compute_type not in values:
            section.compute_type = _AUTO
        combo.setCurrentText(section.compute_type)
    finally:
        dlg._loading = was_loading
    dlg._changed()


def escape_separator(raw: str) -> str:
    """Config value as the Separator box shows it. A line edit paints a
    newline as nothing, so the default separator reads as an empty box and the
    first keystroke replaces it without the user knowing there was anything
    there. Only the newline is escaped: a chatbox separator has no use for a
    literal backslash, and one escape is one thing to explain."""
    return raw.replace("\n", "\\n")


def unescape_separator(text: str) -> str:
    """Inverse of :func:`escape_separator`."""
    return text.replace("\\n", "\n")


def build_vrchat_page(dlg: "SettingsDialog") -> QWidget:
    page = QWidget()
    outer = QVBoxLayout(page)
    outer.setContentsMargins(24, 16, 24, 16)

    # Connection.
    conn = QGroupBox(tr("Connection"))
    conn_form = QFormLayout(conn)
    conn_note = QLabel(tr("Where captions are sent. Most people never change this."))
    conn_note.setStyleSheet(dlg._muted_style)
    conn_note.setWordWrap(True)
    conn_form.addRow(conn_note)

    ip = QLineEdit(dlg._cfg.osc.ip)
    ip.setToolTip(tr("Where captions are sent. Most people never change this."))
    dlg._bind_line(ip, dlg._cfg.osc, "ip")
    conn_form.addRow(tr("Address"), ip)

    port = dlg._spin(0, 65535, dlg._cfg.osc.port)
    port.setToolTip(tr("Where captions are sent. Most people never change this."))
    dlg._bind_int(port, dlg._cfg.osc, "port")
    conn_form.addRow(tr("Port"), port)
    outer.addWidget(conn)

    # Message pacing.
    pace = QGroupBox(tr("Message pacing"))
    pace_form = QFormLayout(pace)
    interval = dlg._dspin(0.1, 10.0, dlg._cfg.osc.min_interval_s, 2, 0.1)
    interval.setToolTip(tr("How quickly messages are sent to the chatbox."))
    dlg._bind_float(interval, dlg._cfg.osc, "min_interval_s")
    pace_form.addRow(tr("Minimum time between messages (s)"), interval)

    burst = dlg._spin(1, 20, dlg._cfg.osc.burst)
    burst.setToolTip(tr("How many messages can be sent quickly in a row."))
    dlg._bind_int(burst, dlg._cfg.osc, "burst")
    pace_form.addRow(tr("Burst"), burst)

    split_delay = dlg._dspin(0.5, 10.0, dlg._cfg.osc.split_delay_s, 1, 0.5)
    split_delay.setToolTip(
        tr(
            "How long each part of a long caption stays visible before the "
            "next part replaces it."
        )
    )
    dlg._bind_float(split_delay, dlg._cfg.osc, "split_delay_s")
    pace_form.addRow(tr("Delay between split parts (s)"), split_delay)
    outer.addWidget(pace)

    # Chatbox message format.
    fmt = QGroupBox(tr("Chatbox message"))
    fmt_form = QFormLayout(fmt)
    overflow = no_wheel(QComboBox())
    for label, value in (
        (tr("Send in parts"), "split"),
        (tr("Shorten to fit"), "truncate"),
        (tr("Send full (may be cut off in VRChat)"), "send"),
    ):
        overflow.addItem(label, value)
    oi = overflow.findData(dlg._cfg.osc.overflow)
    if oi >= 0:
        overflow.setCurrentIndex(oi)
    overflow.setToolTip(
        tr("What to do when a caption is too long for one message.")
    )
    dlg._bind_data_combo(overflow, dlg._cfg.osc, "overflow")
    fmt_form.addRow(tr("If a message is too long"), overflow)

    sep = QLineEdit(escape_separator(dlg._cfg.osc.translation_separator))
    sep.setToolTip(tr(_SEPARATOR_TIP))

    def on_separator(text):
        if dlg._loading:
            return
        dlg._cfg.osc.translation_separator = unescape_separator(text)
        dlg._changed()
    sep.textChanged.connect(on_separator)
    dlg._separator_edit = sep
    fmt_form.addRow(tr("Separator"), sep)

    sfx = QCheckBox(tr("Play a sound when the chatbox updates"))
    sfx.setChecked(dlg._cfg.osc.notification_sfx)
    sfx.setToolTip(tr("VRChat's chatbox notification sound."))
    dlg._bind_checkbox(sfx, dlg._cfg.osc, "notification_sfx")
    fmt_form.addRow(sfx)
    outer.addWidget(fmt)

    # When I mute in VRChat.
    mute = QGroupBox(tr("When I mute in VRChat"))
    mute_form = QFormLayout(mute)
    # Captioning starts off every launch and the master toggle outranks every
    # mode, so "Only caption while muted" alone reads as if picking it starts
    # captions; the note names the real precondition.
    mute_note = QLabel(
        tr(
            "These options apply only while captioning is turned on "
            "in the main window."
        )
    )
    mute_note.setStyleSheet(dlg._muted_style)
    mute_note.setWordWrap(True)
    mute_form.addRow(mute_note)
    dlg._mute_note = mute_note

    mute_enabled = QCheckBox(tr("React when I mute myself in VRChat"))
    mute_enabled.setChecked(dlg._cfg.mute_sync.enabled)
    mute_enabled.setToolTip(
        tr("Let muting yourself in VRChat control captioning.")
    )
    dlg._bind_checkbox(mute_enabled, dlg._cfg.mute_sync, "enabled")
    mute_form.addRow(mute_enabled)

    mode_labels = {
        "pause": tr("Pause captions"),
        "ignore": tr("Keep captioning"),
        "invert": tr("Only caption while muted"),
    }
    mode_tips = {
        "pause": tr("Stop captioning while you're muted."),
        "ignore": tr("Ignore mute and keep captioning either way."),
        "invert": tr("Only caption while you're muted."),
    }
    mode_row = QHBoxLayout()
    group = QButtonGroup(dlg)
    dlg._mute_mode_buttons = {}
    for mode in ("pause", "ignore", "invert"):
        rb = QRadioButton(mode_labels[mode])
        rb.setToolTip(mode_tips[mode])
        rb.setChecked(dlg._cfg.mute_sync.mode == mode)
        group.addButton(rb)
        mode_row.addWidget(rb)
        dlg._mute_mode_buttons[mode] = rb

        def make_handler(m):
            def handler(checked):
                if checked and not dlg._loading:
                    dlg._cfg.mute_sync.mode = m
                    dlg._changed()
            return handler
        rb.toggled.connect(make_handler(mode))
    mode_row.addStretch(1)
    mode_holder = QWidget()
    mode_holder.setLayout(mode_row)
    mute_form.addRow(tr("Mode"), mode_holder)
    outer.addWidget(mute)

    outer.addStretch(1)
    return page


def build_advanced_page(dlg: "SettingsDialog") -> QWidget:
    page = QWidget()
    outer = QVBoxLayout(page)
    outer.setContentsMargins(24, 16, 24, 16)

    warning = QLabel(
        tr(
            "These are power-user settings. The defaults work well for most "
            "people; change them only if you know what they do."
        )
    )
    warning.setWordWrap(True)
    warning.setStyleSheet(dlg._warn_style)
    outer.addWidget(warning)

    form = QFormLayout()
    outer.addLayout(form)

    # Run on GPU/CPU + processing precision.
    dlg._stt_device_combo = _make_device_combo(dlg, dlg._cfg.stt)
    dlg._stt_device_combo.setToolTip(
        tr("Use your graphics card (faster) or the processor.")
    )
    form.addRow(tr("Run voice recognition on"), dlg._stt_device_combo)
    dlg._stt_device_auto_label = _auto_label(dlg)
    form.addRow("", dlg._stt_device_auto_label)
    dlg._stt_device_combo.currentIndexChanged.connect(
        lambda _i: settings_reset.refresh_after_stt_device(dlg)
    )

    dlg._stt_compute_combo = _make_compute_combo(dlg, dlg._cfg.stt)
    dlg._stt_compute_combo.setToolTip(
        tr("Lower precision is faster and uses less memory.")
    )
    form.addRow(tr("Voice processing precision"), dlg._stt_compute_combo)
    dlg._stt_device_combo.currentIndexChanged.connect(
        lambda _i: refresh_compute_combo(dlg, dlg._stt_compute_combo, dlg._cfg.stt)
    )

    dlg._mt_device_combo = _make_device_combo(dlg, dlg._cfg.translate)
    dlg._mt_device_combo.setToolTip(
        tr("Use your graphics card (faster) or the processor.")
    )
    form.addRow(tr("Run translation on"), dlg._mt_device_combo)
    dlg._mt_device_auto_label = _auto_label(dlg)
    form.addRow("", dlg._mt_device_auto_label)
    dlg._mt_device_combo.currentIndexChanged.connect(
        lambda _i: settings_reset.update_device_auto_labels(dlg)
    )

    dlg._mt_compute_combo = _make_compute_combo(dlg, dlg._cfg.translate)
    dlg._mt_compute_combo.setToolTip(
        tr("Lower precision is faster and uses less memory.")
    )
    form.addRow(tr("Translation processing precision"), dlg._mt_compute_combo)
    dlg._mt_device_combo.currentIndexChanged.connect(
        lambda _i: refresh_compute_combo(dlg, dlg._mt_compute_combo, dlg._cfg.translate)
    )

    # Threads / workers.
    cpu_threads = dlg._spin(0, 64, dlg._cfg.stt.cpu_threads)
    cpu_threads.setToolTip(
        tr("How many processor cores to use (0 = automatic).")
    )
    dlg._bind_int(cpu_threads, dlg._cfg.stt, "cpu_threads")
    dlg._stt_cpu_threads_spin = cpu_threads
    form.addRow(tr("CPU threads (0 = auto)"), cpu_threads)

    workers = dlg._spin(1, 8, dlg._cfg.stt.num_workers)
    workers.setToolTip(tr("How many voice-recognition jobs run at once."))
    dlg._bind_int(workers, dlg._cfg.stt, "num_workers")
    dlg._stt_workers_spin = workers
    form.addRow(tr("Voice recognition workers"), workers)

    inter = dlg._spin(1, 8, dlg._cfg.translate.inter_threads)
    inter.setToolTip(
        tr("How many processor cores translation may use across jobs.")
    )
    dlg._bind_int(inter, dlg._cfg.translate, "inter_threads")
    dlg._mt_inter_spin = inter
    form.addRow(tr("Translation threads (between jobs)"), inter)

    intra = dlg._spin(0, 64, dlg._cfg.translate.intra_threads)
    intra.setToolTip(
        tr("How many processor cores each translation job may use (0 = auto).")
    )
    dlg._bind_int(intra, dlg._cfg.translate, "intra_threads")
    dlg._mt_intra_spin = intra
    form.addRow(tr("Translation threads (within a job, 0 = auto)"), intra)

    queued = dlg._spin(-1, 64, dlg._cfg.translate.max_queued_batches)
    queued.setToolTip(
        tr(
            "How many translation batches may wait in line "
            "(0 = auto, -1 = unlimited)."
        )
    )
    dlg._bind_int(queued, dlg._cfg.translate, "max_queued_batches")
    dlg._mt_queued_spin = queued
    form.addRow(
        tr("Translation queue size (0 = auto, -1 = unlimited)"), queued
    )

    # Timing.
    for label, field, lo, hi, tip in (
        (tr("Wait before an early caption (ms)"), "speculative_silence_ms",
         0, 5000,
         tr("Pause length that triggers an early, tentative caption.")),
        (tr("Wait before finishing a caption (ms)"), "finalize_silence_ms",
         0, 5000,
         tr("How long a pause has to be to end a sentence.")),
        (tr("Shortest caption (ms)"), "min_utterance_ms", 0, 5000,
         tr("Ignore blips shorter than this.")),
        (tr("Keep audio before you start (ms)"), "pre_roll_ms", 0, 2000,
         tr("Include a moment of audio from just before you start speaking.")),
    ):
        spin = dlg._spin(lo, hi, getattr(dlg._cfg.vad, field))
        spin.setToolTip(tip)
        dlg._bind_int(spin, dlg._cfg.vad, field)
        dlg._vad_spins[field] = spin
        form.addRow(label, spin)

    max_utt = dlg._dspin(1.0, 60.0, dlg._cfg.vad.max_utterance_s, 1, 0.5)
    max_utt.setToolTip(tr("Force a caption to finish after this many seconds."))
    dlg._bind_float(max_utt, dlg._cfg.vad, "max_utterance_s")
    dlg._vad_spins["max_utterance_s"] = max_utt
    form.addRow(tr("Longest caption (s)"), max_utt)

    dlg._update_check = QCheckBox(tr("Tell me when a new version is available"))
    dlg._update_check.setChecked(dlg._cfg.gui.update_check_enabled)
    dlg._update_check.setToolTip(
        tr("Check GitHub for a newer VRCC when the app starts.")
    )
    dlg._bind_checkbox(dlg._update_check, dlg._cfg.gui, "update_check_enabled")
    form.addRow(dlg._update_check)

    # Raw CTranslate2 kwargs tables (power users only).
    kw1 = QLabel(tr("Extra transcribe options (CTranslate2)"))
    outer.addWidget(kw1)
    outer.addWidget(
        settings_kwargs.make_kwargs_editor(dlg, dlg._cfg.stt, "extra_transcribe_kwargs")
    )
    kw2 = QLabel(tr("Extra translate options (CTranslate2)"))
    outer.addWidget(kw2)
    outer.addWidget(
        settings_kwargs.make_kwargs_editor(
            dlg, dlg._cfg.translate, "extra_translate_kwargs"
        )
    )

    outer.addStretch(1)

    settings_reset.update_device_auto_labels(dlg)
    return page
