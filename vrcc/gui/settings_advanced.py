"""Heavier Settings pages: VRChat connection and Advanced / power-user tuning,
plus the raw CTranslate2 kwargs editor. Each ``build_*_page(dlg)`` returns the
tab widget and reuses ``dlg``'s bind/spin helpers (settings imports this module,
never the reverse).
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from PySide6.QtWidgets import (
    QAbstractItemView,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QRadioButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from vrcc.core.hardware import device_names
from vrcc.gui import model_prompts, settings_reset
from vrcc.gui.widgets import no_wheel
from vrcc.i18n import tr

if TYPE_CHECKING:
    from vrcc.gui.settings import SettingsDialog

_AUTO = "auto"


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


def _make_device_combo(dlg: "SettingsDialog", section) -> QComboBox:
    combo = no_wheel(QComboBox())
    for label, device, index in _device_choices():
        combo.addItem(label, (device, index))
    current = (section.device, section.device_index)
    for i in range(combo.count()):
        if combo.itemData(i) == current:
            combo.setCurrentIndex(i)
            break

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


def _make_compute_combo(dlg: "SettingsDialog", section) -> QComboBox:
    values = [_AUTO]
    seen = {_AUTO}
    for device, index in (("cpu", 0), ("cuda", 0)):
        for ct in _supported_compute_types(device, index):
            if ct not in seen:
                seen.add(ct)
                values.append(ct)
    combo = no_wheel(QComboBox())
    combo.addItems(values)
    if section.compute_type not in values:
        combo.addItem(section.compute_type)
    combo.setCurrentText(section.compute_type)
    dlg._bind_text_combo(combo, section, "compute_type")
    return combo


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

    sep = QLineEdit(dlg._cfg.osc.translation_separator)
    sep.setToolTip(tr("Text placed between the original and the translation."))
    dlg._bind_line(sep, dlg._cfg.osc, "translation_separator")
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

    inject = QCheckBox(tr("Send each sentence as soon as you finish it"))
    inject.setChecked(dlg._cfg.vad.sentence_inject)
    inject.setToolTip(
        tr("Show a sentence in the chatbox at a natural pause instead of "
           "waiting for you to stop talking.")
    )
    dlg._bind_checkbox(inject, dlg._cfg.vad, "sentence_inject")
    dlg._sentence_inject_check = inject
    form.addRow(inject)

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
        _make_kwargs_editor(dlg, dlg._cfg.stt, "extra_transcribe_kwargs")
    )
    kw2 = QLabel(tr("Extra translate options (CTranslate2)"))
    outer.addWidget(kw2)
    outer.addWidget(
        _make_kwargs_editor(dlg, dlg._cfg.translate, "extra_translate_kwargs")
    )

    outer.addStretch(1)

    settings_reset.update_device_auto_labels(dlg)
    return page


# -- kwargs editor ---------------------------------------------------------


def _make_kwargs_editor(dlg: "SettingsDialog", section, field: str) -> QWidget:
    holder = QWidget()
    layout = QVBoxLayout(holder)
    layout.setContentsMargins(0, 0, 0, 0)

    table = QTableWidget(0, 2)
    table.setHorizontalHeaderLabels([tr("Key"), tr("Value (JSON)")])
    table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
    table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
    table.setMaximumHeight(120)

    current = dict(getattr(section, field))
    for key, value in current.items():
        _append_kwargs_row(table, key, _dump_scalar(value))

    def rebuild(*_):
        if dlg._loading:
            return
        new: dict = {}
        for r in range(table.rowCount()):
            key_item = table.item(r, 0)
            key = key_item.text().strip() if key_item else ""
            if not key:
                continue
            val_item = table.item(r, 1)
            raw = val_item.text() if val_item else ""
            new[key] = _parse_scalar(raw)
        setattr(section, field, new)
        dlg._changed()
    table.itemChanged.connect(rebuild)

    row = QHBoxLayout()
    add = QPushButton(tr("Add"))
    add.clicked.connect(lambda: (_append_kwargs_row(table, "", ""), rebuild()))
    remove = QPushButton(tr("Remove selected"))

    def do_remove():
        r = table.currentRow()
        if r >= 0:
            table.removeRow(r)
            rebuild()
    remove.clicked.connect(do_remove)
    row.addWidget(add)
    row.addWidget(remove)
    row.addStretch(1)

    layout.addWidget(table)
    layout.addLayout(row)
    return holder


def _append_kwargs_row(table: QTableWidget, key: str, value: str) -> None:
    r = table.rowCount()
    table.insertRow(r)
    table.setItem(r, 0, QTableWidgetItem(key))
    table.setItem(r, 1, QTableWidgetItem(value))


def _dump_scalar(value) -> str:
    try:
        return json.dumps(value)
    except (TypeError, ValueError):
        return str(value)


def _parse_scalar(raw: str):
    raw = raw.strip()
    if raw == "":
        return ""
    try:
        return json.loads(raw)
    except (ValueError, TypeError):
        return raw
