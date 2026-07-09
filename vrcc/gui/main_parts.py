"""Top-bar / status-strip / caption-log / compose-row builders for MainWindow.

Split out of ``main_window.py`` purely to stay under the line-count cap. Each
``build_*(w)`` returns the built widget and writes live control refs back onto
``w`` (the :class:`~vrcc.gui.main_window.MainWindow`), mirroring
``settings_pages.py``'s ``build_*_page(dlg)`` pattern. Imports from
``main_window`` are type-only (main_window imports this, never the reverse).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QPushButton,
    QTextBrowser,
    QWidget,
)

from vrcc.core.languages import LANGUAGES
from vrcc.gui.icons import arrow_svg, dots_svg, gear_svg, mic_svg, x_svg
from vrcc.gui.widgets import Card, IconButton, MicMeter, icon_label, svg_pixmap
from vrcc.i18n import tr

if TYPE_CHECKING:
    from vrcc.gui.main_window import MainWindow

# Mirrors main_window._AUTO / ._NUM_TARGET_SLOTS (small literal, duplicated
# rather than imported back to avoid a main_window <-> main_parts import cycle;
# same pattern as _AUTO in settings.py/settings_pages.py/firstrun.py).
_AUTO = "auto"
_NUM_TARGET_SLOTS = 3


def _compact_combo(combo: QComboBox) -> None:
    # Size to a fixed character count, not the widest item, so long language
    # names (e.g. "Chinese Traditional") don't force the top bar (and the
    # window min-width) wide.
    combo.setMinimumContentsLength(8)
    combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)


def _flow_label(w: "MainWindow", text: str) -> QLabel:
    # Small tracked muted label, case-preserving ("You speak"/"They read").
    # Word-wrap lets it shrink to its longest word instead of forcing the row
    # to its full two-word width. Sized with the window's text-size scale.
    lbl = QLabel(text)
    lbl.setWordWrap(True)
    lbl.setStyleSheet(
        f"color: {w._p['muted']}; font-size: {round(10 * w._scale)}px; "
        "letter-spacing: 1px; background: transparent;"
    )
    return lbl


def build_top_bar(w: "MainWindow") -> QWidget:
    card = Card(colors=w._p)
    # Two rows, not one: every control on a single row forced a ~900px
    # minimum window width. Each row now sizes to only its own content.
    bar = QHBoxLayout()
    bar.setSpacing(8)
    card.body.addLayout(bar)

    # -- language flow: You speak [src] -> They read [tgt] (+ up to 3) ----
    bar.addWidget(_flow_label(w, tr("You speak")))
    w._source_combo = QComboBox()
    w._source_combo.addItems([_AUTO, *LANGUAGES.keys()])
    _compact_combo(w._source_combo)
    w._source_combo.currentTextChanged.connect(w._on_source_changed)
    bar.addWidget(w._source_combo)

    bar.addWidget(
        icon_label(arrow_svg(w._p["muted"]), 16, colors=w._p, fallback_text="->")
    )
    bar.addWidget(_flow_label(w, tr("They read")))

    # Three target slots. Slot 0 is always on; slots 1-2 add/remove via +/x.
    # Each carries a hidden checkbox as the enabled-state source of truth.
    w._target_combos: list[QComboBox] = []
    w._target_checks: list[QCheckBox | None] = []
    w._target_conts: list[QWidget | None] = []
    for slot in range(_NUM_TARGET_SLOTS):
        combo = QComboBox()
        combo.addItems(list(LANGUAGES.keys()))
        _compact_combo(combo)
        combo.currentTextChanged.connect(w._on_targets_changed)
        w._target_combos.append(combo)
        if slot == 0:
            w._target_checks.append(None)
            w._target_conts.append(None)
            bar.addWidget(combo)
        else:
            check = QCheckBox()
            check.hide()  # state carrier only; shown via the container
            check.toggled.connect(w._on_target_enabled_changed)
            w._target_checks.append(check)

            cont = QWidget()
            # Transparent: the app-wide QWidget QSS would otherwise paint this
            # container ground-dark inside the surface-colored bar.
            cont.setStyleSheet("background: transparent;")
            row = QHBoxLayout(cont)
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(6)
            row.addWidget(check)
            row.addWidget(combo)
            remove = IconButton(
                x_svg(w._p["muted"]), tr("Remove this language"), fallback_text="x"
            )
            remove.setFixedSize(34, 34)
            remove.clicked.connect(lambda _=False, s=slot: w._remove_target(s))
            row.addWidget(remove)
            cont.hide()
            w._target_conts.append(cont)
            bar.addWidget(cont)

    w._add_target_btn = QPushButton(tr("+ Language"))
    w._add_target_btn.setToolTip(
        tr("Add another language your captions are translated into.")
    )
    w._add_target_btn.clicked.connect(w._add_target)
    bar.addWidget(w._add_target_btn)
    bar.addStretch(1)

    # -- captioning on/off (left), gear + overflow (right), own row -------
    actions = QHBoxLayout()
    actions.setSpacing(8)
    card.body.addLayout(actions)

    w._captioning_btn = QPushButton(tr("Start captioning"))
    # One static mic icon; the label text alone carries the on/off state.
    mic_pm = svg_pixmap(mic_svg(w._p["muted"]), 20)
    if mic_pm is not None:
        w._captioning_btn.setIcon(QIcon(mic_pm))
    w._captioning_btn.setCheckable(True)
    w._captioning_btn.setToolTip(tr("Pause or resume captioning without closing VRCC."))
    w._captioning_btn.toggled.connect(w._on_captions_toggled)
    actions.addWidget(w._captioning_btn)

    actions.addStretch(1)

    w._gear_btn = IconButton(
        gear_svg(w._p["muted"]), tr("Settings"), fallback_text="Set"
    )
    w._gear_btn.clicked.connect(lambda: w._on_open_settings())
    actions.addWidget(w._gear_btn)

    # QMenu via setMenu() makes Fusion draw its own drop-down arrow over the
    # icon, so the menu-indicator is switched off in QSS below.
    w._overflow_btn = IconButton(
        dots_svg(w._p["muted"]), tr("More"), fallback_text="..."
    )
    w._overflow_btn.setStyleSheet("QPushButton::menu-indicator { width: 0; }")
    menu = QMenu(w)
    menu.addAction(tr("Models…")).triggered.connect(lambda: w._on_open_models())
    menu.addAction(tr("About")).triggered.connect(w._show_about)
    menu.addSeparator()
    menu.addAction(tr("Exit")).triggered.connect(w.close)
    w._overflow_btn.setMenu(menu)
    actions.addWidget(w._overflow_btn)

    return card


def build_status_strip(w: "MainWindow") -> QWidget:
    strip = QWidget()
    row = QHBoxLayout(strip)
    row.setContentsMargins(4, 0, 4, 0)
    row.setSpacing(8)

    # Live mic level (animated while capturing, dimmed when paused).
    w._mic_meter = MicMeter(colors=w._p)
    row.addWidget(w._mic_meter)
    row.addWidget(_flow_label(w, tr("Microphone")))

    row.addStretch(1)

    w._mute_chip = QLabel("–")
    w._mute_chip.setAlignment(Qt.AlignmentFlag.AlignCenter)
    w._set_mute_chip(None)
    row.addWidget(w._mute_chip)

    # Connection chip + honest capture-status, folded off the status bar.
    w._vrchat_label = QLabel("")
    row.addWidget(w._vrchat_label)
    w._capture_label = QLabel("")
    row.addWidget(w._capture_label)

    return strip


def build_caption_log(w: "MainWindow") -> QTextBrowser:
    w._log = QTextBrowser()
    w._log.setOpenExternalLinks(False)
    # Without this, QTextDocument keeps undo data for every insert/trim, so
    # the block cap wouldn't actually bound memory over a long session.
    w._log.document().setUndoRedoEnabled(False)
    # Breathing room so caption text isn't flush against the log edge.
    w._log.document().setDocumentMargin(12)
    return w._log


def build_compose_row(w: "MainWindow") -> QWidget:
    card = Card(colors=w._p)
    row = QHBoxLayout()
    row.setSpacing(8)
    card.body.addLayout(row)

    w._text_input = QLineEdit()
    w._text_input.setPlaceholderText(tr("Type to send to your VRChat chatbox…"))
    w._text_input.returnPressed.connect(w._on_send_clicked)
    row.addWidget(w._text_input, stretch=1)
    w._send_button = QPushButton(tr("Send"))
    w._send_button.setProperty("buttonRole", "primary")
    w._send_button.clicked.connect(w._on_send_clicked)
    row.addWidget(w._send_button)

    return card
