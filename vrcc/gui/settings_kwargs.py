"""The raw CTranslate2 kwargs tables on the Advanced settings page.

Split out of ``settings_advanced`` to hold that file under the source cap. A
value is stored as parsed JSON when it parses and as the raw string otherwise,
so ``4`` reaches CTranslate2 as an int while ``float16`` stays text.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from PySide6.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QHeaderView,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from vrcc.i18n import tr

if TYPE_CHECKING:
    from vrcc.gui.settings import SettingsDialog


def make_kwargs_editor(dlg: "SettingsDialog", section, field: str) -> QWidget:
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
