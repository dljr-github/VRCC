"""Branding guard: the bundled window icon and its application at startup.

The ICO lives inside the package (vrcc/vrcc.ico) so a source install and the
frozen build resolve it the same way; tools/make_icon.py regenerates it from
assets/icon/*.svg. apply_theme sets it on the QApplication, from which every
window and dialog inherits.
"""

import os
import struct
from pathlib import Path

_ICO = Path(__file__).resolve().parent.parent / "vrcc" / "vrcc.ico"
_SIZES = [16, 24, 32, 48, 64, 128, 256]


def test_ico_ships_inside_the_package():
    assert _ICO.is_file(), "vrcc/vrcc.ico is missing; run tools/make_icon.py"


def test_ico_header_parses_with_all_sizes():
    data = _ICO.read_bytes()
    reserved, kind, count = struct.unpack_from("<HHH", data, 0)
    assert (reserved, kind) == (0, 1), "vrcc/vrcc.ico is not an ICO file"
    assert count == len(_SIZES)
    edges = []
    for i in range(count):
        w, h = struct.unpack_from("<BB", data, 6 + 16 * i)
        assert w == h, f"ICO entry {i} is not square: {w}x{h}"
        edges.append(256 if w == 0 else w)
    assert sorted(edges) == _SIZES


def test_apply_theme_sets_the_application_window_icon():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtGui import QIcon
    from PySide6.QtWidgets import QApplication

    from vrcc.gui.style import apply_theme

    app = QApplication.instance() or QApplication([])
    # Clear first so the assertion binds this apply_theme call, not icon
    # state left behind by an earlier test in the shared QApplication.
    app.setWindowIcon(QIcon())
    # This file sorts before the window-metrics tests, and the QSS that
    # apply_theme installs (34px control minimums, paddings) inflates their
    # minimumSizeHint measurements; put the sheet back the way it was.
    sheet = app.styleSheet()
    try:
        apply_theme(app, "dark", 1.0)
        assert not app.windowIcon().isNull()
    finally:
        app.setStyleSheet(sheet)
