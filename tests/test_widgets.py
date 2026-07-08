import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def test_segmented_control_selects_and_signals(qapp):
    from vrcc.gui.widgets import SegmentedControl

    seg = SegmentedControl(["Speed", "Quality"], "Speed")
    got = []
    seg.changed.connect(got.append)
    assert seg.value() == "Speed"
    seg.set_value("Quality")
    assert seg.value() == "Quality"
    assert got == ["Quality"]
    seg.deleteLater()


def test_segmented_control_marks_selected_segment_active(qapp):
    from vrcc.gui.widgets import SegmentedControl

    seg = SegmentedControl(["Speed", "Quality"], "Speed")
    seg.set_value("Quality")
    assert seg._buttons["Quality"].property("segActive") is True
    assert not seg._buttons["Speed"].property("segActive")
    seg.deleteLater()


def test_mic_meter_active_toggle_does_not_crash(qapp):
    from vrcc.gui.widgets import MicMeter

    m = MicMeter()
    m.set_level(0.03)
    m.set_active(False)
    m.set_active(True)
    m.deleteLater()


def test_card_uses_provided_palette(qapp):
    from vrcc.gui.style import PALETTE
    from vrcc.gui.widgets import Card

    card = Card(colors=PALETTE["light"])
    sheet = card.styleSheet().lower()
    assert PALETTE["light"]["surface"].lower() in sheet
    assert PALETTE["dark"]["surface"].lower() not in sheet
    card.deleteLater()


def test_mic_meter_segments_are_uniform_size(qapp):
    from vrcc.gui.widgets import MicMeter

    m = MicMeter()
    m.resize(121, 18)  # odd width: would reveal any per-segment rounding drift
    rects = m._segment_rects()
    assert len(rects) == m._BARS
    assert len({r.width() for r in rects}) == 1
    assert len({r.height() for r in rects}) == 1
    assert rects[0].height() == m.height()
    m.deleteLater()


def test_mic_meter_uses_provided_palette(qapp):
    from vrcc.gui.style import PALETTE
    from vrcc.gui.widgets import MicMeter

    m = MicMeter(colors=PALETTE["light"])
    assert m._colors["accent"] == PALETTE["light"]["accent"]
    m.set_level(0.03)  # paint path still works with the injected palette
    m.deleteLater()


def test_mic_meter_segments_center_the_rounding_remainder(qapp):
    from vrcc.gui.widgets import MicMeter

    m = MicMeter()
    m.resize(121, 18)  # 121 - 21 (gaps) = 100; 100 // 8 = 12, a 4px remainder
    rects = m._segment_rects()
    left_margin = rects[0].left()
    right_margin = m.width() - rects[-1].right() - 1
    assert left_margin > 0  # not dumped entirely on the right edge
    assert abs(left_margin - right_margin) <= 1  # split (roughly) evenly
    m.deleteLater()


def test_mic_meter_unlit_segments_use_border_token(qapp):
    from vrcc.gui.widgets import MicMeter
    from vrcc.gui.style import PALETTE

    m = MicMeter(colors=PALETTE["dark"])
    # A low-opacity accent/muted fill used to blend to near-invisible against
    # `ground`; unlit segments must read as a clearly visible border-colored
    # track instead.
    color, opacity = m._segment_style(5, filled_count=2)
    assert color == PALETTE["dark"]["border"]
    assert opacity == 1.0
    m.deleteLater()


def test_mic_meter_filled_segments_still_use_accent_token(qapp):
    from vrcc.gui.widgets import MicMeter
    from vrcc.gui.style import PALETTE

    m = MicMeter(colors=PALETTE["dark"])
    color, opacity = m._segment_style(0, filled_count=2)
    assert color == PALETTE["dark"]["accent"]
    assert opacity == 0.9
    m.deleteLater()


def test_icon_button_renders_svg_to_non_null_icon(qapp):
    from vrcc.gui.widgets import IconButton

    svg = (
        "<svg xmlns='http://www.w3.org/2000/svg' width='20' height='20' "
        "viewBox='0 0 24 24' fill='#98a2b3'>"
        "<circle cx='12' cy='12' r='6'/></svg>"
    )
    btn = IconButton(svg, "Settings", fallback_text="⚙")
    # A valid, colored SVG must render to an actual icon (not blank), and must
    # not fall back to text.
    assert not btn.icon().isNull()
    assert btn.text() == ""
    btn.deleteLater()


def test_icon_button_falls_back_to_glyph_when_svg_invalid(qapp):
    from vrcc.gui.widgets import IconButton

    btn = IconButton("", "Settings", fallback_text="Set")
    # Invalid/empty SVG: the button shows the fallback text instead of a blank.
    assert btn.text() == "Set"
    btn.deleteLater()


def test_icon_label_renders_valid_svg_to_pixmap(qapp):
    from vrcc.gui.widgets import icon_label

    svg = (
        "<svg xmlns='http://www.w3.org/2000/svg' width='20' height='20' "
        "viewBox='0 0 24 24' fill='#98a2b3'><circle cx='12' cy='12' r='6'/></svg>"
    )
    lbl = icon_label(svg, 18, fallback_text="->")
    assert lbl.pixmap() is not None and not lbl.pixmap().isNull()
    assert lbl.text() == ""  # no fallback when the SVG rendered
    lbl.deleteLater()


def test_icon_label_falls_back_to_plain_text_when_svg_invalid(qapp):
    from vrcc.gui.style import PALETTE
    from vrcc.gui.widgets import icon_label

    lbl = icon_label("", 18, colors=PALETTE["light"], fallback_text="->")
    assert lbl.text() == "->"
    assert PALETTE["light"]["muted"].lower() in lbl.styleSheet().lower()
    lbl.deleteLater()
