import pytest
from vrcc.gui.style import PALETTE, resolve_theme, build_qss


def test_palette_has_only_dark_with_all_tokens():
    tokens = {"ground", "surface", "surface_2", "border", "text", "muted",
              "accent", "good", "warn", "bad"}
    assert set(PALETTE) == {"dark"}
    assert tokens <= set(PALETTE["dark"])
    for value in PALETTE["dark"].values():
        assert value.startswith("#") and len(value) in (4, 7)


def test_on_badge_token_is_white():
    # Chip text (e.g. the mute badge) sits on a solid good/bad fill, so white
    # reads correctly -- a single shared token.
    assert PALETTE["dark"]["on_badge"] == "#ffffff"


@pytest.mark.parametrize("name", ["dark", "light", "system", "bogus", ""])
def test_resolve_theme_always_dark(name):
    # The argument is kept for stored configs and existing callers, but there
    # is only one palette, so every input resolves to dark.
    assert resolve_theme(name) == "dark"


def test_build_qss_is_nonempty_and_uses_accent():
    qss = build_qss("dark")
    assert isinstance(qss, str) and len(qss) > 200
    assert PALETTE["dark"]["accent"] in qss


def test_build_qss_styles_control_subparts():
    qss = build_qss("dark")
    for token in ("::indicator", "::drop-down", "::down-arrow", "QScrollBar",
                  "QSlider::handle", 'QPushButton[buttonRole="primary"]',
                  "QSpinBox", "::up-button"):
        assert token in qss, f"{token} missing"


def test_ensure_qss_icons_writes_files():
    from vrcc.gui.style import ensure_qss_icons
    d = ensure_qss_icons("dark")
    for name in ("chevron-down.svg", "check.svg", "arrow-up.svg", "arrow-down.svg"):
        assert (d / name).exists()


def test_ensure_qss_icons_dir_named_by_resolved_theme():
    from vrcc.gui.style import ensure_qss_icons
    # Any input resolves to dark; the dir carries the RESOLVED name so every
    # caller shares one icon dir.
    assert ensure_qss_icons("bogus").name == "vrcc-qss-dark"


def test_build_qss_survives_unwritable_icon_dir(tmp_path, monkeypatch):
    # Icon writes failing (read-only temp dir) must degrade to missing icon
    # glyphs (Qt ignores a dead url()), never an aborted stylesheet.
    import vrcc.gui.style as style_mod

    monkeypatch.setattr(style_mod.tempfile, "gettempdir", lambda: str(tmp_path))

    def boom(self, *args, **kwargs):
        raise OSError("read-only temp dir")

    monkeypatch.setattr(style_mod.Path, "write_text", boom)
    qss = style_mod.build_qss("dark")
    assert "::indicator" in qss  # the full sheet still builds


def test_primary_button_focus_rule_follows_primary_rule():
    # The plain primary rule (later, equal specificity) used to swallow
    # :focus; a dedicated primary-focus rule AFTER it keeps a visible ring.
    qss = build_qss("dark")
    base = qss.index('QPushButton[buttonRole="primary"] ')  # raises if absent
    focus = qss.index('QPushButton[buttonRole="primary"]:focus')
    assert focus > base


def test_line_edit_focus_border_unified_to_1px():
    qss = build_qss("dark")
    assert (
        "QComboBox:focus, QLineEdit:focus, QSpinBox:focus, "
        "QDoubleSpinBox:focus { border: 1px solid" in qss
    )
    assert "QLineEdit:focus { border: 2px" not in qss


def test_build_qss_scales_font_size():
    qss = build_qss("dark", 1.2)
    assert "font-size: 17px" in qss  # round(14 * 1.2)
    assert "font-size: 14px" not in qss


def test_build_qss_default_scale_is_14px():
    qss = build_qss("dark")
    assert "font-size: 14px" in qss


def test_build_qss_clamps_scale():
    assert "font-size: 28px" in build_qss("dark", 9.0)  # clamp to 2.0 -> 28
    assert "font-size: 7px" in build_qss("dark", 0.01)  # clamp to 0.5 -> 7


def test_disabled_popup_items_render_in_muted_color():
    # The QWidget color rule (and the flat palette) level every color group,
    # Disabled included, so grey_unsupported_languages is only visible through
    # the QComboBox::item:disabled rule. Rendered pixels bind the whole chain.
    # Classification is by color distance only, never glyph position. Where
    # real fonts exist (Linux offscreen uses fontconfig), antialiased text
    # edges blend toward the background and pass through the muted band, so
    # an enabled row legitimately shows some muted-band pixels. The palette
    # keeps text and muted far apart (~122 RGB distance), which means muted
    # ink never lands in the text band and vice versa at full strength; the
    # assertions therefore only inspect full-strength ink per row.
    import math
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication, QComboBox

    from vrcc.core.languages import LANGUAGES
    from vrcc.gui.model_prompts import grey_unsupported_languages
    from vrcc.gui.style import apply_theme

    def rgb(token):
        value = PALETTE["dark"][token]
        return tuple(int(value[i:i + 2], 16) for i in (1, 3, 5))

    text_rgb, muted_rgb = rgb("text"), rgb("muted")

    app = QApplication.instance() or QApplication([])
    apply_theme(app, "dark", 1.0)
    combo = QComboBox()
    try:
        combo.addItem("auto")
        combo.addItems(list(LANGUAGES.keys()))
        combo.show()
        grey_unsupported_languages(combo, "parakeet-tdt-0.6b-v3")
        combo.showPopup()
        app.processEvents()
        view = combo.view()
        img = view.viewport().grab().toImage()
        model = combo.model()
        texts = [combo.itemText(i) for i in range(combo.count())]

        def counts(row_text):
            r = view.visualRect(model.index(texts.index(row_text), 0))
            text_px = muted_px = 0
            for y in range(max(0, r.y()), min(img.height(), r.y() + r.height())):
                for x in range(max(0, r.x()), min(img.width(), r.x() + r.width())):
                    c = img.pixelColor(x, y)
                    px = (c.red(), c.green(), c.blue())
                    if math.dist(px, text_rgb) < 25:
                        text_px += 1
                    elif math.dist(px, muted_rgb) < 25:
                        muted_px += 1
            return text_px, muted_px

        french_text, _ = counts("French")  # parakeet v3 covers fr
        japanese_text, japanese_muted = counts("Japanese")  # not in its set
        # A wrongly greyed French row would leave no pixel near the text
        # color; a wrongly ungreyed Japanese row would put full-strength
        # text ink where only muted ink may appear.
        assert french_text > 0
        assert japanese_muted > 0 and japanese_text == 0
    finally:
        combo.hidePopup()
        combo.close()
        combo.deleteLater()
        app.processEvents()


def test_apply_font_scale_never_compounds_across_live_changes():
    # The live text-size path calls apply_font_scale repeatedly; every call
    # must set base*scale from the captured base (Large then Small must not
    # yield base*1.2*0.9, and Normal must restore the base exactly).
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    import vrcc.gui.style as style_mod

    app = QApplication.instance() or QApplication([])
    style_mod._BASE_POINT_SIZE = None  # this test owns the captured base
    base = app.font().pointSizeF()
    assert base > 0
    try:
        style_mod.apply_font_scale(app, 1.2)
        style_mod.apply_font_scale(app, 0.9)
        assert app.font().pointSizeF() == pytest.approx(base * 0.9)
        style_mod.apply_font_scale(app, 1.0)
        assert app.font().pointSizeF() == pytest.approx(base)
    finally:
        style_mod.apply_font_scale(app, 1.0)  # leave the shared app font unscaled
        style_mod._BASE_POINT_SIZE = None


def test_apply_theme_scale_changes_resolved_font_height():
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication, QLabel

    from vrcc.gui.style import apply_theme

    app = QApplication.instance() or QApplication([])

    try:
        apply_theme(app, "dark", 1.0)
        small = QLabel("Ag")
        small.ensurePolished()
        h1 = small.fontMetrics().height()

        apply_theme(app, "dark", 2.0)
        big = QLabel("Ag")
        big.ensurePolished()
        h2 = big.fontMetrics().height()

        assert h2 > h1  # the QSS font-size actually scales the resolved font
    finally:
        # The QApplication is shared across test files; a leftover 2x QSS
        # breaks any later window-metrics test.
        apply_theme(app, "dark", 1.0)
