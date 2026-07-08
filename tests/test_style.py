import pytest
from vrcc.gui.style import PALETTE, resolve_theme, build_qss


def test_palette_has_both_themes_and_all_tokens():
    tokens = {"ground", "surface", "surface_2", "border", "text", "muted",
              "accent", "good", "warn", "bad"}
    for theme in ("dark", "light"):
        assert tokens <= set(PALETTE[theme])
        for value in PALETTE[theme].values():
            assert value.startswith("#") and len(value) in (4, 7)


def test_dark_and_light_grounds_differ():
    assert PALETTE["dark"]["ground"] != PALETTE["light"]["ground"]


def test_on_badge_token_is_white_in_both_themes():
    # Chip text (e.g. the mute badge) sits on a solid good/bad fill in both
    # themes, so white reads correctly either way -- a single shared token.
    assert PALETTE["dark"]["on_badge"] == "#ffffff"
    assert PALETTE["light"]["on_badge"] == "#ffffff"


@pytest.mark.parametrize("name,expected", [("dark", "dark"), ("light", "light"), ("bogus", "dark")])
def test_resolve_theme_passthrough_and_fallback(name, expected):
    assert resolve_theme(name) == expected


def test_build_qss_is_nonempty_and_uses_accent():
    qss = build_qss("dark")
    assert isinstance(qss, str) and len(qss) > 200
    assert PALETTE["dark"]["accent"] in qss


def test_build_qss_styles_control_subparts_both_themes():
    from vrcc.gui.style import build_qss
    for theme in ("dark", "light"):
        qss = build_qss(theme)
        for token in ("::indicator", "::drop-down", "::down-arrow", "QScrollBar",
                      "QSlider::handle", 'QPushButton[buttonRole="primary"]',
                      "QSpinBox", "::up-button"):
            assert token in qss, f"{token} missing in {theme}"


def test_ensure_qss_icons_writes_files():
    from vrcc.gui.style import ensure_qss_icons
    d = ensure_qss_icons("dark")
    for name in ("chevron-down.svg", "check.svg", "arrow-up.svg", "arrow-down.svg"):
        assert (d / name).exists()


def test_ensure_qss_icons_dir_named_by_resolved_theme():
    from vrcc.gui.style import ensure_qss_icons
    # "bogus" resolves to dark; the dir must carry the RESOLVED name so
    # "system"/unknown themes share the dark/light dir they map to.
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
    for theme in ("dark", "light"):
        qss = build_qss(theme)
        base = qss.index('QPushButton[buttonRole="primary"] ')  # raises if absent
        focus = qss.index('QPushButton[buttonRole="primary"]:focus')
        assert focus > base, theme


def test_line_edit_focus_border_unified_to_1px():
    qss = build_qss("dark")
    assert (
        "QComboBox:focus, QLineEdit:focus, QSpinBox:focus, "
        "QDoubleSpinBox:focus { border: 1px solid" in qss
    )
    assert "QLineEdit:focus { border: 2px" not in qss


def test_light_warn_token_darkened_for_contrast():
    # Light warn is #9a6a10 (~4.7:1 on #ffffff, WCAG AA); the old shared
    # amber was ~2:1 on white. The dark palette keeps its amber unchanged.
    assert PALETTE["light"]["warn"] != PALETTE["dark"]["warn"]
    assert PALETTE["dark"]["warn"] == "#e0a33e"


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


def test_apply_theme_scale_changes_resolved_font_height():
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication, QLabel

    from vrcc.gui.style import apply_theme

    app = QApplication.instance() or QApplication([])

    apply_theme(app, "dark", 1.0)
    small = QLabel("Ag")
    small.ensurePolished()
    h1 = small.fontMetrics().height()

    apply_theme(app, "dark", 2.0)
    big = QLabel("Ag")
    big.ensurePolished()
    h2 = big.fontMetrics().height()

    assert h2 > h1  # the QSS font-size actually scales the resolved font
