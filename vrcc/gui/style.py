"""App-wide theming: palette tokens + a QSS stylesheet builder.

One place owns the look. main_window / settings / firstrun style themselves
through the classes this QSS targets; the caption-log HTML reads the same
PALETTE so the feed matches the active theme.
"""

import logging
import tempfile
from pathlib import Path

logger = logging.getLogger("vrcc.gui.style")

PALETTE = {
    "dark": {
        "ground": "#14161d", "surface": "#1c1f29", "surface_2": "#232734",
        "border": "#2a2e3a", "text": "#e6e9f0", "muted": "#98a2b3",
        "accent": "#3ea6ff", "accent_hover": "#5cb6ff",
        "good": "#2ecc71", "warn": "#e0a33e", "bad": "#e5544b",
        "on_badge": "#ffffff",
    },
    "light": {
        "ground": "#f5f6f8", "surface": "#ffffff", "surface_2": "#eef1f5",
        "border": "#dfe3ea", "text": "#1a1d24", "muted": "#5b6472",
        "accent": "#1f7ae0", "accent_hover": "#3d8fe8",
        # warn darkened for light surfaces: #9a6a10 on #ffffff is ~4.7:1
        # (the old #e0a33e was ~2:1, unreadable in the restart banner).
        "good": "#2ecc71", "warn": "#9a6a10", "bad": "#e5544b",
        "on_badge": "#ffffff",
    },
}

# Tiny stroke icons QSS `image:` needs as files (data URIs aren't supported).
_ICONS = {
    "chevron-down.svg": "M4 6l4 4 4-4",
    "check.svg": "M3 8l3 3 7-7",
    "arrow-up.svg": "M4 10l4-4 4 4",
    "arrow-down.svg": "M4 6l4 4 4-4",
}


def ensure_qss_icons(theme: str) -> Path:
    """Write the QSS control-kit icon SVGs for `theme` to a temp dir, once.

    Best-effort: an unwritable temp dir degrades to missing icon glyphs (Qt
    ignores a dead url()), never an aborted stylesheet. The dir is named by
    the RESOLVED theme so "system" shares the dark/light dir it maps to."""
    resolved = resolve_theme(theme)
    p = PALETTE[resolved]
    d = Path(tempfile.gettempdir()) / f"vrcc-qss-{resolved}"
    try:
        d.mkdir(parents=True, exist_ok=True)
        for name, path_d in _ICONS.items():
            f = d / name
            if f.exists():
                continue
            stroke = "#ffffff" if name == "check.svg" else p["muted"]
            f.write_text(
                '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16">\n'
                f'<path d="{path_d}" fill="none" stroke="{stroke}" stroke-width="2" '
                'stroke-linecap="round" stroke-linejoin="round"/>\n'
                "</svg>",
                encoding="utf-8",
            )
    except OSError:
        logger.warning(
            "could not write QSS icons to %s; controls render without them", d,
            exc_info=True,
        )
    return d


def resolve_theme(name: str) -> str:
    if name in ("dark", "light"):
        return name
    if name == "system":
        try:
            from PySide6.QtGui import Qt
            from PySide6.QtWidgets import QApplication

            app = QApplication.instance()
            if app is not None:
                scheme = app.styleHints().colorScheme()
                return "light" if scheme == Qt.ColorScheme.Light else "dark"
        except Exception:  # noqa: BLE001 -- any failure falls back to dark
            pass
    return "dark"


def build_qss(theme: str, scale: float = 1.0) -> str:
    resolved = resolve_theme(theme)
    p = PALETTE[resolved]
    icons = ensure_qss_icons(resolved)
    # QSS font rules win over QApplication.setFont, so the text-size preset must
    # bake its scale into every font-size here (14px base) or it does nothing.
    scale = max(0.5, min(2.0, scale))
    fs = round(14 * scale)

    def icon(name: str) -> str:
        return (icons / name).as_posix()

    return f"""
    QWidget {{ background: {p['ground']}; color: {p['text']};
        font-family: "Segoe UI", system-ui, Roboto, Arial, sans-serif; font-size: {fs}px; }}
    QLabel {{ background: transparent; }}
    QPushButton {{ background: {p['surface']}; color: {p['text']};
        border: 1px solid {p['border']}; border-radius: 10px; padding: 8px 14px; }}
    QPushButton:hover {{ border-color: {p['accent']}; }}
    QPushButton:focus {{ outline: none; border: 2px solid {p['accent']}; }}
    QComboBox, QLineEdit, QSpinBox, QDoubleSpinBox {{ background: {p['surface']}; color: {p['text']};
        border: 1px solid {p['border']}; border-radius: 10px; padding: 7px 10px; }}
    QComboBox {{ padding-right: 30px; }}
    QComboBox:hover {{ border-color: {p['accent']}; }}
    QComboBox:focus, QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus {{ border: 1px solid {p['accent']}; }}
    QComboBox, QLineEdit, QSpinBox, QDoubleSpinBox, QPushButton {{ min-height: 34px; }}
    QPushButton[compact="true"] {{ min-height: 0; }}
    QCheckBox {{ background: transparent; spacing: 8px; }}
    QCheckBox::indicator, QRadioButton::indicator {{ width: 16px; height: 16px;
        border: 1px solid {p['border']}; border-radius: 4px; background: {p['surface_2']}; }}
    QRadioButton::indicator {{ border-radius: 8px; }}
    QCheckBox::indicator:hover, QRadioButton::indicator:hover {{ border-color: {p['accent']}; }}
    QCheckBox::indicator:checked, QRadioButton::indicator:checked {{
        background: {p['accent']}; border-color: {p['accent']}; image: url("{icon('check.svg')}"); }}
    QCheckBox::indicator:disabled, QRadioButton::indicator:disabled {{
        background: {p['surface']}; border-color: {p['border']}; }}
    QComboBox::drop-down {{ border: none; background: transparent; width: 26px; }}
    QComboBox::down-arrow {{ image: url("{icon('chevron-down.svg')}"); width: 12px; height: 12px; }}
    QSpinBox::up-button, QDoubleSpinBox::up-button,
    QSpinBox::down-button, QDoubleSpinBox::down-button {{
        subcontrol-origin: border; width: 18px; border: none; background: transparent; }}
    QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {{ image: url("{icon('arrow-up.svg')}"); width: 10px; height: 10px; }}
    QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {{ image: url("{icon('arrow-down.svg')}"); width: 10px; height: 10px; }}
    QScrollBar:vertical {{ background: transparent; width: 10px; margin: 0; }}
    QScrollBar::handle:vertical {{ background: {p['surface_2']}; border-radius: 5px; min-height: 24px; }}
    QScrollBar::handle:vertical:hover {{ border: 1px solid {p['border']}; }}
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
    QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{ background: transparent; }}
    QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 0; }}
    QScrollBar::handle:horizontal {{ background: {p['surface_2']}; border-radius: 5px; min-width: 24px; }}
    QScrollBar::handle:horizontal:hover {{ border: 1px solid {p['border']}; }}
    QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{ width: 0; }}
    QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {{ background: transparent; }}
    QSlider::groove:horizontal {{ height: 6px; border-radius: 3px; background: {p['surface_2']}; }}
    QSlider::sub-page:horizontal {{ background: {p['accent']}; border-radius: 3px; }}
    QSlider::handle:horizontal {{ width: 16px; height: 16px; border-radius: 8px;
        background: {p['accent']}; border: 2px solid {p['ground']}; margin: -6px 0; }}
    QPushButton[buttonRole="primary"] {{ background: {p['accent']}; border: 1px solid {p['accent']}; color: #ffffff; }}
    QPushButton[buttonRole="primary"]:hover {{ background: {p['accent_hover']}; border-color: {p['accent_hover']}; }}
    QPushButton[buttonRole="primary"]:focus {{ background: {p['accent_hover']}; border: 1px solid {p['ground']}; }}
    QPushButton[buttonRole="primary"]:disabled {{ background: {p['surface_2']}; color: {p['muted']}; border-color: {p['border']}; }}
    QPushButton[segActive="true"] {{ background: {p['accent']}; border-color: {p['accent']}; color: #ffffff; }}
    QTabWidget::pane {{ border: 1px solid {p['border']}; border-radius: 10px; }}
    QTabBar::tab {{ background: {p['ground']}; color: {p['muted']};
        padding: 8px 14px; border-radius: 8px; margin: 2px; }}
    QTabBar::tab:selected {{ background: {p['surface']}; color: {p['text']}; }}
    QProgressBar {{ background: {p['surface']}; border: 1px solid {p['border']};
        border-radius: 8px; text-align: center; }}
    QProgressBar::chunk {{ background: {p['accent']}; border-radius: 8px; }}
    QToolTip {{ background: {p['surface_2']}; color: {p['text']};
        border: 1px solid {p['border']}; padding: 6px 8px; }}
    QGroupBox {{ margin-top: 14px; }}
    QGroupBox::title {{ subcontrol-origin: margin; left: 8px; padding: 0 4px; }}
    """.strip()


def apply_theme(app, theme: str, scale: float = 1.0) -> str:
    """Set Fusion + a token palette + the QSS. Returns the resolved theme."""
    from PySide6.QtGui import QColor, QPalette

    resolved = resolve_theme(theme)
    p = PALETTE[resolved]
    try:
        app.setStyle("Fusion")
    except Exception:  # noqa: BLE001
        pass
    pal = QPalette()
    pal.setColor(QPalette.ColorRole.Window, QColor(p["ground"]))
    pal.setColor(QPalette.ColorRole.Base, QColor(p["surface"]))
    pal.setColor(QPalette.ColorRole.AlternateBase, QColor(p["surface_2"]))
    pal.setColor(QPalette.ColorRole.Text, QColor(p["text"]))
    pal.setColor(QPalette.ColorRole.WindowText, QColor(p["text"]))
    pal.setColor(QPalette.ColorRole.ButtonText, QColor(p["text"]))
    pal.setColor(QPalette.ColorRole.Button, QColor(p["surface"]))
    pal.setColor(QPalette.ColorRole.Highlight, QColor(p["accent"]))
    pal.setColor(QPalette.ColorRole.HighlightedText, QColor("#ffffff"))
    pal.setColor(QPalette.ColorRole.ToolTipBase, QColor(p["surface_2"]))
    pal.setColor(QPalette.ColorRole.ToolTipText, QColor(p["text"]))
    app.setPalette(pal)
    app.setStyleSheet(build_qss(resolved, scale))
    return resolved
