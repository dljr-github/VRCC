import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication, QAbstractButton, QLabel, QMessageBox

from vrcc.core.bus import EventBus
from vrcc.core.config import ConfigStore, default_paths
from vrcc.stt.registry import WHISPER_MODELS
from vrcc.translate.registry import MT_MODELS


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


class _FakeDM:
    def __init__(self, models_dir=None):
        self.downloaded = set()
        self.models_dir = models_dir
    def is_whisper_downloaded(self, mid): return mid in self.downloaded
    def is_mt_downloaded(self, spec): return spec.id in self.downloaded
    def ensure_whisper(self, mid): self.downloaded.add(mid)
    def ensure_mt(self, spec): self.downloaded.add(spec.id)
    def delete(self, kind, mid): self.downloaded.discard(mid)


def _dlg(tmp_path, dm=None, theme=None):
    from vrcc.gui.bridge import BusBridge
    from vrcc.gui.models_dialog import ModelsDialog
    store = ConfigStore(default_paths(portable=True, app_dir=tmp_path).config_file)
    if theme is not None:
        store.config.gui.theme = theme
    bridge = BusBridge(EventBus())
    dm = dm or _FakeDM(tmp_path / "models")
    return ModelsDialog(dm, bridge, config_store=store), store, bridge, dm


_FORBIDDEN = ["stt", " mt", "whisper", "int8", "-ct2", "beam", "compute", "vad", "ctranslate2"]


def test_no_jargon_in_models_dialog(qapp, tmp_path):
    dlg, _s, bridge, _dm = _dlg(tmp_path)
    try:
        blob = " ".join(
            w.text().lower() for kind in (QAbstractButton, QLabel)
            for w in dlg.findChildren(kind)
        )
        for term in _FORBIDDEN:
            assert term not in blob, f"jargon {term!r} in Models dialog"
    finally:
        dlg.close(); dlg.deleteLater(); bridge.detach()


def test_two_sections_cover_all_models(qapp, tmp_path):
    dlg, _s, bridge, _dm = _dlg(tmp_path)
    try:
        assert len(dlg._rows) == len(WHISPER_MODELS) + len(MT_MODELS)
        assert any(r.kind == "whisper" for r in dlg._rows)
        assert any(r.kind == "mt" for r in dlg._rows)
    finally:
        dlg.close(); dlg.deleteLater(); bridge.detach()


def test_section_icons_fall_back_to_asterisk_when_svg_invalid(qapp, tmp_path, monkeypatch):
    # Both section headers (Voice model, Translation model) must not go blank
    # if the icon SVG fails to render -- a short ASCII fallback beats silence.
    import vrcc.gui.widgets as widgets_mod

    monkeypatch.setattr(widgets_mod, "svg_pixmap", lambda svg, size: None)
    dlg, _s, bridge, _dm = _dlg(tmp_path)
    try:
        labels = [w.text() for w in dlg.findChildren(QLabel)]
        assert labels.count("*") >= 2
    finally:
        dlg.close(); dlg.deleteLater(); bridge.detach()


def test_translation_row_shows_full_model_name(qapp, tmp_path):
    """Full model names are shown (user decision): no brand-stripping."""
    dlg, _s, bridge, _dm = _dlg(tmp_path)
    try:
        mt_rows = [r for r in dlg._rows if r.kind == "mt"]
        assert any("NLLB" in r.display_name for r in mt_rows)
    finally:
        dlg.close(); dlg.deleteLater(); bridge.detach()


def test_title_font_size_scales_with_font_scale(qapp, tmp_path):
    # Inline px titles must follow the text-size preset (at Large the 16px
    # title used to render smaller than the 17px scaled body text).
    from vrcc.gui.bridge import BusBridge
    from vrcc.gui.models_dialog import ModelsDialog
    store = ConfigStore(default_paths(portable=True, app_dir=tmp_path).config_file)
    store.config.gui.font_scale = 1.2
    bridge = BusBridge(EventBus())
    dlg = ModelsDialog(_FakeDM(tmp_path / "models"), bridge, config_store=store)
    try:
        assert f"font-size: {round(16 * 1.2)}px" in dlg._title.styleSheet()
    finally:
        dlg.close(); dlg.deleteLater(); bridge.detach()


def test_no_use_button_anywhere(qapp, tmp_path):
    from PySide6.QtWidgets import QAbstractButton
    dlg, _s, bridge, _dm = _dlg(tmp_path)
    try:
        labels = [b.text().lower() for b in dlg.findChildren(QAbstractButton)]
        assert not any("use this model" in t for t in labels)
    finally:
        dlg.close(); dlg.deleteLater(); bridge.detach()


def test_active_model_row_shows_in_use(qapp, tmp_path):
    dm = _FakeDM()
    a_whisper = next(iter(WHISPER_MODELS))
    dm.downloaded.add(a_whisper)
    dlg, store, bridge, _dm = _dlg(tmp_path, dm)
    store.config.stt.model = a_whisper
    try:
        row = next(r for r in dlg._rows if r.kind == "whisper" and r.model_id == a_whisper)
        row.render(downloaded=True, active=True, downloading=False)
        # The dialog is never shown in this test, so isVisible() (which
        # requires a mapped top-level window) would always be False; check
        # visibility relative to the dialog instead.
        assert row._inuse_pill.isVisibleTo(dlg)
    finally:
        dlg.close(); dlg.deleteLater(); bridge.detach()


def test_light_theme_has_no_hardcoded_dark_colors(qapp, tmp_path):
    from vrcc.gui.style import PALETTE
    dlg, _s, bridge, _dm = _dlg(tmp_path, theme="light")
    try:
        dark_muted = PALETTE["dark"]["muted"].lower()
        light_muted = PALETTE["light"]["muted"].lower()
        sheets = [dlg.styleSheet().lower()]
        sheets += [lbl.styleSheet().lower() for lbl in dlg.findChildren(QLabel)]
        assert not any(dark_muted in s for s in sheets)
        assert any(light_muted in s for s in sheets)
    finally:
        dlg.close(); dlg.deleteLater(); bridge.detach()


def test_light_theme_cards_use_light_surface(qapp, tmp_path):
    from vrcc.gui.style import PALETTE
    from vrcc.gui.widgets import Card
    dlg, _s, bridge, _dm = _dlg(tmp_path, theme="light")
    try:
        cards = dlg.findChildren(Card)
        assert cards
        for card in cards:
            sheet = card.styleSheet().lower()
            assert PALETTE["dark"]["surface"].lower() not in sheet
            assert PALETTE["light"]["surface"].lower() in sheet
    finally:
        dlg.close(); dlg.deleteLater(); bridge.detach()


def test_models_dialog_has_bold_title_above_lead(qapp, tmp_path):
    """Design review: the dialog opened with body copy and no title."""
    dlg, _s, bridge, _dm = _dlg(tmp_path)
    try:
        titles = [
            lbl for lbl in dlg.findChildren(QLabel) if lbl.text() == "Models"
        ]
        assert titles
        assert "700" in titles[0].styleSheet()  # bold
        assert "16px" in titles[0].styleSheet()
    finally:
        dlg.close(); dlg.deleteLater(); bridge.detach()


def test_recommended_badge_follows_tier_not_active_model(qapp, tmp_path, monkeypatch):
    """Design review: the badge used to mark the ACTIVE model while first-run
    recommends a (possibly different) model for the hardware. The badge must
    follow the tier preset instead -- the "In use" pill alone marks active."""
    from vrcc.core import recommend

    monkeypatch.setattr(recommend, "detect_tier", lambda: "cpu")
    dlg, store, bridge, _dm = _dlg(tmp_path)
    try:
        cpu_whisper, _cpu_mt = recommend.PRESETS["cpu"]
        assert cpu_whisper != "large-v3"  # sanity: distinct from the active pick below
        store.config.stt.model = "large-v3"  # active, but NOT the cpu-tier preset
        dlg._render_all()
        rec_row = next(r for r in dlg._rows if r.kind == "whisper" and r.model_id == cpu_whisper)
        active_row = next(r for r in dlg._rows if r.kind == "whisper" and r.model_id == "large-v3")
        assert rec_row._badge.isVisibleTo(dlg)
        assert rec_row._badge.text() == "Recommended for your PC"
        assert not active_row._badge.isVisibleTo(dlg)
    finally:
        dlg.close(); dlg.deleteLater(); bridge.detach()


def test_recommended_badge_follows_config_device_over_detected_tier(qapp, tmp_path, monkeypatch):
    """A forced-CPU config (the wizard's CPU choice) must badge the cpu preset
    even when the detected hardware tier says gpu_high."""
    from vrcc.core import recommend
    from vrcc.gui.bridge import BusBridge
    from vrcc.gui.models_dialog import ModelsDialog

    monkeypatch.setattr(recommend, "detect_tier", lambda: "gpu_high")
    store = ConfigStore(default_paths(portable=True, app_dir=tmp_path).config_file)
    store.config.stt.device = "cpu"
    bridge = BusBridge(EventBus())
    dlg = ModelsDialog(_FakeDM(tmp_path / "models"), bridge, config_store=store)
    try:
        assert dlg._recommended_ids == recommend.PRESETS["cpu"]
        cpu_whisper, _cpu_mt = recommend.PRESETS["cpu"]
        row = next(r for r in dlg._rows if r.kind == "whisper" and r.model_id == cpu_whisper)
        assert row._badge.isVisibleTo(dlg)
    finally:
        dlg.close(); dlg.deleteLater(); bridge.detach()


def test_delete_active_model_warns_captions_stop_until_reassigned(qapp, tmp_path, monkeypatch):
    """Honest copy under live hotswap: deleting the in-use model stops captions
    now (until another model is chosen in Settings), not "on the next start"."""
    dm = _FakeDM(tmp_path / "models")
    dm.downloaded.add("tiny")
    dlg, store, bridge, _dm = _dlg(tmp_path, dm)
    store.config.stt.model = "tiny"
    asked = []
    def fake_question(parent, title, text, *a, **k):
        asked.append(text)
        return QMessageBox.StandardButton.No
    monkeypatch.setattr(QMessageBox, "question", fake_question)
    try:
        row = next(r for r in dlg._rows if r.model_id == "tiny")
        dlg._delete(row)
        assert asked
        assert (
            "This is the model VRCC is currently using. Captions stop until "
            "you choose another in Settings." in asked[0]
        )
        assert "next start" not in asked[0]
    finally:
        dlg.close(); dlg.deleteLater(); bridge.detach()


def test_disk_warning_no_aborts_download(qapp, tmp_path, monkeypatch):
    from vrcc.gui import model_fit

    dlg, _s, bridge, _dm = _dlg(tmp_path)
    try:
        monkeypatch.setattr(model_fit, "disk_warning", lambda *a, **k: "Not enough free disk space.")
        monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.No)
        started = []
        monkeypatch.setattr(dlg, "_start_download", lambda row: started.append(row))
        dlg._download(dlg._rows[0])
        assert started == []
    finally:
        dlg.close(); dlg.deleteLater(); bridge.detach()
