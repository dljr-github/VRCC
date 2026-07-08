import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication, QAbstractButton, QLabel

from vrcc.core.bus import EventBus
from vrcc.core.config import ConfigStore, default_paths


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _window(tmp_path, theme=None, mt_available=True, store=None):
    from vrcc.gui.bridge import BusBridge
    from vrcc.gui.main_window import MainWindow

    if store is None:
        store = ConfigStore(default_paths(portable=True, app_dir=tmp_path).config_file)
    if theme is not None:
        store.config.gui.theme = theme
    bridge = BusBridge(EventBus())

    class _P:
        captioning_enabled = False  # matches the real Pipeline's startup default
        def submit_typed(self, t): return True
        def set_captioning(self, e): self.captioning_enabled = e

    w = MainWindow(bridge, store, _P(), on_open_settings=lambda: None,
                   on_open_models=lambda: None, mt_available=mt_available)
    return w, bridge


_FORBIDDEN = ["vad", "stt", "mt", "ctranslate2", "beam", "compute", "nllb",
              "whisper", "int8", "-ct2"]


def test_main_window_has_no_jargon(qapp, tmp_path):
    w, bridge = _window(tmp_path)
    try:
        texts = []
        for kind in (QAbstractButton, QLabel):
            for widget in w.findChildren(kind):
                texts.append(widget.text().lower())
        blob = " ".join(texts)
        for term in _FORBIDDEN:
            assert term not in blob, f"jargon {term!r} leaked onto the main window"
    finally:
        w.close(); w.deleteLater(); bridge.detach()


def test_captioning_button_toggles_pipeline(qapp, tmp_path):
    w, bridge = _window(tmp_path)
    try:
        # Captioning starts off: toggle unchecked, "Start captioning" label.
        assert not w._captioning_btn.isChecked()
        assert "start" in w._captioning_btn.text().lower()

        w._captioning_btn.click()
        assert w._pipeline.captioning_enabled is True
        assert "on" in w._captioning_btn.text().lower()

        w._captioning_btn.click()
        assert w._pipeline.captioning_enabled is False
        assert "start" in w._captioning_btn.text().lower()
    finally:
        w.close(); w.deleteLater(); bridge.detach()


def test_top_bar_row2_captioning_left_gear_overflow_right(qapp, tmp_path):
    # Row 2: captioning toggle left-aligned, then a stretch, then gear +
    # overflow grouped tight on the right -- not all three bunched together.
    w, bridge = _window(tmp_path)
    try:
        w.resize(900, 600)
        w.show()
        qapp.processEvents()
        cap = w._captioning_btn.geometry()
        gear = w._gear_btn.geometry()
        overflow = w._overflow_btn.geometry()
        assert cap.left() < 40
        assert gear.left() - cap.right() > 200  # the stretch's gap
        assert overflow.left() - gear.right() < 20  # tight, no stretch here
    finally:
        w.close(); w.deleteLater(); bridge.detach()


def test_language_flow_labels_present(qapp, tmp_path):
    w, bridge = _window(tmp_path)
    try:
        labels = " ".join(l.text() for l in w.findChildren(QLabel))
        assert "You speak" in labels
        assert "They read" in labels
    finally:
        w.close(); w.deleteLater(); bridge.detach()


def test_light_theme_reaches_caption_log(qapp, tmp_path):
    from vrcc.gui.style import PALETTE
    w, bridge = _window(tmp_path, theme="light")
    try:
        html = w._log.toHtml()
        assert PALETTE["light"]["muted"].lower() in html.lower()
        assert PALETTE["dark"]["text"].lower() not in html.lower()
    finally:
        w.close(); w.deleteLater(); bridge.detach()


def test_main_window_min_width_under_threshold(qapp, tmp_path):
    # A ~900px forced minimum meant the window couldn't share a small VR
    # overlay / secondary-monitor corner. Compressed layout must fit 680px.
    w, bridge = _window(tmp_path)
    try:
        assert w.minimumSizeHint().width() <= 680
    finally:
        w.close(); w.deleteLater(); bridge.detach()


def test_main_window_min_width_with_three_targets_is_bounded(qapp, tmp_path):
    # All 3 target slots visible grows row 1 by ~2 combo+remove groups:
    # measured 869px offscreen (vs 635 with one target). Bound it at 900
    # (measurement + font-metric tolerance) so growth is caught. This exceeds
    # the 680px single-target budget -- wrapping slots onto a second row is
    # future work if the 3-target case must fit small overlays too.
    store = ConfigStore(default_paths(portable=True, app_dir=tmp_path).config_file)
    store.config.translate.targets = ["Japanese", "Korean", "Spanish"]
    w, bridge = _window(tmp_path, store=store)
    try:
        assert all(c is None or c.isChecked() for c in w._target_checks)
        assert w.minimumSizeHint().width() <= 900
    finally:
        w.close(); w.deleteLater(); bridge.detach()


def test_flow_labels_scale_with_font_scale(qapp, tmp_path):
    # The 10px "You speak"/"They read" labels must follow the text-size preset.
    store = ConfigStore(default_paths(portable=True, app_dir=tmp_path).config_file)
    store.config.gui.font_scale = 1.2
    w, bridge = _window(tmp_path, store=store)
    try:
        flow = [l for l in w.findChildren(QLabel) if l.text() == "You speak"]
        assert flow and f"font-size: {round(10 * 1.2)}px" in flow[0].styleSheet()
    finally:
        w.close(); w.deleteLater(); bridge.detach()


def test_mute_chip_hidden_by_default(qapp, tmp_path):
    # No mute-sync state is known yet at construction, so the chip must be
    # hidden rather than showing an empty-looking "-" box.
    w, bridge = _window(tmp_path)
    try:
        assert w._mute_chip.isHidden()
    finally:
        w.close(); w.deleteLater(); bridge.detach()


def test_mute_chip_shown_for_known_states(qapp, tmp_path):
    w, bridge = _window(tmp_path)
    try:
        w._set_mute_chip(True)
        assert not w._mute_chip.isHidden()
        assert w._mute_chip.text() == "MUTED"
        w._set_mute_chip(False)
        assert not w._mute_chip.isHidden()
        assert w._mute_chip.text() == "LIVE"
    finally:
        w.close(); w.deleteLater(); bridge.detach()


def test_mute_chip_text_uses_on_badge_palette_token(qapp, tmp_path):
    # The chip text color must come from the palette (no hex outside style.py).
    w, bridge = _window(tmp_path)
    try:
        w._set_mute_chip(True)
        assert w._p["on_badge"] in w._mute_chip.styleSheet()
        assert "color: white" not in w._mute_chip.styleSheet()
    finally:
        w.close(); w.deleteLater(); bridge.detach()


def test_capture_status_copy_is_honest_and_jargon_free(qapp, tmp_path):
    from PySide6.QtWidgets import QAbstractButton, QLabel

    w, bridge = _window(tmp_path)
    try:
        # Loading state must not already claim to be listening.
        assert "Listening" not in w._capture_label.text()
        # Captioning starts off; turn it on to reach the "Listening" copy.
        w._captioning_btn.setChecked(True)
        w.set_capture_status(True)
        assert "Listening" in w._capture_label.text()
        texts = " ".join(
            wd.text().lower()
            for kind in (QAbstractButton, QLabel)
            for wd in w.findChildren(kind)
        )
        assert "capturing" not in texts
    finally:
        w.close(); w.deleteLater(); bridge.detach()


def test_add_target_tooltip_wording(qapp, tmp_path):
    w, bridge = _window(tmp_path)
    try:
        assert w._add_target_btn.toolTip() == (
            "Add another language your captions are translated into."
        )
    finally:
        w.close(); w.deleteLater(); bridge.detach()


def test_composer_placeholder_and_send_is_primary(qapp, tmp_path):
    w, bridge = _window(tmp_path)
    try:
        assert w._text_input.placeholderText() == "Type to send to your VRChat chatbox…"
        assert w._send_button.property("buttonRole") == "primary"
    finally:
        w.close(); w.deleteLater(); bridge.detach()


def test_overflow_glyph_is_three_dots_no_caret(qapp, tmp_path):
    from vrcc.gui.main_window import _dots_svg

    svg = _dots_svg("#98a2b3")
    assert svg.count("<circle") == 3
    assert "path" not in svg  # no extra caret/arrow shape alongside the dots


def test_translate_gate_reads_live_config_not_ctor_snapshot(qapp, tmp_path):
    # mt_available=False used to permanently suppress "translating..." even
    # once config turned translation on -- engines hot-swap mid-session now,
    # so only the live config value may decide this.
    from types import SimpleNamespace

    w, bridge = _window(tmp_path, mt_available=False)
    try:
        w._store.config.translate.enabled = True
        w._on_phrase_recognized(SimpleNamespace(utterance_id=1, text="hi"))
        assert "translating" in w._log.toPlainText().lower()
    finally:
        w.close(); w.deleteLater(); bridge.detach()


def test_about_credits_github_account(qapp, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QMessageBox

    captured = []
    monkeypatch.setattr(
        QMessageBox, "about", staticmethod(lambda *a: captured.append(a[-1]))
    )
    w, bridge = _window(tmp_path)
    try:
        w._show_about()
        assert captured and "dljr-github" in captured[0]
        assert "github.com/dljr-github" in captured[0]
    finally:
        w.close(); w.deleteLater(); bridge.detach()
