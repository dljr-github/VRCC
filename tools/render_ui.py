"""Render every VRCC GUI surface offscreen to PNGs for a design/visual QA pass.

Run from the repo root:
    .venv/Scripts/python tools/render_ui.py [out_dir]   (default ./ui-renders)

Dev tool only -- not packaged, no test coverage.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

# Real Windows platform: the offscreen QPA renders every glyph as tofu (no
# system fonts), which is useless for a design review. grab() renders hidden
# widgets faithfully, so no window ever appears on screen -- never call show().
os.environ.pop("QT_QPA_PLATFORM", None)

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "ui-renders")
OUT.mkdir(parents=True, exist_ok=True)

from PySide6.QtWidgets import QApplication  # noqa: E402

from vrcc.core.bus import EventBus  # noqa: E402
from vrcc.core.config import ConfigStore, default_paths  # noqa: E402
from vrcc.core.events import (  # noqa: E402
    ChatboxSent,
    MicLevel,
    PhraseRecognized,
    PhraseTranslated,
    VrchatDetected,
)
from vrcc.gui.bridge import BusBridge  # noqa: E402
from vrcc.gui.style import apply_theme  # noqa: E402

app = QApplication.instance() or QApplication([])

tmp = Path(tempfile.mkdtemp(prefix="vrcc-render-"))
store = ConfigStore(default_paths(portable=True, app_dir=tmp).config_file)


def set_theme(theme: str) -> None:
    """Set both the app-wide QSS (P1) and the config field each surface reads
    for its own palette at construction time (P2) -- they must agree."""
    apply_theme(app, theme)
    store.config.gui.theme = theme


class FakePipeline:
    captioning_enabled = False  # matches the real Pipeline's startup default

    def submit_typed(self, t):
        return True

    def set_captioning(self, e):
        self.captioning_enabled = e


class FakeDM:
    """One whisper + one MT model downloaded, so rows/pickers show mixed
    states instead of an empty list."""

    def __init__(self):
        from vrcc.stt.registry import WHISPER_MODELS
        from vrcc.translate.registry import MT_MODELS

        w = list(WHISPER_MODELS)
        m = list(MT_MODELS)
        self.downloaded = {w[0], m[0]}
        # Keep configured/active models aligned with the downloaded ones so
        # Settings' pickers have a valid current selection.
        store.config.stt.model = w[0]
        store.config.translate.model = m[0]

    def is_whisper_downloaded(self, mid):
        return mid in self.downloaded

    def is_mt_downloaded(self, spec):
        return spec.id in self.downloaded

    @property
    def models_dir(self):
        return tmp


class WizardDM(FakeDM):
    def ensure_whisper(self, mid):
        self.downloaded.add(mid)

    def ensure_mt(self, spec):
        self.downloaded.add(spec.id)


def shot(widget, name, size=None):
    # Never show(): grab() renders the hidden widget faithfully and no window
    # flashes on the user's screen.
    widget.ensurePolished()
    if size:
        widget.resize(*size)
    if widget.layout() is not None:
        widget.layout().activate()
    app.processEvents()
    pm = widget.grab()
    path = OUT / f"{name}.png"
    pm.save(str(path))
    print(f"wrote {path} ({pm.width()}x{pm.height()})")


def build_main(theme, populated):
    from vrcc.gui.main_window import MainWindow

    set_theme(theme)
    bus = EventBus()
    bridge = BusBridge(bus)
    w = MainWindow(
        bridge,
        store,
        FakePipeline(),
        on_open_settings=lambda: None,
        on_open_models=lambda: None,
        mt_available=True,
    )
    w.set_capture_status(True)
    if populated:
        bus.publish(VrchatDetected(detected=True))
        bus.publish(MicLevel(rms=0.12, vad_prob=0.8))
        bus.publish(
            PhraseRecognized(
                utterance_id=1,
                text="Hey, how's everyone doing tonight?",
                language="en",
                avg_logprob=-0.2,
                no_speech_prob=0.01,
            )
        )
        bus.publish(
            PhraseTranslated(
                utterance_id=1,
                original="Hey, how's everyone doing tonight?",
                source_lang="English",
                translations=(("Japanese", "みんな、今夜は調子どう？"),),
            )
        )
        bus.publish(ChatboxSent(text="Hey, how's everyone doing tonight?", utterance_id=1))
        bus.publish(
            PhraseRecognized(
                utterance_id=2,
                text="This one is a long caption that will get clipped by the chatbox limit eventually because VRChat only allows short messages",
                language="en",
                avg_logprob=-0.4,
                no_speech_prob=0.02,
            )
        )
        bus.publish(
            ChatboxSent(
                text="This one is a long caption that will get clipped",
                utterance_id=2,
                truncated=True,
            )
        )
        bus.publish(
            PhraseRecognized(
                utterance_id=3,
                text="And one still being translated right now",
                language="en",
                avg_logprob=-0.3,
                no_speech_prob=0.01,
            )
        )
    app.processEvents()
    return w, bridge


def render_main():
    w, bridge = build_main("dark", populated=True)
    shot(w, "main-dark-populated", size=(760, 560))
    bridge.detach()
    w.deleteLater()
    w, bridge = build_main("dark", populated=False)
    shot(w, "main-dark-empty", size=(760, 560))
    bridge.detach()
    w.deleteLater()


def render_models():
    from vrcc.gui.models_dialog import ModelsDialog

    set_theme("dark")
    bridge = BusBridge(EventBus())
    dlg = ModelsDialog(FakeDM(), bridge, config_store=store)
    shot(dlg, "models-dark", size=(700, 640))
    dlg.deleteLater()
    bridge.detach()


def render_settings():
    from vrcc.gui.settings import SettingsDialog

    set_theme("dark")
    dlg = SettingsDialog(store, download_manager=FakeDM(), on_model_change=lambda kind: None)
    shot(dlg, "settings-simple-dark", size=(680, 600))
    dlg._tabs.setCurrentIndex(1)  # Voice recognition
    shot(dlg, "settings-voice-dark", size=(680, 600))
    dlg.deleteLater()


def render_firstrun():
    from vrcc.gui.firstrun import FirstRunWizard

    set_theme("dark")
    bridge = BusBridge(EventBus())
    wiz = FirstRunWizard(store, WizardDM(), bridge)
    shot(wiz, "firstrun-dark", size=(640, 620))
    wiz.deleteLater()
    bridge.detach()


render_main()
render_models()
render_settings()
render_firstrun()
print("done")
