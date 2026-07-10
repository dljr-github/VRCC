"""Render the four README/walkthrough screenshots into assets/images/.

Run from the repo root:
    .venv/Scripts/python tools/walkthrough_shots.py [out_dir]   (default assets/images)

Builds the app's real widgets offscreen, feeds them a staged caption
session, and grabs each surface as a PNG. Everything that shapes the pixels
is pinned in this file (hardware verdict, config values, caption text,
clock, timestamps, device pixel ratio), so the output does not depend on
the machine's GPU or display scale. Dev tool only -- not packaged, no test
coverage.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

# Real Windows platform: the offscreen QPA renders every glyph as tofu (no
# system fonts), which is useless for a screenshot. grab() renders hidden
# widgets faithfully, so no window ever appears on screen -- never call show().
os.environ.pop("QT_QPA_PLATFORM", None)

# The README shows these images scaled down, so render at 1.25x for crisp
# text. Disabling the per-screen factor and pinning our own keeps the output
# size independent of the machine's Windows display scale.
os.environ["QT_ENABLE_HIGHDPI_SCALING"] = "0"
os.environ["QT_SCALE_FACTOR"] = "1.25"

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "assets/images")
OUT.mkdir(parents=True, exist_ok=True)

from PySide6.QtWidgets import QApplication  # noqa: E402

from vrcc.core import hardware, recommend  # noqa: E402
from vrcc.core.bus import EventBus  # noqa: E402
from vrcc.core.config import ConfigStore, default_paths  # noqa: E402
from vrcc.core.events import (  # noqa: E402
    ChatboxSent,
    MicLevel,
    MuteChanged,
    PhraseRecognized,
    PhraseTranslated,
    VrchatDetected,
)
from vrcc.gui.bridge import BusBridge  # noqa: E402
from vrcc.gui.caption_log import CaptionModel  # noqa: E402
from vrcc.gui.style import apply_theme  # noqa: E402

# The wizard screenshot says "Detected: fast graphics card", so the verdict
# is pinned to the high GPU tier rather than probed. recommend imported the
# two probes by name at module load, so its copies are patched too.
hardware.cuda_device_count = lambda: 1
hardware.total_vram_bytes = lambda index=0: 24 * 1024**3
recommend.cuda_device_count = hardware.cuda_device_count
recommend.total_vram_bytes = hardware.total_vram_bytes

app = QApplication.instance() or QApplication([])

tmp = Path(tempfile.mkdtemp(prefix="vrcc-shots-"))
store = ConfigStore(default_paths(portable=True, app_dir=tmp).config_file)
apply_theme(app, "dark")
store.config.gui.theme = "dark"
store.config.stt.model = "parakeet-tdt-0.6b-v3"
store.config.stt.source_language = "English"
store.config.translate.model = "nllb-600M-int8"
store.config.translate.targets = ["Japanese", "Spanish"]

# The staged conversation: original, translations, and the recognized->sent
# gap the log renders as the per-caption latency.
SESSION = [
    (
        "Hello, how are you today?",
        (("Japanese", "こんにちは、今日は元気ですか？"), ("Spanish", "Hola, ¿cómo estás hoy?")),
        1.2,
    ),
    (
        "The weather is great tonight.",
        (("Japanese", "今夜は天気がいいですね。"), ("Spanish", "El clima está muy agradable esta noche.")),
        0.9,
    ),
    (
        "See you at the meetup later.",
        (("Japanese", "また後でミートアップで会いましょう。"), ("Spanish", "Nos vemos luego en el meetup.")),
        1.4,
    ),
]
TIME_LABELS = ["21:04", "21:05", "21:05"]


class FakePipeline:
    captioning_enabled = True

    def submit_typed(self, t):
        return True

    def set_captioning(self, e):
        self.captioning_enabled = e


class FakeDM:
    """One spare voice model, the active pair, nothing else: the Models list
    shows Downloaded, In use and Download rows side by side."""

    downloaded = {"small", "parakeet-tdt-0.6b-v3", "nllb-600M-int8"}

    def is_whisper_downloaded(self, mid):
        return mid in self.downloaded

    def is_mt_downloaded(self, spec):
        return spec.id in self.downloaded

    @property
    def models_dir(self):
        return tmp


def shot(widget, name, size):
    # Never show(): grab() renders the hidden widget faithfully and no window
    # flashes on the user's screen.
    widget.ensurePolished()
    widget.resize(*size)
    if widget.layout() is not None:
        widget.layout().activate()
    app.processEvents()
    pm = widget.grab()
    path = OUT / f"{name}.png"
    pm.save(str(path))
    print(f"wrote {path} ({pm.width()}x{pm.height()})")


def render_main():
    from vrcc.gui.main_window import MainWindow

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
    # A scripted clock and fixed wall-clock labels replace the window's own
    # caption model, so the latencies and timestamps are the same every run.
    clock = [0.0]
    stamps = iter(TIME_LABELS)
    w._caption_model = CaptionModel(clock=lambda: clock[0], time_label=lambda: next(stamps))
    bus.publish(VrchatDetected(detected=True))
    bus.publish(MuteChanged(muted=False))
    bus.publish(MicLevel(rms=0.12, vad_prob=0.8))
    for uid, (original, translations, latency_s) in enumerate(SESSION, start=1):
        bus.publish(
            PhraseRecognized(
                utterance_id=uid,
                text=original,
                language="en",
                avg_logprob=-0.2,
                no_speech_prob=0.01,
            )
        )
        bus.publish(
            PhraseTranslated(
                utterance_id=uid,
                original=original,
                source_lang="English",
                translations=translations,
            )
        )
        clock[0] += latency_s
        bus.publish(ChatboxSent(text=original, utterance_id=uid))
    app.processEvents()
    shot(w, "main-window", size=(900, 640))
    bridge.detach()
    w.deleteLater()


def render_models():
    from vrcc.gui.models_dialog import ModelsDialog

    bridge = BusBridge(EventBus())
    dlg = ModelsDialog(FakeDM(), bridge, config_store=store)
    shot(dlg, "models", size=(680, 900))
    dlg.deleteLater()
    bridge.detach()


def render_settings():
    from vrcc.gui.settings import SettingsDialog

    dlg = SettingsDialog(store, download_manager=FakeDM(), on_model_change=lambda kind: None)
    dlg._tabs.setCurrentIndex(1)  # Voice recognition
    shot(dlg, "settings-voice", size=(660, 724))
    dlg.deleteLater()


def render_firstrun():
    from vrcc.gui.firstrun import FirstRunWizard

    bridge = BusBridge(EventBus())
    wiz = FirstRunWizard(store, FakeDM(), bridge)
    shot(wiz, "firstrun", size=(564, 620))
    wiz.deleteLater()
    bridge.detach()


render_main()
render_models()
render_settings()
render_firstrun()
print("done")
