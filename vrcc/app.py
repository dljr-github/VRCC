"""Application composition root: build the engine stack, load models, run the GUI.

Wires the individually-tested components into one app, split so the wiring is
testable without a display: build_engine_stack, EngineLoader and _Reloader are
Qt-free; :func:`run` is the only Qt-aware entry point.
"""

from __future__ import annotations

import logging
import logging.handlers
import os
from dataclasses import dataclass
from pathlib import Path

from vrcc.audio.devices import list_input_devices
from vrcc.audio.segmenter import Segmenter
from vrcc.audio.source import AudioSource, MicSource
from vrcc.audio.vad import StreamingVad
from vrcc.core import hardware
from vrcc.core.bus import EventBus
from vrcc.core.config import ConfigStore, Paths, default_paths
from vrcc.core.events import AppError
from vrcc.core.pipeline import Pipeline
from vrcc.core.reloading import _FAILED, EngineLoader, _Reloader, _status_after_swap
from vrcc.download.manager import DownloadManager
from vrcc.i18n import tr
from vrcc.osc.chatbox import ChatboxSender
from vrcc.osc.mutesync import MuteSync
from vrcc.osc.vrchat_detect import VrchatDetector
from vrcc.stt.engine import SttEngine
from vrcc.translate.engine import TranslateEngine
from vrcc.translate.registry import MT_MODELS

logger = logging.getLogger("vrcc.app")

_LOG_MAX_BYTES = 5 * 1024 * 1024
_LOG_BACKUP_COUNT = 2

# Sentinel: "argument not supplied" vs an explicit None (mt/mute are
# legitimately None when translation / mute sync is disabled).
_UNSET = object()


@dataclass
class EngineStack:
    """Everything :func:`run` needs to operate the app, built by
    :func:`build_engine_stack`. A plain data holder -- it starts nothing."""

    pipeline: Pipeline
    source: AudioSource
    segmenter: Segmenter
    vad: StreamingVad | None
    stt: SttEngine
    mt: TranslateEngine | None
    chatbox: ChatboxSender
    mute: MuteSync | None


def _resolve_audio_device(device_cfg: str) -> int | None:
    """Turn ``audio.device`` config into a PortAudio device index (or ``None``).

    ``"auto"`` -> ``None`` (system default); otherwise look the configured
    device *name* up via :func:`list_input_devices`, falling back to ``None``
    if absent.
    """
    if device_cfg == "auto":
        return None
    for index, name in list_input_devices():
        if name == device_cfg:
            return index
    logger.warning(
        "configured audio device %r not found; using the system default",
        device_cfg,
    )
    return None


def build_engine_stack(
    config_store: ConfigStore,
    bus: EventBus,
    paths: Paths,
    *,
    stt_engine=None,
    mt_engine=_UNSET,
    chatbox=None,
    mute=_UNSET,
    source=None,
) -> EngineStack:
    """Assemble the full engine stack from config, or from injected fakes.

    Every component is built for real unless overridden. ``mt`` is ``None``
    when ``translate.enabled`` is False; ``mute`` is ``None`` when
    ``mute_sync.enabled`` is False. Imports no Qt and starts no threads/servers.
    """
    cfg = config_store.config

    vad: StreamingVad | None = None
    if source is None:
        source = MicSource(_resolve_audio_device(cfg.audio.device))

    vad = StreamingVad(threshold=cfg.vad.threshold)
    segmenter = Segmenter(cfg.vad, vad.prob)

    if stt_engine is None:
        stt_engine = SttEngine(
            cfg.stt, paths.models_dir / "whisper" / cfg.stt.model, bus
        )

    if mt_engine is _UNSET:
        spec = MT_MODELS.get(cfg.translate.model) if cfg.translate.enabled else None
        if cfg.translate.enabled and spec is None:
            logger.warning(
                "translate.model %r is not a known MT model; disabling "
                "translation for this session",
                cfg.translate.model,
            )
        if spec is not None:
            mt_engine = TranslateEngine(
                spec, paths.models_dir / "mt" / spec.id, cfg.translate, bus
            )
        else:
            mt_engine = None

    if chatbox is None:
        chatbox = ChatboxSender(cfg.osc, bus)

    if mute is _UNSET:
        if cfg.mute_sync.enabled:
            mute = MuteSync(cfg.mute_sync, cfg.osc.ip, bus)
        else:
            mute = None

    pipeline = Pipeline(
        cfg, bus, source, segmenter, stt_engine, mt_engine, chatbox, mute
    )

    return EngineStack(
        pipeline=pipeline,
        source=source,
        segmenter=segmenter,
        vad=vad,
        stt=stt_engine,
        mt=mt_engine,
        chatbox=chatbox,
        mute=mute,
    )


def _setup_logging(logs_dir: Path, verbose: bool) -> None:
    """Configure the root logger: a 5 MB x2 rotating file in ``logs_dir`` plus
    a console handler when ``verbose``."""
    root = logging.getLogger()
    root.setLevel(logging.DEBUG if verbose else logging.INFO)

    fmt = logging.Formatter(
        "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
    )

    try:
        logs_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            logs_dir / "vrcc.log",
            maxBytes=_LOG_MAX_BYTES,
            backupCount=_LOG_BACKUP_COUNT,
            encoding="utf-8",
        )
        file_handler.setFormatter(fmt)
        root.addHandler(file_handler)
    except OSError:
        # Never let a log-file problem stop the app from launching.
        logging.getLogger("vrcc.app").warning(
            "could not open log file in %s; continuing without file logging",
            logs_dir,
            exc_info=True,
        )

    if verbose:
        console = logging.StreamHandler()
        console.setFormatter(fmt)
        root.addHandler(console)


def _start_pipeline_guarded(pipeline: Pipeline, bus: EventBus) -> bool:
    """Start the capture pipeline, turning a failure (usually the mic refusing
    to open) into an ``AppError("MIC_OPEN_FAILED")`` instead of an unhandled
    exception dying inside a Qt slot. Returns whether capture started; Qt-free."""
    try:
        pipeline.start()
        return True
    except Exception as exc:  # noqa: BLE001 -- any capture failure must surface, not crash
        logger.exception("pipeline failed to start (could not open the microphone?)")
        bus.publish(
            AppError(
                "MIC_OPEN_FAILED",
                "Could not open the microphone — check Settings > Audio",
                detail=str(exc),
            )
        )
        return False


def _models_ready(cfg, dm: DownloadManager) -> bool:
    """True if the configured STT model (and MT model, when translation is on)
    are already downloaded -- i.e. the app can start without the wizard."""
    if not dm.is_whisper_downloaded(cfg.stt.model):
        return False
    if cfg.translate.enabled:
        spec = MT_MODELS.get(cfg.translate.model)
        if spec is None or not dm.is_mt_downloaded(spec):
            return False
    return True


def run(portable: bool = False, verbose: bool = False) -> int:
    """Launch the GUI app. Returns the process exit code."""
    paths = default_paths(portable)
    _setup_logging(paths.logs_dir, verbose)
    logger.info("VRCC starting (portable=%s)", portable)

    store = ConfigStore(paths.config_file)
    store.load()
    for warning in store.load_warnings:
        logger.warning("config: %s", warning)

    hardware.setup_cuda_dlls()

    # Qt imports are deliberately lazy so this module (and build_engine_stack /
    # EngineLoader) stay importable and testable without a display.
    from PySide6.QtCore import QObject, Signal
    from PySide6.QtWidgets import QApplication, QDialog, QMessageBox

    app = QApplication.instance() or QApplication([])
    # The UI language must apply before any GUI module builds a widget
    # (translated strings are read at construction).
    from vrcc.i18n.qt import apply_ui_language

    apply_ui_language(app, store.config.gui.ui_language)

    from vrcc.gui.bridge import BusBridge
    from vrcc.gui.firstrun import FirstRunWizard
    from vrcc.gui.main_window import MainWindow
    from vrcc.gui.models_dialog import ModelsDialog
    from vrcc.gui.settings import SettingsDialog

    _apply_theme(app, store.config.gui.theme, store.config.gui.font_scale)
    _apply_font_scale(app, store.config.gui.font_scale)

    bus = EventBus()
    bridge = BusBridge(bus)
    dm = DownloadManager(paths.models_dir, bus)

    if not _models_ready(store.config, dm):
        wizard = FirstRunWizard(store, dm, bridge)
        if wizard.exec() != QDialog.DialogCode.Accepted:
            logger.info("first-run wizard cancelled; exiting")
            bridge.detach()
            store.save_now()
            return 0

    stack = build_engine_stack(store, bus, paths)

    # Flipped once _start_pipeline_guarded succeeds; _status_after_swap reads
    # it so a green swap never claims "Capturing" over a never-started pipeline.
    pipeline_started = [False]

    class _Starter(QObject):
        """Marshals the loader-thread completion onto the GUI thread, then
        starts the I/O services and (on success) the capture pipeline."""

        done = Signal(bool)

        def __init__(self) -> None:
            super().__init__()
            self.done.connect(self._on_done)

        def _on_done(self, success: bool) -> None:
            stack.chatbox.start()
            if stack.mute is not None:
                stack.mute.start()
            if not success:
                # Mark each dead kind so re-picking the SAME configured model
                # in Settings runs a real swap instead of no-opping into the
                # startup-dead engine (the reloader was seeded from config).
                # One engine can fail seconds before the other finishes
                # loading, so guard against a user swap won inside that
                # window: only seed a kind still on its startup id, and never
                # paint a red status over a capture the swap already started.
                for kind in loader.failed_kinds:
                    if reloader._loaded.get(kind) == startup_ids[kind]:
                        reloader._loaded[kind] = _FAILED
                logger.warning(
                    "engines failed to load; capture not started "
                    "(open Models to re-download, then restart)"
                )
                if not pipeline_started[0]:
                    window.set_capture_status(False, tr("an engine failed to load"))
                return
            if not _start_pipeline_guarded(stack.pipeline, bus):
                # The AppError already flashed the status bar; a dead mic kills
                # the core function, so also say it loudly. App stays up: fix
                # the device in Settings > Audio, then restart.
                window.set_capture_status(False, tr("microphone unavailable"))
                QMessageBox.warning(
                    window,
                    tr("Microphone error"),
                    tr("Could not open the microphone — check Settings > "
                       "Audio, then restart VRCC."),
                )
                return
            pipeline_started[0] = True
            window.set_capture_status(True)

    class _GuiMarshal(QObject):
        """Runs an arbitrary callable on the GUI thread: the reloader's daemon
        swap thread emits ``run`` with the finish closure and the queued
        connection hops it across (same marshalling pattern as ``_Starter``)."""

        run = Signal(object)

        def __init__(self) -> None:
            super().__init__()
            self.run.connect(lambda fn: fn())

    gui_marshal = _GuiMarshal()

    def _build_engine(kind: str, target_id):
        """Build a fresh, *unloaded* engine for ``target_id`` (+ the id it
        represents) -- exactly what the swap asked for, never a re-read config
        value. ``(None, None)`` when there's nothing to build (a None mt
        target disables translation)."""
        cfg = store.config
        if target_id is None:
            return None, None
        if kind == "stt":
            return (
                SttEngine(cfg.stt, paths.models_dir / "whisper" / target_id, bus),
                target_id,
            )
        spec = MT_MODELS.get(target_id)
        if spec is None:
            return None, None
        return (
            TranslateEngine(
                spec, paths.models_dir / "mt" / spec.id, cfg.translate, bus
            ),
            spec.id,
        )

    def _load_engine(engine) -> None:
        if engine is not None:
            engine.load()
            engine.warm_up()

    def _target_for(kind: str):
        """The id the reloader should end up on for ``kind`` given current
        config. ``mt`` -> ``None`` unless translation is on, the model is known
        AND downloaded (so a None target cleanly disables translation vs
        loading a missing model)."""
        cfg = store.config
        if kind == "stt":
            return cfg.stt.model
        if not cfg.translate.enabled:
            return None
        spec = MT_MODELS.get(cfg.translate.model)
        if spec is None or not dm.is_mt_downloaded(spec):
            return None
        return spec.id

    # The engines the reloader is seeded with at startup. _Starter._on_done
    # compares against these before marking a kind _FAILED, so a user swap
    # completed mid-load is never clobbered by the late failure report.
    startup_ids = {
        "stt": store.config.stt.model,
        "mt": store.config.translate.model if stack.mt is not None else None,
    }

    reloader = _Reloader(
        pipeline=stack.pipeline,
        build=_build_engine,
        load=_load_engine,
        set_swapping=stack.pipeline.set_swapping,
        set_status=lambda ok, reason="": _status_after_swap(
            ok,
            reason,
            started=pipeline_started,
            start=lambda: _start_pipeline_guarded(stack.pipeline, bus),
            set_status=window.set_capture_status,
        ),
        marshal=gui_marshal.run.emit,
        bus=bus,
        loaded=dict(startup_ids),
    )

    def on_model_change(kind: str) -> None:
        """Settings wrote a live model change (or toggled translate); hot-swap
        it into the running pipeline without a restart."""
        reloader.request(kind, _target_for(kind))

    # Coalesced (mid-swap) requests recompute their target from current config.
    reloader._on_pending = on_model_change

    def open_settings() -> None:
        dlg = SettingsDialog(
            store,
            parent=window,
            download_manager=dm,
            on_model_change=on_model_change,
        )
        dlg.exec()
        dlg.deleteLater()
        # Settings can edit fields the main-window toolbar also shows (source
        # language, profile, translate toggle); re-sync so they don't diverge.
        window.reload_from_config()

    def open_models() -> None:
        dlg = ModelsDialog(dm, bridge, config_store=store, parent=window)
        dlg.exec()
        dlg.deleteLater()

    window = MainWindow(
        bridge,
        store,
        stack.pipeline,
        on_open_settings=open_settings,
        on_open_models=open_models,
        mt_available=stack.mt is not None,
    )
    window.show()

    # Run the driver-floor check before the loader (its flag drives resolve()'s
    # CPU fallback) but after the window subscribes to the bridge, so a
    # DRIVER_TOO_OLD AppError reaches the status bar (delivered synchronously).
    hardware.check_driver_floor(bus)

    starter = _Starter()
    loader = EngineLoader(
        stack.stt, stack.mt, bus, on_complete=starter.done.emit
    )
    loader.start()

    # Passively watch for VRChat's OSCQuery service so the UI can tell the user
    # whether the chatbox is actually reachable (OSC has no delivery ack).
    detector = VrchatDetector(bus)
    detector.start()

    exit_code = 1
    try:
        exit_code = app.exec()
    finally:
        detector.stop()
        stack.pipeline.stop()
        stack.chatbox.stop()
        if stack.mute is not None:
            stack.mute.stop()
        bridge.detach()
        store.save_now()
        logger.info("VRCC stopped")
        if loader.is_alive():
            # Loader still inside a native model load (no cancellation point).
            # Returning would race interpreter finalization with that native
            # code on a daemon thread (Windows access-violation / hung process).
            # Hard-exit instead: state saved and services stopped above.
            logger.warning(
                "engine loader still running at exit; hard-exiting to avoid "
                "racing interpreter finalization with a native model load"
            )
            logging.shutdown()
            os._exit(int(exit_code))

    return int(exit_code)


def _apply_theme(app, theme: str, scale: float = 1.0) -> None:
    """Apply the app theme (Fusion + palette + stylesheet). Never raises.
    ``scale`` bakes the text-size preset into the QSS font-sizes (QSS wins over
    setFont)."""
    try:
        from vrcc.gui.style import apply_theme
        apply_theme(app, theme, scale)
    except Exception:  # noqa: BLE001 -- theming must never block startup
        logger.warning("could not apply theme %r", theme, exc_info=True)


def _apply_font_scale(app, scale: float) -> None:
    """Scale the base font by ``scale`` (clamped). Best-effort: a failure
    leaves the default font."""
    try:
        scale = max(0.5, min(2.0, float(scale)))
        if abs(scale - 1.0) < 1e-3:
            return
        font = app.font()
        base = font.pointSizeF()
        if base > 0:
            font.setPointSizeF(base * scale)
            app.setFont(font)
    except Exception:  # noqa: BLE001 -- font scaling must never block startup
        logger.debug("could not apply font scale", exc_info=True)
