"""Application composition root: build the engine stack, load models, run the GUI.

Wires the individually-tested components into one app, split so the wiring is
testable without a display: build_engine_stack, EngineLoader and _Reloader are
Qt-free; :func:`run` is the only Qt-aware entry point.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

from vrcc.audio.segmenter import Segmenter
from vrcc.audio.source import AudioSource, MicSource
from vrcc.audio.vad import StreamingVad
from vrcc.core import hardware
from vrcc.core.bus import EventBus
from vrcc.core.config import ConfigStore, Paths, default_paths
from vrcc.core.events import AppError
from vrcc.core.live_apply import LiveApply
from vrcc.core.logs import setup_logging
from vrcc.core.pipeline import Pipeline
from vrcc.core.reloading import _FAILED, EngineLoader, _Reloader, _status_after_swap
from vrcc.core.startup import (
    default_source_language as _default_source_language,
    models_ready as _models_ready,
    resolve_audio_device as _resolve_audio_device,
)
from vrcc.download.manager import DownloadManager
from vrcc.i18n import tr
from vrcc.osc.chatbox import ChatboxSender
from vrcc.osc.mutesync import MuteSync
from vrcc.osc.vrchat_detect import VrchatDetector
from vrcc.stt import create_stt_engine
from vrcc.stt.engine import SttEngine
from vrcc.translate.engine import TranslateEngine
from vrcc.translate.registry import MT_MODELS

logger = logging.getLogger("vrcc.app")

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
        stt_engine = create_stt_engine(
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
                "Could not open the microphone. Check Settings > Audio",
                detail=str(exc),
            )
        )
        return False


def _swap_main_window(old, make_window, detector):
    """Replace ``old`` with a freshly built MainWindow and carry its runtime
    state across: nothing replays bus events for a late subscriber, so the
    fresh window would otherwise sit on "Starting" and "VRChat: checking"
    until the next transition. The capture label carries verbatim; a red
    failure must stay red whether or not the pipeline ever started, and
    paused-vs-listening re-derives from the captioning toggle, which the
    fresh window reads from the pipeline at construction."""
    old.disconnect_bridge()
    fresh = make_window()
    fresh.restoreGeometry(old.saveGeometry())
    fresh._engine_states.update(old._engine_states)
    fresh._render_log()
    fresh.set_capture_status(old._capture_ok, old._capture_reason)
    detector.republish()
    fresh.show()
    old.hide()
    old.deleteLater()
    return fresh


def run(portable: bool = False, verbose: bool = False) -> int:
    """Launch the GUI app. Returns the process exit code."""
    paths = default_paths(portable)
    setup_logging(paths.logs_dir, verbose)
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
    from vrcc.i18n.qt import apply_ui_language, system_locale_preference

    apply_ui_language(app, store.config.gui.ui_language)

    if store.missing_on_load:
        # First launch only: pre-select the OS display language as the caption
        # language (the wizard shows it). Never fires for an existing config.
        _default_source_language(store.config, system_locale_preference())

    from vrcc.gui.bridge import BusBridge
    from vrcc.gui.firstrun import FirstRunWizard
    from vrcc.gui.main_window import MainWindow
    from vrcc.gui.models_dialog import ModelsDialog
    from vrcc.gui.settings import SettingsDialog
    from vrcc.gui.style import apply_font_scale, apply_theme_guarded

    apply_theme_guarded(app, store.config.gui.theme, store.config.gui.font_scale)
    apply_font_scale(app, store.config.gui.font_scale)

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
            try:
                self._finish_startup(success)
            finally:
                # The engines were the loader's until here; run the swaps
                # Settings asked for during startup now that each engine has a
                # single caller again. After the failure seeding, so a replayed
                # swap in flight is never stamped _FAILED underneath.
                for kind, force in sorted(startup_deferred.items()):
                    reloader.request(kind, _target_for(kind), force=force)
                startup_deferred.clear()

        def _finish_startup(self, success: bool) -> None:
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
                    tr("Could not open the microphone. Check Settings > "
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
            model_dir = paths.models_dir / "whisper" / target_id
            return (
                create_stt_engine(cfg.stt, model_dir, bus, model_id=target_id),
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

    # Swap requests that arrive while the startup EngineLoader still owns the
    # engines are held here (kind -> force) and replayed by _Starter._on_done.
    # The reloader's swap thread would otherwise unload() the very object the
    # loader thread is inside load()/warm_up() on; both engines document a
    # single-caller contract. on_model_change/_on_done both run on the GUI
    # thread, so the check-then-defer below cannot race the replay.
    startup_deferred: dict[str, bool] = {}

    def _request_or_defer(kind: str, force: bool) -> None:
        if loader.is_alive():
            startup_deferred[kind] = startup_deferred.get(kind, False) or force
            return
        reloader.request(kind, _target_for(kind), force=force)

    def on_model_change(kind: str) -> None:
        """Settings wrote a live model change (or toggled translate); hot-swap
        it into the running pipeline without a restart."""
        _request_or_defer(kind, force=False)

    def reload_engine(kind: str) -> None:
        """Rebuild an engine for a device/compute/thread change that keeps the
        same model id: forced so the reloader can't no-op on the unchanged id
        (same hot-swap path as a model switch)."""
        _request_or_defer(kind, force=True)

    # Coalesced (mid-swap) requests recompute their target from current config.
    reloader._on_pending = on_model_change

    def open_settings() -> None:
        lang_before = store.config.gui.ui_language
        dlg = SettingsDialog(
            store,
            parent=window,
            download_manager=dm,
            on_model_change=on_model_change,
            apply=live_apply,
        )
        dlg.exec()
        dlg.deleteLater()
        if store.config.gui.ui_language != lang_before:
            # tr() is read at widget construction, so rebuild in the new language.
            rebuild_main_window()
        else:
            # Settings can edit fields the toolbar also shows (source language,
            # profile, translate toggle); re-sync so they don't diverge.
            window.reload_from_config()

    def open_models() -> None:
        dlg = ModelsDialog(dm, bridge, config_store=store, parent=window)
        dlg.exec()
        dlg.deleteLater()

    def make_window() -> MainWindow:
        return MainWindow(
            bridge,
            store,
            stack.pipeline,
            on_open_settings=open_settings,
            on_open_models=open_models,
            mt_available=stack.mt is not None,
            download_manager=dm,
            on_model_change=on_model_change,
        )

    # Qt-free façade for live Settings changes: the dialog pushes audio/VAD/OSC/
    # engine edits into the running stack with no restart (rebuilds reuse the reloader).
    live_apply = LiveApply(
        pipeline=stack.pipeline,
        segmenter=stack.segmenter,
        chatbox=stack.chatbox,
        bus=bus,
        reload_engine=reload_engine,
        make_source=lambda device_cfg: MicSource(_resolve_audio_device(device_cfg)),
        make_mute=lambda: MuteSync(
            store.config.mute_sync, store.config.osc.ip, bus
        ),
        mute=stack.mute,
    )
    window = make_window()
    window.show()

    def rebuild_main_window() -> None:
        nonlocal window
        apply_ui_language(app, store.config.gui.ui_language)
        window = _swap_main_window(window, make_window, detector)

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
        # Covers both the startup coordinator and one LiveApply built lazily
        # when mute sync was enabled after launch (stack.mute is None then).
        live_apply.stop_mute()
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
