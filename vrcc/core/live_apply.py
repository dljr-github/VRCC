"""Qt-free façade for pushing Settings changes into the running stack.

The GUI (a separate module) calls :class:`LiveApply` to apply audio-device,
VAD, OSC, mute-sync and engine changes without restarting the app. Each method
delegates to an already-tested live-reconfigure entry point; engine (stt/mt)
rebuilds go through the proven :class:`~vrcc.core.reloading._Reloader` hot-swap
so a device/compute/thread change uses the same path as a model switch. Imports
no Qt so the whole layer stays unit-testable without a display.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Callable

from vrcc.core.events import AppError

if TYPE_CHECKING:
    from vrcc.audio.segmenter import Segmenter
    from vrcc.audio.source import AudioSource
    from vrcc.core.bus import EventBus
    from vrcc.core.config import OscConfig, VadConfig
    from vrcc.core.pipeline import Pipeline
    from vrcc.osc.chatbox import ChatboxSender
    from vrcc.osc.mutesync import MuteSync

logger = logging.getLogger("vrcc.core.live")


class LiveApply:
    """Applies live config changes to the engine/audio/OSC stack (GUI thread).

    Holds the long-lived handles the composition root already built; every
    method is a thin delegation so all the thread-safety lives in the target
    components. ``make_source`` turns an ``audio.device`` config string into a
    fresh :class:`AudioSource`; ``make_mute`` builds a coordinator on the first
    enable (mute sync off at launch means none was ever built);
    ``reload_engine`` is the composition root's forced-rebuild closure over the
    reloader.
    """

    def __init__(
        self,
        *,
        pipeline: "Pipeline",
        segmenter: "Segmenter",
        chatbox: "ChatboxSender",
        bus: "EventBus",
        reload_engine: Callable[[str], None],
        make_source: Callable[[str], "AudioSource"],
        make_mute: "Callable[[], MuteSync]",
        mute: "MuteSync | None" = None,
    ) -> None:
        self._pipeline = pipeline
        self._segmenter = segmenter
        self._chatbox = chatbox
        self._bus = bus
        self._reload_engine = reload_engine
        self._make_source = make_source
        self._make_mute = make_mute
        self._mute = mute

    def apply_audio_device(self, device_cfg: str) -> bool:
        """Swap the mic to ``device_cfg`` live. A failed open publishes
        ``MIC_OPEN_FAILED`` (like :func:`~vrcc.app._start_pipeline_guarded`) and
        returns False rather than propagating into the GUI slot; returns whether
        capture is running afterwards."""
        source = self._make_source(device_cfg)
        try:
            return self._pipeline.restart_source(source)
        except Exception as exc:  # noqa: BLE001 -- surface, don't crash the slot
            logger.exception("live audio-device swap could not open the mic")
            self._bus.publish(
                AppError(
                    "MIC_OPEN_FAILED",
                    "Could not open the microphone. Check Settings > Audio",
                    detail=str(exc),
                )
            )
            return False

    def apply_vad(self, cfg: "VadConfig") -> None:
        """Apply new VAD timings/threshold; the next utterance adopts them."""
        self._segmenter.reconfigure(cfg)

    def apply_osc(self, cfg: "OscConfig") -> None:
        """Retarget the chatbox client (ip/port) and retune its send rate."""
        self._chatbox.reconfigure(cfg.ip, cfg.port)
        self._chatbox.reconfigure_rate(cfg.burst, cfg.min_interval_s)

    def apply_mute_sync(self, enabled: bool) -> None:
        """Start or stop mute sync. Mute sync off at launch means no
        coordinator was ever built, so the first enable builds one and installs
        it in the pipeline; disabling stops it but keeps it for the next
        enable. ``MuteSync.start()`` applies its own config and localhost
        gating, and reports its own failures on the bus."""
        if enabled and self._mute is None:
            self._mute = self._make_mute()
            self._pipeline.set_mute(self._mute)
        if self._mute is None:
            return
        if enabled:
            self._mute.start()
        else:
            self._mute.stop()

    def reload_engine(self, kind: str) -> None:
        """Rebuild the ``stt``/``mt`` engine through the reloader. Device /
        compute / thread changes keep the model id, so this forces the swap
        rather than no-opping on the unchanged id."""
        self._reload_engine(kind)
