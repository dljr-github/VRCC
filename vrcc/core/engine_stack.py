"""Assemble the engine stack from config (or injected fakes). Qt-free.

Split out of app.py so the composition root stays under the source cap; imports
no Qt and starts no threads or servers, so it stays unit-testable without a
display (same rationale as vrcc/core/startup.py).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from vrcc.audio.segmenter import Segmenter
from vrcc.audio.source import AudioSource, MicSource
from vrcc.audio.vad import StreamingVad
from vrcc.core.bus import EventBus
from vrcc.core.events import AppError, MicLevel
from vrcc.core.config import ConfigStore, Paths
from vrcc.core.pipeline import Pipeline
from vrcc.core.startup import resolve_audio_device as _resolve_audio_device
from vrcc.osc.chatbox import ChatboxSender
from vrcc.osc.mutesync import MuteSync
from vrcc.stt import create_stt_engine
from vrcc.stt.engine import SttEngine
from vrcc.translate.engine import TranslateEngine
from vrcc.translate.registry import MT_MODELS

logger = logging.getLogger("vrcc.core.engine_stack")

# Sentinel: "argument not supplied" vs an explicit None (mt/mute are
# legitimately None when translation / mute sync is disabled).
_UNSET = object()


@dataclass
class EngineStack:
    """Everything run() needs to operate the app, built by build_engine_stack.
    A plain data holder -- it starts nothing."""

    pipeline: Pipeline
    source: AudioSource
    segmenter: Segmenter
    vad: StreamingVad | None
    stt: SttEngine
    mt: TranslateEngine | None
    chatbox: ChatboxSender
    mute: MuteSync | None
    # Present only when the user opted in to captioning what they hear. Its own
    # capture and segmenter, the pipeline's engines and locks.
    heard: "HeardStream | None" = None


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
        from vrcc.audio.denoise import Denoiser

        denoiser = Denoiser()
        denoiser.configure(cfg.audio.denoise_enabled, cfg.audio.denoise_strength)
        source = MicSource(
            _resolve_audio_device(cfg.audio.device), denoiser=denoiser
        )

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

    heard = _build_heard(cfg, bus, pipeline, stt_engine, mt_engine)
    if heard is not None:
        # The mic's own VAD decides when the user is speaking; the heard stream
        # only needs to be told, so it can drop its own echo.
        bus.subscribe(MicLevel, lambda e: heard.note_mic_level(e.rms, e.vad_prob))

    return EngineStack(
        pipeline=pipeline,
        source=source,
        segmenter=segmenter,
        vad=vad,
        stt=stt_engine,
        mt=mt_engine,
        chatbox=chatbox,
        mute=mute,
        heard=heard,
    )


def _build_heard(cfg, bus, pipeline, stt_engine, mt_engine):
    """The speaker-capture stream, built whether or not it is switched on.

    Built unconditionally because the user can turn it on mid-session and a
    relaunch to hear the room is a poor trade. Construction is cheap: no device
    is opened and soundcard is not even imported until start().

    Its own Segmenter, because VAD state is per stream and one shared instance
    would let either voice end the other's utterance. The ENGINES are the
    pipeline's, under the pipeline's locks: a second copy of the voice model
    costs the VRAM the card was sized for once.
    """
    from vrcc.audio.loopback import LoopbackSource
    from vrcc.core.heard import HeardStream

    heard_vad = StreamingVad(threshold=cfg.vad.threshold)
    return HeardStream(
        cfg,
        bus,
        LoopbackSource(cfg.audio.hear_others_device or None),
        Segmenter(cfg.vad, heard_vad.prob),
        stt_engine,
        mt_engine,
        pipeline._stt_lock,
        pipeline._mt_lock,
    )


def apply_hear_others(stack: EngineStack, cfg) -> None:
    """Start or stop the speaker-capture stream to match config.

    Called from the main-window toggle and after Settings closes, so turning it
    on never needs a relaunch. The source is rebuilt on every start because a
    LoopbackSource is bound to one device for its lifetime, which is how a
    speaker change takes effect.
    """
    heard = stack.heard
    if heard is None:
        return
    if not cfg.audio.hear_others_enabled:
        heard.stop()
        return
    from vrcc.audio.loopback import LoopbackSource

    if heard.running:
        heard.stop()

    def on_failure(code: str, detail: str) -> None:
        # From the capture thread. Switch the setting back off so the toggle
        # cannot sit on "on" beside a stream that is not running, which is
        # what made this feature look like it did nothing at all. The code
        # carries a sentence of its own in FRIENDLY_ERRORS; detail is for the
        # log, where the underlying exception text belongs.
        cfg.audio.hear_others_enabled = False
        heard.bus.publish(AppError(code, detail))

    heard.set_source(
        LoopbackSource(cfg.audio.hear_others_device or None, on_failure=on_failure)
    )
    heard.start()
