"""Assemble the engine stack from config (or injected fakes). Qt-free.

Split out of app.py so the composition root stays under the source cap; imports
no Qt and starts no threads or servers, so it stays unit-testable without a
display (same rationale as vrcc/core/startup.py).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

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

if TYPE_CHECKING:
    from vrcc.core.heard import HeardStream

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

    The source is rebuilt on every start because a LoopbackSource is bound to
    one device for its lifetime, so a speaker change only takes effect here.
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
        # Runs on the capture thread. Switch the setting back off so the toggle
        # cannot sit on "on" beside a stream that is not running. The code has a
        # sentence in FRIENDLY_ERRORS; detail is for the log.
        cfg.audio.hear_others_enabled = False
        heard.bus.publish(AppError(code, detail))

    heard.set_source(
        LoopbackSource(cfg.audio.hear_others_device or None, on_failure=on_failure)
    )
    heard.start()


def start_hear_others_guarded(stack: EngineStack, cfg, bus: EventBus) -> None:
    """Start the speaker-capture stream at launch, if the user turned it on.

    Never fatal: captioning what you hear is an extra, and losing it must not
    take the microphone down with it. A failure inside the capture thread is
    already handled there; this covers a start() that raises outright.
    """
    if stack.heard is None:
        return
    try:
        apply_hear_others(stack, cfg)
    except Exception as exc:
        logger.warning(
            "could not start captioning what you hear; the microphone is "
            "unaffected", exc_info=True,
        )
        # The code the capture thread uses for the same outcome, rather than
        # one of its own: it is the code that carries a sentence a user can
        # act on, and the one the main window watches to put the toggle back
        # down. Switch the setting off here too, since nothing else will.
        cfg.audio.hear_others_enabled = False
        bus.publish(AppError("HEARD_DEVICE_FAILED", f"start failed: {exc}"))


class EngineOwners:
    """Route an engine hot-swap into every consumer of the shared engines.

    :class:`~vrcc.core.reloading._Reloader` installs into one object, and for
    most of this app's life that object was the pipeline. The heard stream is
    a second holder of the same STT and MT engines, so a swap that reached only
    the pipeline left it calling an engine the reloader had already unloaded:
    every decode raised, the handler swallowed it, and captioning what you hear
    went silent for the rest of the session with its toggle still lit.

    Exposes exactly the four methods the reloader uses, so it drops in where
    the pipeline did.
    """

    def __init__(self, pipeline, heard) -> None:
        self._pipeline = pipeline
        self._heard = heard

    def detach_stt(self):
        # The consumer first. It reads its engine under the same lock
        # detach_stt takes, so clearing it here makes the wait inside
        # detach_stt the last decode that engine can ever see.
        if self._heard is not None:
            self._heard.set_stt(None)
        return self._pipeline.detach_stt()

    def set_stt(self, engine) -> None:
        self._pipeline.set_stt(engine)
        if self._heard is not None:
            self._heard.set_stt(engine)

    def detach_mt(self):
        if self._heard is not None:
            self._heard.set_mt(None)
        return self._pipeline.detach_mt()

    def set_mt(self, engine) -> None:
        self._pipeline.set_mt(engine)
        if self._heard is not None:
            self._heard.set_mt(engine)
