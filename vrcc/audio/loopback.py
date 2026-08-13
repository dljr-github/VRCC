"""Capture what the speakers are playing, so VRCC can caption other people.

An :class:`~vrcc.audio.source.AudioSource` backed by WASAPI loopback. PortAudio
cannot do this: sounddevice exposes no loopback flag and an output device
refuses to open as an input, so this backend uses ``soundcard`` (WASAPI via
ctypes, no rebuild) while the microphone keeps its PortAudio path.

Its recorder is blocking rather than callback-driven, so capture runs on its
own thread. WASAPI is COM, so the capture thread has to join an apartment
itself: ``soundcard`` initialises COM at import on the IMPORTING thread only,
and a worker that skips :func:`_com_initialize` fails with CO_E_NOTINITIALIZED
(0x800401f0) at the first device call.

This captures the whole output device, not VRChat's voice channel. Windows
offers no per-application voice tap, so world audio, music and any other app
land in the same stream. The VAD and the quality gates reject some of it, but
audio from a video player or a stream is transcribed too.
"""

from __future__ import annotations

import logging
import threading
from typing import Callable

import numpy as np

from vrcc.audio.source import FRAME_LEN, SAMPLE_RATE, _Rechunker, _to_mono

logger = logging.getLogger("vrcc.audio.loopback")

# How many frames to ask for per blocking read. One frame at a time would wake
# the thread 31 times a second for 32 ms of audio; this trades a little latency
# for far fewer wakeups, and the segmenter re-chunks to FRAME_LEN regardless.
_READ_FRAMES = FRAME_LEN * 4

# A read that raises is usually a device disappearing (headset unplugged, output
# switched). Retry a few times before giving up, so a transient glitch does not
# end captioning for the session.
_MAX_CONSECUTIVE_ERRORS = 5


# CoInitializeEx flags. Multithreaded matches how the capture thread uses the
# device (one long blocking read loop, no window messages), and is what WASAPI
# wants. RPC_E_CHANGED_MODE means this thread is already in the other kind of
# apartment, which is fine: something else initialised it and the calls work.
_COINIT_MULTITHREADED = 0x0
_RPC_E_CHANGED_MODE = -2147417850  # 0x80010106


def _com_initialize() -> bool:
    """Join a COM apartment on the calling thread. True if this call is the one
    that must uninitialize it."""
    import ctypes

    hr = ctypes.windll.ole32.CoInitializeEx(None, _COINIT_MULTITHREADED)
    if hr == _RPC_E_CHANGED_MODE:
        return False
    return hr >= 0


def _com_uninitialize() -> None:
    import ctypes

    ctypes.windll.ole32.CoUninitialize()


def _soundcard():
    """Import ``soundcard`` lazily.

    It initialises COM at import time on Windows, so keeping it out of module
    import means a user who never turns this on never pays for it, and a
    missing dependency surfaces when the feature is used rather than at launch.
    """
    import soundcard

    return soundcard


def loopback_devices() -> list[str]:
    """Names of the output devices that can be captured, playback first.

    Names rather than indices: ``soundcard`` identifies devices by name, and an
    index would not survive the user plugging in a headset anyway. Never raises;
    an unavailable audio stack yields an empty list, which the UI shows as "no
    speakers found" rather than crashing.
    """
    try:
        sc = _soundcard()
        return [mic.name for mic in sc.all_microphones(include_loopback=True)
                if getattr(mic, "isloopback", False)]
    except Exception:
        logger.debug("failed to list loopback devices", exc_info=True)
        return []


def default_loopback_device() -> str | None:
    """The name of the default speaker, or ``None`` if there is not one."""
    try:
        return _soundcard().default_speaker().name
    except Exception:
        logger.debug("failed to resolve the default speaker", exc_info=True)
        return None


class LoopbackSource:
    """`AudioSource` capturing an output device via WASAPI loopback.

    ``device`` is a name from :func:`loopback_devices`, or ``None`` for the
    current default speaker, resolved at :meth:`start` so a user who changes
    their default between sessions is followed.
    """

    def __init__(
        self, device: str | None = None, recorder_factory=None, on_failure=None
    ) -> None:
        self._device = device
        # Called with (error code, detail) when capture cannot run. The failure
        # happens on the worker thread, so it can never reach a try/except
        # around start(); without it the feature is dead with nothing on
        # screen.
        self._on_failure = on_failure
        # Defaults to soundcard's recorder; tests inject a fake exposing the
        # same record(numframes) context manager.
        self._recorder_factory = recorder_factory
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self.read_errors = 0

    def start(self, on_frame: Callable[[np.ndarray], None]) -> None:
        if self._thread is not None:
            logger.warning(
                "LoopbackSource.start() called while already capturing; "
                "stopping the previous capture first"
            )
            self.stop()

        self._stop.clear()
        self.read_errors = 0
        self._thread = threading.Thread(
            target=self._run, args=(on_frame,), name="LoopbackSource", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        thread, self._thread = self._thread, None
        if thread is not None:
            # Bounded: a blocking record() call returns within one read period,
            # and shutdown must not hang on a wedged audio device.
            thread.join(timeout=2.0)

    def _fail(self, code: str, detail: str) -> None:
        # A stop the caller asked for is not a failure, whichever exit path
        # notices it. A speaker change tears this source down and starts
        # another, and reporting the teardown would switch the setting off and
        # unpick the toggle over the capture that had just replaced it.
        if self._on_failure is None or self._stop.is_set():
            return
        try:
            self._on_failure(code, detail)
        except Exception:
            logger.debug("loopback failure callback raised", exc_info=True)

    def _open(self):
        if self._recorder_factory is not None:
            return self._recorder_factory(self._device)
        sc = _soundcard()
        name = self._device or sc.default_speaker().name
        mic = sc.get_microphone(name, include_loopback=True)
        return mic.recorder(
            samplerate=SAMPLE_RATE, channels=1, blocksize=_READ_FRAMES
        )

    def _run(self, on_frame: Callable[[np.ndarray], None]) -> None:
        rechunker = _Rechunker(FRAME_LEN)
        # Import BEFORE joining an apartment. soundcard initialises COM in its
        # own module body and treats S_FALSE ("this thread already has one") as
        # a fatal error, so initialising first makes its very first import
        # raise 0x100000001. Importing first lets it initialise this thread
        # when it is the first import, and the call below then returns S_FALSE
        # harmlessly; when it was imported elsewhere the import is a cached
        # no-op and the call below is what gives this thread its apartment.
        owns_com = False
        if self._recorder_factory is None:
            try:
                _soundcard()
            except Exception:
                logger.warning("soundcard is unavailable", exc_info=True)
                self._fail("HEARD_NO_LIBRARY", "soundcard is not installed")
                return
            owns_com = _com_initialize()
        try:
            self._capture(on_frame, rechunker)
        finally:
            if owns_com:
                _com_uninitialize()

    def _capture(self, on_frame, rechunker) -> None:
        # The `with` is inside the try, not around it. recorder() only builds
        # the object: it is __enter__ that starts the WASAPI stream, so
        # exclusive-mode contention and an endpoint that refuses 16 kHz mono
        # both raise there rather than from _open(), and outside the try they
        # would kill this thread with nothing on screen.
        try:
            with self._open() as rec:
                self._read_loop(rec, on_frame, rechunker)
        except Exception as exc:
            logger.warning(
                "loopback capture for %r ended in an error; captioning what "
                "you hear is off for this session",
                self._device, exc_info=True,
            )
            self._fail("HEARD_DEVICE_FAILED", f"the output device failed: {exc}")

    def _read_loop(self, rec, on_frame, rechunker) -> None:
        consecutive = 0
        while not self._stop.is_set():
            try:
                block = rec.record(numframes=_READ_FRAMES)
                consecutive = 0
            except Exception as exc:
                self.read_errors += 1
                consecutive += 1
                logger.warning("loopback read failed", exc_info=True)
                if consecutive >= _MAX_CONSECUTIVE_ERRORS:
                    logger.warning(
                        "giving up on loopback capture after %d consecutive "
                        "read failures", consecutive
                    )
                    # Every way this thread ends without the caller asking has
                    # to say so, or the toggle sits lit beside a dead stream.
                    self._fail(
                        "HEARD_DEVICE_FAILED",
                        f"the output device stopped delivering audio: {exc}",
                    )
                    return
                continue
            for frame in rechunker.push(_to_mono(np.asarray(block))):
                try:
                    on_frame(frame)
                except Exception:
                    # One bad frame must not end capture, the same contract
                    # MicSource's callback keeps.
                    logger.warning("loopback on_frame raised", exc_info=True)
