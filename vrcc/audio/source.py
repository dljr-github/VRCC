"""Microphone capture producing exact 512-sample float32 mono frames at 16 kHz.

Opens `sounddevice.InputStream` directly; on a PortAudioError (exclusive /
native-rate devices) reopens at the device rate, downmixing + resampling (soxr)
+ rechunking. Callback failures never propagate (they'd tear down the stream):
each category logs once, counts repeats, summarized on stop(). Zero Qt.
"""

from __future__ import annotations

import logging
from typing import Callable, Protocol

import numpy as np
import sounddevice as sd
import soxr

from vrcc.audio.denoise import Denoiser

logger = logging.getLogger("vrcc.audio")

FRAME_LEN = 512
SAMPLE_RATE = 16000


class AudioSource(Protocol):
    def start(self, on_frame: Callable[[np.ndarray], None]) -> None:
        """Begin capture; `on_frame` is called with a fresh float32[512]
        mono array at 16 kHz for every complete frame produced."""
        ...

    def stop(self) -> None:
        """Stop capture. Safe to call more than once, or before `start`."""
        ...


def _to_mono(x: np.ndarray) -> np.ndarray:
    """Downmix `x` to a 1-D mono float32 signal.

    1-D passes through unchanged; 2-D (frames, channels) is averaged over the
    channel axis (`np.mean` allocates fresh, so that branch is copy-safe; the
    1-D passthrough is not -- callers needing a copy make their own).
    """
    if x.ndim == 1:
        return x
    return x.mean(axis=1, dtype=np.float32)


class _Rechunker:
    """Accumulates pushed arrays and yields exact `frame_len`-sample frames,
    carrying the partial remainder to the next `push()`. Every returned frame is
    freshly allocated (never a view), since PortAudio's callback `indata` may be
    reused/overwritten right after `push()` returns.
    """

    def __init__(self, frame_len: int = FRAME_LEN) -> None:
        self._frame_len = frame_len
        self._remainder = np.empty(0, dtype=np.float32)

    def push(self, samples: np.ndarray) -> list[np.ndarray]:
        combined = np.concatenate(
            [self._remainder, np.asarray(samples, dtype=np.float32)]
        )
        n = self._frame_len
        full_count = combined.shape[0] // n

        frames = [combined[i * n : (i + 1) * n].copy() for i in range(full_count)]
        self._remainder = combined[full_count * n :].copy()
        return frames


class MicSource:
    """`AudioSource` backed by a real microphone via `sounddevice`.

    `device` is a PortAudio index or `None` (system default). `stream_factory`
    defaults to `sounddevice.InputStream`; tests inject a fake with the same
    signature to drive `MicSource` without hardware.
    """

    def __init__(
        self,
        device: int | None = None,
        stream_factory: Callable[..., object] | None = None,
        denoiser: Denoiser | None = None,
    ) -> None:
        self._device = device
        self._stream_factory = stream_factory if stream_factory is not None else sd.InputStream
        self._denoiser = denoiser
        self._stream = None
        self._on_frame: Callable[[np.ndarray], None] | None = None
        self._rechunker = _Rechunker(FRAME_LEN)
        self._resample_in_rate: float | None = None
        # Log-flood guards for the ~31 Hz callback: each category logs once,
        # counts repeats, summarized on stop(). Reset on every start().
        self._on_frame_errors = 0
        self._callback_errors = 0
        self._status_flags = 0

    def start(self, on_frame: Callable[[np.ndarray], None]) -> None:
        if self._stream is not None:
            logger.warning(
                "MicSource.start() called while already capturing; "
                "stopping the previous stream first"
            )
            self.stop()

        self._on_frame = on_frame
        self._rechunker = _Rechunker(FRAME_LEN)  # never carry a stale remainder
        self._on_frame_errors = 0
        self._callback_errors = 0
        self._status_flags = 0

        stream = None
        try:
            stream = self._stream_factory(
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="float32",
                blocksize=FRAME_LEN,
                device=self._device,
                callback=self._direct_callback,
            )
            stream.start()
        except sd.PortAudioError:
            logger.warning(
                "direct 16 kHz/mono/512 stream open failed for device %r; "
                "falling back to the device's native rate with resampling",
                self._device,
                exc_info=True,
            )
            if stream is not None:
                # .start() raised after construction: close the half-open stream
                # before the fallback reopens the device (exclusive hosts refuse
                # a second open).
                try:
                    stream.close()
                except Exception:
                    logger.warning(
                        "failed to close partially-opened direct stream",
                        exc_info=True,
                    )
            stream = self._start_fallback()

        self._stream = stream

    def _start_fallback(self):
        info = sd.query_devices(self._device, "input")
        in_rate = float(info["default_samplerate"])
        channels = max(1, int(info.get("max_input_channels", 1)))
        self._resample_in_rate = in_rate

        stream = self._stream_factory(
            samplerate=in_rate,
            channels=channels,
            dtype="float32",
            blocksize=0,
            device=self._device,
            callback=self._fallback_callback,
        )
        stream.start()
        return stream

    def set_denoise(self, enabled: bool, strength: float) -> None:
        """Update the live denoiser; no stream restart. No-op if no processor."""
        if self._denoiser is not None:
            self._denoiser.configure(enabled, strength)

    def stop(self) -> None:
        # A restart must not inherit stale recurrent/smoothing state from the
        # prior run's stream. stop() runs on the caller's thread while the
        # audio callback may still be mid-flight on the audio thread, so the
        # resets wait until the stream is fully stopped and closed; resetting
        # earlier could interleave with a callback's in-progress cache update.
        if self._stream is None:
            if self._denoiser is not None:
                self._denoiser.reset()
            return
        stream, self._stream = self._stream, None
        try:
            stream.stop()
            stream.close()
        except Exception:
            logger.warning("error stopping audio stream", exc_info=True)
        if self._denoiser is not None:
            self._denoiser.reset()
        self._log_suppressed_summary()

    def _log_suppressed_summary(self) -> None:
        parts = []
        if self._on_frame_errors > 1:
            parts.append(f"{self._on_frame_errors - 1} repeated on_frame errors")
        if self._callback_errors > 1:
            parts.append(f"{self._callback_errors - 1} repeated callback errors")
        if self._status_flags > 1:
            parts.append(f"{self._status_flags - 1} repeated stream status flags")
        if parts:
            logger.warning("audio capture stopped; suppressed: %s", "; ".join(parts))

    def _note_status(self, status) -> None:
        self._status_flags += 1
        if self._status_flags == 1:
            logger.warning(
                "audio input stream status: %s (repeats will be counted and "
                "summarized on stop)",
                status,
            )

    def _apply_denoise(self, mono: np.ndarray) -> np.ndarray:
        return self._denoiser.process(mono) if self._denoiser is not None else mono

    def _direct_callback(self, indata, frames, time, status) -> None:
        # indata is PortAudio-owned/reused; _to_mono's 2-D branch allocates
        # fresh via np.mean and _Rechunker.push copies again, so no separate
        # copy is needed here.
        try:
            if status:
                self._note_status(status)
            mono = _to_mono(indata)
            for frame in self._rechunker.push(mono):
                self._emit(frame)
        except Exception:
            self._callback_errors += 1
            if self._callback_errors == 1:
                logger.exception(
                    "unhandled error in direct audio callback (repeats will "
                    "be counted and summarized on stop)"
                )

    def _fallback_callback(self, indata, frames, time, status) -> None:
        # soxr.resample() treats each chunk independently, so filter state
        # resets at chunk boundaries (tiny ~31x/sec discontinuity).
        # soxr.ResampleStream is the stateful upgrade if that proves audible.
        try:
            if status:
                self._note_status(status)
            mono = _to_mono(indata)
            resampled = soxr.resample(mono, self._resample_in_rate, SAMPLE_RATE)
            resampled = np.asarray(resampled, dtype=np.float32)
            for frame in self._rechunker.push(resampled):
                self._emit(frame)
        except Exception:
            self._callback_errors += 1
            if self._callback_errors == 1:
                logger.exception(
                    "unhandled error in resample-fallback audio callback "
                    "(repeats will be counted and summarized on stop)"
                )

    def _emit(self, frame: np.ndarray) -> None:
        # Denoise is applied here, per exact 512-sample frame, rather than in
        # the callback, so both the direct and resample-fallback paths share
        # one place that touches the samples before on_frame sees them.
        try:
            self._on_frame(self._apply_denoise(frame))
        except Exception:
            self._on_frame_errors += 1
            if self._on_frame_errors == 1:
                logger.exception(
                    "on_frame raised; continuing capture (repeats will be "
                    "counted and summarized on stop)"
                )
