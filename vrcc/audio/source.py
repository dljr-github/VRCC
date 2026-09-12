"""Microphone capture producing exact 512-sample float32 mono frames at 16 kHz.

Opens `sounddevice.InputStream` directly at 16 kHz mono, asking a WASAPI host
to convert the rate (WASAPI shared mode otherwise refuses 16 kHz outright, on
every device measured on this machine); on a PortAudioError reopens at the
device's native rate, downmixing + resampling (soxr) + rechunking -- the path
every WASAPI shared-mode microphone takes, not an exotic exclusive-mode one.
Callback failures never propagate (they'd tear down the stream): each
category logs once, counts repeats, summarized on stop(). Zero Qt.
"""

from __future__ import annotations

import logging
from typing import Callable, Protocol

import numpy as np
import sounddevice as sd
import soxr

from vrcc.audio.denoise import Denoiser
from vrcc.audio.frames import FRAME_LEN, SAMPLE_RATE

logger = logging.getLogger("vrcc.audio")


class AudioSource(Protocol):
    def start(self, on_frame: Callable[[np.ndarray], None]) -> None:
        """Begin capture; `on_frame` is called with a fresh float32[512]
        mono array at 16 kHz for every complete frame produced."""
        ...

    def stop(self) -> None:
        """Stop capture. Safe to call more than once, or before `start`."""
        ...


# float32 full scale is +-1.0; "at or beyond" catches a driver that never
# lands on exactly 1.0.
_CLIP_THRESHOLD = 0.999


def _count_clipped(indata: np.ndarray) -> int:
    """Count samples (rows) with any channel at or beyond `_CLIP_THRESHOLD`.

    Any-channel semantics on raw `indata`, ahead of the mono downmix and (on
    the fallback path) the resampler: averaging channels can mask a clipped
    one, and a polyphase resample filter both softens a true flat top and
    overshoots past 1.0 on a hot-but-unclipped signal, so neither downstream
    signal reports what the driver actually delivered.
    """
    peak = np.abs(indata) if indata.ndim == 1 else np.abs(indata).max(axis=1)
    return int(np.count_nonzero(peak >= _CLIP_THRESHOLD))


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
        # Saturation counters: raw-indata samples (any channel) at or beyond
        # _CLIP_THRESHOLD, and the total seen, reset alongside the error
        # counters below on every start().
        self._clipped_samples = 0
        self._total_samples = 0

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
        self._clipped_samples = 0
        self._total_samples = 0
        # Cleared here (not just in __init__) so the open-summary log below
        # reflects this session's chain, not a fallback from a prior one.
        self._resample_in_rate = None

        stream = None
        extra = None
        hostapi_name = "unknown"  # only the WASAPI probe below can name it
        try:
            # WASAPI shared mode rejects 16 kHz outright (measured with
            # sd.check_input_settings on every WASAPI input on this
            # machine: PaErrorCode -9997 "Invalid sample rate"); asking it
            # to convert is what lets the direct 16 kHz/mono/512 open
            # succeed instead of always falling back for these devices.
            # MME raises -9984 "Incompatible host API specific stream
            # info" if handed a WasapiSettings (measured on two MME
            # indices), so this must only apply to a WASAPI device.
            hostapi_index = sd.query_devices(self._device, "input")["hostapi"]
            hostapi_name = sd.query_hostapis(hostapi_index)["name"]
            wasapi_settings = getattr(sd, "WasapiSettings", None)
            if wasapi_settings is not None and "WASAPI" in hostapi_name.upper():
                extra = wasapi_settings(auto_convert=True)
        except Exception:
            # sd.query_devices raises PortAudioError for an invalid index,
            # and CI has no capture hardware to query at all; the probe is
            # informational only, so any failure here must degrade to the
            # unconverted direct open (and from there to the existing
            # fallback) rather than raising out of start(). Logged because an
            # unanticipated failure here is otherwise indistinguishable from
            # a device that simply is not WASAPI.
            logger.debug("host API probe failed; opening without conversion", exc_info=True)

        try:
            open_kwargs = dict(
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="float32",
                blocksize=FRAME_LEN,
                device=self._device,
                callback=self._direct_callback,
            )
            if extra is not None:
                open_kwargs["extra_settings"] = extra
            stream = self._stream_factory(**open_kwargs)
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
        logger.info(
            "microphone capture open: device=%r host_api=%s resample_fallback=%s",
            self._device,
            hostapi_name,
            self._resample_in_rate is not None,
        )

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
        # Saturation is a measurement, not a de-duplicated repeat, so it gets
        # its own labeled segment rather than joining the "suppressed:" list
        # -- folding it in there would read as if the loud samples themselves
        # had been suppressed, which is false on a session with no other errors.
        error_parts = []
        if self._on_frame_errors > 1:
            error_parts.append(f"{self._on_frame_errors - 1} repeated on_frame errors")
        if self._callback_errors > 1:
            error_parts.append(f"{self._callback_errors - 1} repeated callback errors")
        if self._status_flags > 1:
            error_parts.append(f"{self._status_flags - 1} repeated stream status flags")

        saturation = None
        if self._clipped_samples > 0:
            fraction = self._clipped_samples / self._total_samples
            saturation = (
                f"saturation: {fraction:.2%} samples "
                f"({self._clipped_samples}/{self._total_samples})"
            )

        if not error_parts and saturation is None:
            return
        segments = []
        if error_parts:
            segments.append("suppressed: " + "; ".join(error_parts))
        if saturation is not None:
            segments.append(saturation)
        logger.warning("audio capture stopped; %s", "; ".join(segments))

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
            self._total_samples += indata.shape[0]
            self._clipped_samples += _count_clipped(indata)
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
            self._total_samples += indata.shape[0]
            self._clipped_samples += _count_clipped(indata)
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
