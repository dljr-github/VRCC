"""Tests for microphone capture: the resample-fallback open path, and one
integration-marked test exercising real audio capture. Device enumeration
lives in ``test_audio_devices.py``.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest
import sounddevice as sd

from vrcc.audio.source import FRAME_LEN, SAMPLE_RATE, MicSource


class FakeStream:
    """Stand-in for `sounddevice.InputStream`: records the kwargs it was
    constructed with, tracks start/stop/close calls, and exposes `deliver()`
    to synchronously invoke the stored callback as PortAudio would. Setting
    `start_error` makes `start()` raise (after counting the attempt), to
    simulate construction succeeding but the stream failing to start."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.start_calls = 0
        self.stop_calls = 0
        self.close_calls = 0
        self.start_error = None

    def start(self):
        self.start_calls += 1
        if self.start_error is not None:
            raise self.start_error

    def stop(self):
        self.stop_calls += 1

    def close(self):
        self.close_calls += 1

    def deliver(self, indata, status=0):
        callback = self.kwargs["callback"]
        callback(indata, indata.shape[0], None, status)


class FakeFactory:
    """Fake `stream_factory`: records every construction attempt and can be
    told to raise `PortAudioError` for calls matching a predicate --
    either at construction time (`fail_when`, simulating the open itself
    failing) or from the created stream's `start()` (`start_fail_when`,
    simulating a stream that opens but cannot start)."""

    def __init__(self, fail_when=None, start_fail_when=None):
        self._fail_when = fail_when
        self._start_fail_when = start_fail_when
        self.attempts: list[dict] = []
        self.streams: list[FakeStream] = []

    def __call__(self, **kwargs):
        self.attempts.append(kwargs)
        if self._fail_when is not None and self._fail_when(kwargs):
            raise sd.PortAudioError("simulated open failure")
        stream = FakeStream(**kwargs)
        if self._start_fail_when is not None and self._start_fail_when(kwargs):
            stream.start_error = sd.PortAudioError("simulated start failure")
        self.streams.append(stream)
        return stream


def _direct_fail(kwargs: dict) -> bool:
    return kwargs.get("samplerate") == SAMPLE_RATE and kwargs.get("blocksize") == FRAME_LEN


class TestMicSourceResampleFallback:
    def test_portaudio_error_falls_back_to_device_default_rate(self, monkeypatch):
        from vrcc.audio import source as source_module

        monkeypatch.setattr(
            source_module.sd,
            "query_devices",
            lambda device, kind: {
                "default_samplerate": 44100.0,
                "max_input_channels": 2,
            },
        )
        factory = FakeFactory(fail_when=_direct_fail)
        mic = MicSource(device=5, stream_factory=factory)

        mic.start(lambda frame: None)

        assert len(factory.attempts) == 2
        first, second = factory.attempts
        assert first["samplerate"] == 16000
        assert first["blocksize"] == 512
        assert second["samplerate"] == 44100.0
        assert second["blocksize"] == 0
        assert second["device"] == 5
        assert factory.streams[0].start_calls == 1  # only the fallback stream started

    def test_fallback_callback_downmixes_resamples_and_rechunks(self, monkeypatch):
        from vrcc.audio import source as source_module

        monkeypatch.setattr(
            source_module.sd,
            "query_devices",
            lambda device, kind: {"default_samplerate": 44100.0, "max_input_channels": 2},
        )

        # Deterministic stand-in for soxr.resample: assert the rates it was
        # given are correct, then return the mono signal unchanged so the
        # rechunker's exact-frame-count math is easy to verify by hand.
        resample_calls = []

        def fake_resample(x, in_rate, out_rate):
            resample_calls.append((in_rate, out_rate))
            return x

        monkeypatch.setattr(source_module.soxr, "resample", fake_resample)

        factory = FakeFactory(fail_when=_direct_fail)
        mic = MicSource(device=None, stream_factory=factory)
        received = []
        mic.start(received.append)

        stream = factory.streams[0]
        # 600 stereo frames -> mono (600,) -> "resampled" (identity stub) ->
        # rechunked: one 512 frame + 88-sample remainder held internally.
        indata = np.tile(np.arange(600, dtype=np.float32).reshape(600, 1), (1, 2))
        stream.deliver(indata)

        assert resample_calls == [(44100.0, 16000)]
        assert len(received) == 1
        assert received[0].shape == (512,)
        assert received[0].dtype == np.float32
        np.testing.assert_array_equal(received[0], np.arange(512, dtype=np.float32))

    def test_fallback_indata_buffer_reuse_does_not_corrupt_emitted_frame(self, monkeypatch):
        from vrcc.audio import source as source_module

        monkeypatch.setattr(
            source_module.sd,
            "query_devices",
            lambda device, kind: {"default_samplerate": 16000.0, "max_input_channels": 1},
        )
        monkeypatch.setattr(source_module.soxr, "resample", lambda x, i, o: x)

        factory = FakeFactory(fail_when=_direct_fail)
        mic = MicSource(stream_factory=factory)
        received = []
        mic.start(received.append)

        indata = np.full((512, 1), 2.0, dtype=np.float32)
        factory.streams[0].deliver(indata)
        indata[:] = -1.0

        assert len(received) == 1
        assert np.all(received[0] == 2.0)

    def test_direct_stream_closed_when_its_start_raises(self, monkeypatch):
        # Construction can succeed and only .start() raise PortAudioError.
        # The half-open direct stream must be closed before the fallback
        # reopens the device (exclusive-mode hosts refuse a second open of
        # a device that still has a live handle).
        from vrcc.audio import source as source_module

        monkeypatch.setattr(
            source_module.sd,
            "query_devices",
            lambda device, kind: {"default_samplerate": 48000.0, "max_input_channels": 2},
        )
        factory = FakeFactory(start_fail_when=_direct_fail)
        mic = MicSource(device=2, stream_factory=factory)

        mic.start(lambda frame: None)

        assert len(factory.streams) == 2
        direct, fallback = factory.streams
        assert direct.start_calls == 1  # the attempt that raised
        assert direct.close_calls == 1  # cleaned up, not leaked
        assert fallback.kwargs["samplerate"] == 48000.0
        assert fallback.kwargs["blocksize"] == 0
        assert fallback.start_calls == 1

    def test_fallback_callback_exception_does_not_propagate(self, monkeypatch, caplog):
        from vrcc.audio import source as source_module

        monkeypatch.setattr(
            source_module.sd,
            "query_devices",
            lambda device, kind: {"default_samplerate": 16000.0, "max_input_channels": 1},
        )

        def bad_resample(x, i, o):
            raise RuntimeError("resample blew up")

        monkeypatch.setattr(source_module.soxr, "resample", bad_resample)

        factory = FakeFactory(fail_when=_direct_fail)
        mic = MicSource(stream_factory=factory)
        mic.start(lambda frame: None)

        indata = np.zeros((512, 1), dtype=np.float32)
        with caplog.at_level(logging.ERROR, logger="vrcc.audio"):
            factory.streams[0].deliver(indata)  # must not raise


def test_fallback_path_applies_gain_per_512_frame_not_per_callback_chunk(monkeypatch):
    # Gain must be applied inside _emit (per exact 512-sample frame), not in
    # the fallback callback on the raw host-sized chunk: applying it before
    # the rechunker would pace auto-gain at the host's ~85ms callback rate
    # instead of the 32ms frame rate the direct path gets.
    from vrcc.audio import source as source_module
    from vrcc.audio.gain import GainProcessor

    seen_sizes = []

    class RecordingGain(GainProcessor):
        def process(self, frame):
            seen_sizes.append(frame.size)
            return super().process(frame)

    gain = RecordingGain()
    gain.configure(0.0, auto=False)

    monkeypatch.setattr(
        source_module.sd,
        "query_devices",
        lambda device, kind: {"default_samplerate": 44100.0, "max_input_channels": 1},
    )
    monkeypatch.setattr(source_module.soxr, "resample", lambda x, i, o: x)

    factory = FakeFactory(fail_when=_direct_fail)
    mic = MicSource(device=None, stream_factory=factory, gain=gain)
    mic.start(lambda frame: None)

    # 1024 host-rate samples in one callback -> two 512-sample frames after
    # the (identity-stubbed) resample + rechunk.
    indata = np.arange(1024, dtype=np.float32).reshape(1024, 1)
    factory.streams[0].deliver(indata)

    assert seen_sizes == [512, 512]


def test_mic_source_applies_gain_to_frames():
    from vrcc.audio.gain import GainProcessor

    frames = []
    gain = GainProcessor()
    gain.configure(6.0206, auto=False)  # ~2x

    captured = {}

    class FakeStream:
        def __init__(self, **kw):
            captured["callback"] = kw["callback"]
        def start(self):
            pass
        def stop(self):
            pass
        def close(self):
            pass

    src = MicSource(device=None, stream_factory=FakeStream, gain=gain)
    src.start(frames.append)
    cb = captured["callback"]
    block = np.full((512, 1), 0.1, dtype=np.float32)
    cb(block, 512, None, None)
    assert frames, "no frame emitted"
    assert float(np.sqrt(np.mean(frames[0] ** 2))) > 0.19


class TestMicSourceHardware:
    @pytest.mark.integration
    def test_real_capture_produces_frames(self):
        import time

        received = []
        mic = MicSource()
        mic.start(received.append)
        try:
            time.sleep(1.0)
        finally:
            mic.stop()

        assert len(received) > 0
        assert received[0].shape == (FRAME_LEN,)
        assert received[0].dtype == np.float32
