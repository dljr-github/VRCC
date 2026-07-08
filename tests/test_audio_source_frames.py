"""Tests for microphone capture: mono downmix, the rechunker, and the
direct-open MicSource frame/callback path (via a fake ``stream_factory``).
"""

from __future__ import annotations

import logging

import numpy as np
import sounddevice as sd

from vrcc.audio.source import MicSource, _Rechunker, _to_mono


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


class TestToMono:
    def test_1d_is_passthrough(self):
        x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        assert _to_mono(x) is x

    def test_2d_means_over_channel_axis(self):
        x = np.array([[1.0, 3.0], [2.0, 4.0]], dtype=np.float32)
        result = _to_mono(x)
        assert result.dtype == np.float32
        assert result.shape == (2,)
        np.testing.assert_allclose(result, [2.0, 3.0])

    def test_2d_result_is_independent_of_input(self):
        # mean() always allocates a fresh array; mutating the source after
        # the call must not change the returned mono signal.
        x = np.array([[1.0, 1.0], [2.0, 2.0]], dtype=np.float32)
        result = _to_mono(x).copy()
        x[:] = 999.0
        np.testing.assert_allclose(result, [1.0, 2.0])


class TestRechunker:
    def test_push_below_frame_len_yields_nothing(self):
        r = _Rechunker(512)
        frames = r.push(np.zeros(300, dtype=np.float32))
        assert frames == []

    def test_300_then_300_yields_one_frame_and_88_remainder(self):
        r = _Rechunker(512)
        r.push(np.arange(300, dtype=np.float32))
        frames = r.push(np.arange(300, 600, dtype=np.float32))
        assert len(frames) == 1
        assert frames[0].shape == (512,)
        expected = np.arange(600, dtype=np.float32)[:512]
        np.testing.assert_array_equal(frames[0], expected)
        assert r._remainder.shape == (88,)
        np.testing.assert_array_equal(r._remainder, np.arange(600, dtype=np.float32)[512:])

    def test_push_1024_yields_two_frames_no_remainder(self):
        r = _Rechunker(512)
        frames = r.push(np.arange(1024, dtype=np.float32))
        assert len(frames) == 2
        assert frames[0].shape == (512,)
        assert frames[1].shape == (512,)
        np.testing.assert_array_equal(frames[0], np.arange(0, 512, dtype=np.float32))
        np.testing.assert_array_equal(frames[1], np.arange(512, 1024, dtype=np.float32))
        assert r._remainder.shape == (0,)

    def test_returned_frames_are_fresh_arrays_not_views(self):
        r = _Rechunker(512)
        source = np.arange(512, dtype=np.float32)
        frames = r.push(source)
        assert len(frames) == 1
        source[:] = -1.0  # mutate caller's buffer after push returns
        assert np.all(frames[0] == np.arange(512, dtype=np.float32))

    def test_returned_frames_own_their_memory_not_views(self):
        # Contract (brief): "each returned frame is a fresh array (no views
        # into the input)". `np.concatenate` itself never aliases its
        # inputs, so a mutation-based test can't distinguish a genuine copy
        # from an unowned slice of the internal concatenated buffer here --
        # check ownership directly. A numpy view has `.base` pointing at its
        # parent array; an independent copy has `.base is None`.
        r = _Rechunker(512)
        frames = r.push(np.arange(1024, dtype=np.float32))
        assert len(frames) == 2
        assert frames[0].base is None
        assert frames[1].base is None

    def test_remainder_survives_across_many_small_pushes(self):
        r = _Rechunker(512)
        collected = []
        for _ in range(4):
            collected.extend(r.push(np.full(200, 1.0, dtype=np.float32)))
        # 4*200 = 800 -> one 512 frame, 288 remainder
        assert len(collected) == 1
        assert r._remainder.shape == (288,)

    def test_default_frame_len_is_512(self):
        r = _Rechunker()
        frames = r.push(np.zeros(512, dtype=np.float32))
        assert len(frames) == 1


class TestMicSourceDirectOpen:
    def test_opens_16k_mono_512_stream_with_requested_device(self):
        factory = FakeFactory()
        source = MicSource(device=3, stream_factory=factory)

        received = []
        source.start(received.append)

        assert len(factory.attempts) == 1
        kwargs = factory.attempts[0]
        assert kwargs["samplerate"] == 16000
        assert kwargs["channels"] == 1
        assert kwargs["dtype"] == "float32"
        assert kwargs["blocksize"] == 512
        assert kwargs["device"] == 3
        assert callable(kwargs["callback"])
        assert factory.streams[0].start_calls == 1

    def test_callback_delivers_flattened_512_frame(self):
        factory = FakeFactory()
        source = MicSource(device=None, stream_factory=factory)
        received = []
        source.start(received.append)

        indata = np.arange(512, dtype=np.float32).reshape(512, 1)
        factory.streams[0].deliver(indata)

        assert len(received) == 1
        assert received[0].shape == (512,)
        assert received[0].dtype == np.float32
        np.testing.assert_array_equal(received[0], np.arange(512, dtype=np.float32))

    def test_indata_buffer_reuse_does_not_corrupt_emitted_frame(self):
        factory = FakeFactory()
        source = MicSource(stream_factory=factory)
        received = []
        source.start(received.append)

        indata = np.full((512, 1), 1.0, dtype=np.float32)
        factory.streams[0].deliver(indata)
        indata[:] = -999.0  # PortAudio would reuse/overwrite this buffer

        assert np.all(received[0] == 1.0)

    def test_on_frame_exception_does_not_propagate_or_stop_capture(self, caplog):
        factory = FakeFactory()
        source = MicSource(stream_factory=factory)

        def blows_up(frame):
            raise ValueError("boom")

        source.start(blows_up)
        indata = np.zeros((512, 1), dtype=np.float32)
        with caplog.at_level(logging.ERROR, logger="vrcc.audio"):
            factory.streams[0].deliver(indata)  # must not raise
            indata2 = np.ones((512, 1), dtype=np.float32)
            factory.streams[0].deliver(indata2)  # second callback still runs fine

        # The failure IS logged (once, with traceback) -- not swallowed silently.
        error_records = [r for r in caplog.records if r.levelno >= logging.ERROR]
        assert len(error_records) == 1
        assert error_records[0].exc_info is not None

    def test_repeated_on_frame_exceptions_log_once_then_summarize_on_stop(self, caplog):
        # A persistently failing on_frame runs ~31x/sec on the audio
        # callback thread; it must log the full traceback exactly once,
        # count the rest, and emit a single summary line on stop().
        factory = FakeFactory()
        source = MicSource(stream_factory=factory)

        def blows_up(frame):
            raise ValueError("boom")

        source.start(blows_up)
        indata = np.zeros((512, 1), dtype=np.float32)
        with caplog.at_level(logging.WARNING, logger="vrcc.audio"):
            for _ in range(50):
                factory.streams[0].deliver(indata)
            error_records = [r for r in caplog.records if r.levelno >= logging.ERROR]
            assert len(error_records) == 1

            source.stop()

        summaries = [r for r in caplog.records if "suppressed" in r.getMessage()]
        assert len(summaries) == 1
        assert "49" in summaries[0].getMessage()

    def test_repeated_status_flags_log_once(self, caplog):
        factory = FakeFactory()
        source = MicSource(stream_factory=factory)
        source.start(lambda frame: None)

        indata = np.zeros((512, 1), dtype=np.float32)
        with caplog.at_level(logging.WARNING, logger="vrcc.audio"):
            for _ in range(10):
                factory.streams[0].deliver(indata, status="input overflow")

        status_records = [r for r in caplog.records if "status" in r.getMessage()]
        assert len(status_records) == 1

    def test_stop_stops_and_closes_stream(self):
        factory = FakeFactory()
        source = MicSource(stream_factory=factory)
        source.start(lambda frame: None)

        source.stop()

        assert factory.streams[0].stop_calls == 1
        assert factory.streams[0].close_calls == 1

    def test_stop_is_idempotent(self):
        factory = FakeFactory()
        source = MicSource(stream_factory=factory)
        source.start(lambda frame: None)

        source.stop()
        source.stop()  # must not raise

        assert factory.streams[0].stop_calls == 1
        assert factory.streams[0].close_calls == 1

    def test_stop_before_start_does_not_raise(self):
        source = MicSource(stream_factory=FakeFactory())
        source.stop()  # must not raise

    def test_double_start_stops_the_previous_stream_first(self):
        factory = FakeFactory()
        source = MicSource(stream_factory=factory)
        source.start(lambda frame: None)
        source.start(lambda frame: None)  # must not raise or leak

        assert len(factory.streams) == 2
        assert factory.streams[0].stop_calls == 1
        assert factory.streams[0].close_calls == 1
        assert factory.streams[1].start_calls == 1
        source.stop()
        assert factory.streams[1].stop_calls == 1
        assert factory.streams[1].close_calls == 1

    def test_restart_clears_stale_rechunker_remainder(self):
        # 300 samples buffered, then stop/start: the fresh session must not
        # inherit the old remainder (300 stale + 300 new = 600 would emit a
        # frame mixing audio from two different capture sessions).
        factory = FakeFactory()
        source = MicSource(stream_factory=factory)
        received = []
        source.start(received.append)
        factory.streams[0].deliver(np.zeros((300, 1), dtype=np.float32))
        source.stop()

        source.start(received.append)
        factory.streams[1].deliver(np.full((300, 1), 1.0, dtype=np.float32))

        assert received == []  # 300 < 512 against an empty remainder
