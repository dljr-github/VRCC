"""LoopbackSource: capturing what the speakers play, without real audio.

The recorder is injected, so these run on a CI box with no sound card and no
soundcard package. The one thing they cannot cover is the COM apartment dance
(soundcard initialises COM in its module body and rejects S_FALSE, so the
capture thread has to import BEFORE joining an apartment), which needs a real
Windows audio stack; that is verified by hand and recorded in the module
docstring.
"""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from vrcc.audio.loopback import LoopbackSource
from vrcc.audio.source import FRAME_LEN


class _FakeRecorder:
    """Stands in for soundcard's recorder: a context manager whose
    ``record(numframes)`` returns a (numframes, 1) float32 block."""

    def __init__(self, blocks=None, raises=None):
        self._blocks = blocks
        self._raises = raises
        self.entered = False
        self.exited = False
        self.reads = 0

    def __enter__(self):
        self.entered = True
        return self

    def __exit__(self, *exc):
        self.exited = True
        return False

    def record(self, numframes):
        self.reads += 1
        if self._raises is not None:
            raise self._raises
        if self._blocks is not None:
            return self._blocks[(self.reads - 1) % len(self._blocks)]
        # A quiet but non-constant block, so a test asserting on content is not
        # fooled by an all-zero array.
        return (np.arange(numframes, dtype=np.float32).reshape(-1, 1) % 7) / 100.0


def _drain(source, frames, seconds=0.4):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline and not frames:
        time.sleep(0.01)
    source.stop()


def test_frames_arrive_as_the_pipeline_expects():
    """512 samples, 1-D, float32. The segmenter and VAD take nothing else, and
    soundcard hands back a 2-D (frames, channels) block."""
    frames: list[np.ndarray] = []
    source = LoopbackSource(recorder_factory=lambda device: _FakeRecorder())

    source.start(frames.append)
    _drain(source, frames)

    assert frames, "no frames were produced"
    for frame in frames[:5]:
        assert frame.shape == (FRAME_LEN,)
        assert frame.dtype == np.float32
        assert frame.ndim == 1


def test_stop_ends_the_capture_thread():
    frames: list[np.ndarray] = []
    source = LoopbackSource(recorder_factory=lambda device: _FakeRecorder())

    source.start(frames.append)
    _drain(source, frames)

    assert source._thread is None
    live = [t for t in threading.enumerate() if t.name == "LoopbackSource"]
    assert live == [], "the capture thread outlived stop()"


def test_stop_is_safe_before_start_and_twice():
    source = LoopbackSource(recorder_factory=lambda device: _FakeRecorder())
    source.stop()
    source.start(lambda _f: None)
    source.stop()
    source.stop()


def test_a_failed_open_gives_up_quietly():
    """A device that cannot be opened must not raise into the caller: capturing
    the speakers is an optional extra, and losing it cannot take the microphone
    down with it."""
    def boom(device):
        raise RuntimeError("no such device")

    source = LoopbackSource(recorder_factory=boom)
    source.start(lambda _f: None)
    source.stop()

    assert source._thread is None


def test_repeated_read_failures_stop_rather_than_spin():
    """A device that disappears mid-session (headset unplugged) would otherwise
    log a failure per read forever."""
    recorder = _FakeRecorder(raises=OSError("device went away"))
    source = LoopbackSource(recorder_factory=lambda device: recorder)

    source.start(lambda _f: None)
    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline and source._thread.is_alive():
        time.sleep(0.01)
    alive = source._thread is not None and source._thread.is_alive()
    source.stop()

    assert not alive, "the capture thread kept spinning on a dead device"
    assert source.read_errors >= 1


def test_one_bad_consumer_frame_does_not_end_capture():
    """Matches MicSource's contract: an exception from the consumer is logged
    and capture continues, because dropping the rest of the session over one
    frame is worse than dropping the frame."""
    seen: list[int] = []

    def on_frame(_frame):
        seen.append(1)
        if len(seen) == 1:
            raise ValueError("boom")

    source = LoopbackSource(recorder_factory=lambda device: _FakeRecorder())
    source.start(on_frame)
    deadline = time.monotonic() + 0.5
    while time.monotonic() < deadline and len(seen) < 3:
        time.sleep(0.01)
    source.stop()

    assert len(seen) >= 3, "capture stopped after the consumer raised"


def test_starting_twice_replaces_the_first_capture():
    recorders: list[_FakeRecorder] = []

    def factory(device):
        recorders.append(_FakeRecorder())
        return recorders[-1]

    frames: list[np.ndarray] = []
    source = LoopbackSource(recorder_factory=factory)
    source.start(frames.append)
    while not frames:
        time.sleep(0.01)
    source.start(frames.append)
    source.stop()

    assert len(recorders) == 2
    assert recorders[0].exited, "the first recorder was left open"


@pytest.mark.parametrize("blocksize", [FRAME_LEN // 2, FRAME_LEN, FRAME_LEN * 3 + 7])
def test_any_block_size_is_rechunked_to_whole_frames(blocksize):
    """soundcard is not obliged to return exactly what was asked for, and a
    partial frame carried into the VAD would shift every window after it."""
    block = np.zeros((blocksize, 1), dtype=np.float32)
    frames: list[np.ndarray] = []
    source = LoopbackSource(
        recorder_factory=lambda device: _FakeRecorder(blocks=[block])
    )

    source.start(frames.append)
    _drain(source, frames, seconds=0.6)

    assert frames
    assert all(f.shape == (FRAME_LEN,) for f in frames)


def test_a_missing_soundcard_is_reported_not_swallowed():
    """The defect this feature shipped with. The import happens on the capture
    thread, so an ImportError can never reach a try/except around start(): the
    thread logged a warning and returned, and the stream sat there receiving
    nothing for the life of the process. Five sessions of logs looked like
    this."""
    import sys

    reported = []
    src = LoopbackSource(on_failure=lambda code, detail: reported.append(code))
    saved = sys.modules.get("soundcard", "absent")
    sys.modules["soundcard"] = None  # makes `import soundcard` raise
    try:
        src.start(lambda frame: None)
        deadline = time.monotonic() + 2.0
        while not reported and time.monotonic() < deadline:
            time.sleep(0.01)
    finally:
        src.stop()
        if saved == "absent":
            sys.modules.pop("soundcard", None)
        else:
            sys.modules["soundcard"] = saved

    assert reported == ["HEARD_NO_LIBRARY"]


def test_a_device_that_will_not_open_is_reported():
    """The other way this dies silently: the library is there but the chosen
    output cannot be opened for loopback."""
    reported = []

    def refuse(_device):
        raise OSError("device is in exclusive use")

    src = LoopbackSource(
        recorder_factory=refuse,
        on_failure=lambda code, detail: reported.append((code, detail)),
    )
    try:
        src.start(lambda frame: None)
        deadline = time.monotonic() + 2.0
        while not reported and time.monotonic() < deadline:
            time.sleep(0.01)
    finally:
        src.stop()

    assert reported and reported[0][0] == "HEARD_DEVICE_FAILED"
    # The detail carries the cause to the log. It was an undefined name here
    # once, which turned the report itself into a second silent failure.
    assert "exclusive use" in reported[0][1]


def test_every_reported_code_has_a_sentence_a_user_can_act_on():
    """A code with no entry falls through to the generic handler message,
    which would throw away the one thing worth saying: which failure it was."""
    from vrcc.gui.icons import FRIENDLY_ERRORS

    for code in ("HEARD_NO_LIBRARY", "HEARD_DEVICE_FAILED"):
        assert code in FRIENDLY_ERRORS, code


def test_a_recorder_that_fails_to_start_is_reported():
    """soundcard's recorder() only builds the object: __enter__ is what starts
    the WASAPI stream, so exclusive-mode contention and an endpoint that
    refuses 16 kHz mono raise there. Outside the try that killed the capture
    thread with nothing on screen."""
    class _RefusingRecorder(_FakeRecorder):
        def __enter__(self):
            raise OSError("could not start the stream")

    reported = []
    src = LoopbackSource(
        recorder_factory=lambda _device: _RefusingRecorder(),
        on_failure=lambda code, detail: reported.append((code, detail)),
    )
    try:
        src.start(lambda _frame: None)
        deadline = time.monotonic() + 2.0
        while not reported and time.monotonic() < deadline:
            time.sleep(0.01)
    finally:
        src.stop()

    assert reported and reported[0][0] == "HEARD_DEVICE_FAILED"
    assert "could not start the stream" in reported[0][1]


def test_giving_up_on_a_dead_device_is_reported():
    """Every way this thread ends without the caller asking has to say so, or
    the toggle sits lit beside a stream that is not running."""
    reported = []
    src = LoopbackSource(
        recorder_factory=lambda _device: _FakeRecorder(raises=OSError("gone")),
        on_failure=lambda code, detail: reported.append((code, detail)),
    )
    try:
        src.start(lambda _frame: None)
        deadline = time.monotonic() + 2.0
        while not reported and time.monotonic() < deadline:
            time.sleep(0.01)
    finally:
        src.stop()

    assert reported and reported[0][0] == "HEARD_DEVICE_FAILED"
