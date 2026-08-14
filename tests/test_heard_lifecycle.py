"""HeardStream lifecycle: the shared engines it does not own, the queue it
starts each run with, and the microphone reading its echo guard applies.

Split from test_heard_stream for the source cap. That file is about what the
stream publishes; this one is about what happens to it from outside, which is
where every defect in it so far has lived: an engine swapped underneath it, a
restart, and a reply arriving while an utterance was still decoding.
"""

from __future__ import annotations

import threading
import time

import numpy as np

from vrcc.core.config import AppConfig

from .heard_fakes import _Mt, _Stt, _phrases, _stream, _wait


# -- the engines are shared, and shared engines get swapped --------------------


def test_a_swapped_voice_model_reaches_this_stream_too():
    """Only the pipeline is told when an engine is hot-swapped. Holding the
    object handed over at build time meant the reloader unloaded it under this
    stream, every decode raised into _work's handler, and the feature went
    silent for the session with its toggle still lit."""
    old, new = _Stt(text="old"), _Stt(text="new")
    stream, source, bus = _stream(stt=old)
    try:
        stream.start()
        stream.set_stt(new)
        source.feed()
        _wait(bus)
    finally:
        stream.stop()

    assert new.calls == 1
    assert old.calls == 0, "decoded on the engine the swap unloaded"


def test_a_detached_voice_model_drops_the_utterance_rather_than_raising():
    """The swap sets it to None between detach and install; an utterance
    landing in that window has no engine to run on."""
    stream, source, bus = _stream()
    try:
        stream.start()
        stream.set_stt(None)
        source.feed()
        time.sleep(0.15)
    finally:
        stream.stop()

    assert _phrases(bus) == []


def test_translation_switched_on_after_launch_reaches_this_stream():
    """Built with mt=None (translation off at launch), the stream would return
    no translations for the rest of the session however many models the
    pipeline installed afterwards."""
    cfg = AppConfig()
    cfg.translate.enabled = True
    cfg.audio.hear_others_language = "English"
    mt = _Mt()
    stream, source, bus = _stream(cfg=cfg, mt=None)
    try:
        stream.start()
        stream.set_mt(mt)
        source.feed()
        _wait(bus)
    finally:
        stream.stop()

    assert mt.calls, "the stream kept the None it was built with"


def test_your_reply_does_not_delete_the_line_you_are_replying_to():
    """The guard asks what the microphone was doing WHILE the utterance was
    captured. Read at dequeue instead, normal turn-taking (they finish, you
    answer while their line is still decoding) suppressed the other person
    every time, and the drop showed up nowhere but a debug log."""
    release = threading.Event()

    class _Slow(_Stt):
        def transcribe(self, samples, detect_language=False):
            if self.calls == 0:
                self.calls += 1
                release.wait(2.0)
                return self._result
            return super().transcribe(samples, detect_language)

    stt = _Slow()
    stream, source, bus = _stream(stt=stt)
    try:
        stream.start()
        source.feed()  # they speak; the worker settles into the decode
        deadline = time.monotonic() + 1.0
        while stt.calls == 0 and time.monotonic() < deadline:
            time.sleep(0.005)
        source.feed()  # they speak again, captured with a quiet microphone
        stream.note_mic_level(0.9)  # you answer, after that was captured
        release.set()
        published = _wait(bus, count=2)
    finally:
        release.set()
        stream.stop()

    assert len(published) == 2, "your reply suppressed their caption"
    assert stream._suppressed == 0


def test_switching_off_and_on_starts_from_an_empty_queue():
    """What was queued before the user switched this off is stale by the time
    they switch it back on, and captioning it then reads as speech that just
    happened. It also leaves a worker that outlived stop()'s join holding a
    queue nobody feeds, rather than racing its replacement for this one."""
    stream, source, bus = _stream()
    stream.start()
    first = stream._queue
    stream.stop()
    first.put_nowait((np.zeros(16000, np.float32), time.monotonic(), 0.0))

    stream.start()
    try:
        assert stream._queue is not first, "the restart reused the old queue"
        assert stream._queue.qsize() == 0
    finally:
        stream.stop()
