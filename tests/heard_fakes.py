"""Fakes and helpers for the HeardStream tests, shared by the two files
that drive it (behaviour and lifecycle), the way stt_fakes serves the STT
engine tests. Not a test module: no test_ functions live here.
"""

from __future__ import annotations

import threading
import time

import numpy as np

from vrcc.audio.segmenter import SegFinal, SegLevel
from vrcc.core.config import AppConfig
from vrcc.core.events import HeardPhrase
from vrcc.core.heard import HeardStream


class _Bus:
    def __init__(self):
        self.published = []

    def publish(self, event):
        self.published.append(event)


class _Source:
    """Hands frames to whatever the stream registers, on demand."""

    def __init__(self):
        self.on_frame = None
        self.started = False
        self.stopped = False

    def start(self, on_frame):
        self.on_frame = on_frame
        self.started = True

    def stop(self):
        self.stopped = True

    def feed(self, n=1):
        for _ in range(n):
            self.on_frame(np.zeros(512, dtype=np.float32))


class _Segmenter:
    """Emits one SegFinal per fed frame, so a test drives utterances directly."""

    def __init__(self, per_frame=1):
        self._per_frame = per_frame
        self._id = 0
        self.reset_calls = 0

    def reset(self):
        self.reset_calls += 1

    def process(self, frame):
        out = [SegLevel(rms=0.1, vad_prob=0.9)]
        for _ in range(self._per_frame):
            self._id += 1
            out.append(SegFinal(utterance_id=self._id, samples=np.zeros(16000, np.float32)))
        return out


class _Stt:
    """A whisper code for `language`, because that is what both real engines
    return. It used to hold the display name "Japanese", which no engine ever
    produces, and that single unrealistic value hid a defect that stopped every
    heard caption from being translated."""

    def __init__(self, text="konnichiwa", language="ja"):
        from vrcc.stt.engine import SttResult

        self._result = SttResult(
            text=text, language=language, avg_logprob=-0.2, no_speech_prob=0.01
        )
        self.calls = 0
        self.detect_language_calls = 0
        self.concurrent = 0
        self.max_concurrent = 0

    def transcribe(self, samples, detect_language=False):
        self.calls += 1
        if detect_language:
            self.detect_language_calls += 1
        self.concurrent += 1
        self.max_concurrent = max(self.max_concurrent, self.concurrent)
        time.sleep(0.01)
        self.concurrent -= 1
        return self._result


class _Mt:
    def __init__(self):
        self.calls = []

    def translate(self, text, src, targets):
        self.calls.append((text, src.display, [t.display for t in targets]))
        return [(t.display, f"{text} in {t.display}") for t in targets]


_DEFAULT = object()


def _stream(cfg=None, stt=None, mt=_DEFAULT, segmenter=None, source=None, locks=None):
    cfg = cfg or AppConfig()
    source = source or _Source()
    stt_lock, mt_lock = locks or (threading.Lock(), threading.Lock())
    stream = HeardStream(
        cfg, _Bus(), source, segmenter or _Segmenter(), stt or _Stt(),
        _Mt() if mt is _DEFAULT else mt, stt_lock, mt_lock,
    )
    return stream, source, stream._bus


def _phrases(bus):
    """Only the caption events. The stream also publishes a HeardLevel per
    frame for the speaker meter, which no test here is about."""
    return [e for e in bus.published if isinstance(e, HeardPhrase)]


def _wait(bus, count=1, seconds=1.0):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline and len(_phrases(bus)) < count:
        time.sleep(0.005)
    return _phrases(bus)
