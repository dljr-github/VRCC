"""Shared pytest fixtures: the ``vrcc.core.pipeline`` fakes used by the
three split ``test_pipeline*`` modules (needed by 3+ files, so centralized
here instead of duplicated per the structure-and-style split rules).
"""

from __future__ import annotations

import contextlib
import threading
from types import SimpleNamespace

import numpy as np

from vrcc.core.bus import EventBus
from vrcc.core.config import AppConfig, OscConfig
from vrcc.core.pipeline import Pipeline
from vrcc.osc.chatbox import format_message
from vrcc.stt.engine import SttResult


# -- fakes -----------------------------------------------------------------


class FakeSource:
    def __init__(self) -> None:
        self.on_frame = None
        self.started = False
        self.stopped = False

    def start(self, on_frame) -> None:
        self.on_frame = on_frame
        self.started = True

    def stop(self) -> None:
        self.stopped = True


class FakeSegmenter:
    def __init__(self) -> None:
        self.frames: list = []
        self.active = False  # mirrors Segmenter.active (energy-gate contract)
        self.resets = 0

    def process(self, frame):
        self.frames.append(frame)
        return []

    def reset(self) -> None:
        # Pipeline.start() drops in-flight segmenter state on every run.
        self.resets += 1
        self.active = False


def make_result(
    text: str = "hello world",
    language: str = "en",
    avg_logprob: float = -0.2,
    no_speech_prob: float = 0.1,
) -> SttResult:
    return SttResult(
        text=text,
        language=language,
        avg_logprob=avg_logprob,
        no_speech_prob=no_speech_prob,
    )


_STT_DEFAULT = object()


class FakeStt:
    """Records transcribe() calls; returns scripted results.

    ``results`` (if given) is consumed one entry per call -- an ``Exception``
    entry is raised rather than returned. Otherwise ``result`` is returned for
    every call (defaulting to ``make_result()``; pass ``result=None`` to
    exercise the quality-gated path). If ``raises`` is set, every call raises
    it. The ``gate`` Event (set by default) lets a test hold a call inside
    ``transcribe`` to exercise the in-flight-discard path; ``entered`` fires
    when a call begins.
    """

    def __init__(self, result=_STT_DEFAULT, results=None, raises: Exception | None = None) -> None:
        self.calls = 0
        self.samples_seen: list = []
        self._result = make_result() if result is _STT_DEFAULT else result
        self._results = list(results) if results is not None else None
        self._raises = raises
        self._lock = threading.Lock()
        self.gate = threading.Event()
        self.gate.set()
        self.entered = threading.Event()

    def transcribe(self, samples):
        with self._lock:
            idx = self.calls
            self.calls += 1
            self.samples_seen.append(samples)
        self.entered.set()
        self.gate.wait()
        if self._raises is not None:
            raise self._raises
        if self._results is not None:
            item = self._results[idx]
            if isinstance(item, Exception):
                raise item
            return item
        return self._result


class FakeMt:
    def __init__(self, translations=None, raises: Exception | None = None) -> None:
        self.calls: list = []
        self._translations = translations
        self._raises = raises

    def translate(self, text, src, targets):
        self.calls.append((text, src, list(targets)))
        if self._raises is not None:
            raise self._raises
        if self._translations is not None:
            return list(self._translations)
        return [(t.display, f"{t.display}:{text}") for t in targets]


class FakeChatbox:
    """Records submits and typing changes, plus one ordered `log` of both
    (so tests can assert on their interleaving). ``fail_submits`` makes
    every submit() raise, to test chatbox-failure containment."""

    def __init__(self, fail_submits: bool = False) -> None:
        self.submits: list = []
        self.typing: list = []
        self.log: list = []
        self.fail_submits = fail_submits
        self._lock = threading.Lock()

    def submit(self, text, utterance_id) -> None:
        with self._lock:
            self.log.append(("submit", text, utterance_id))
            if self.fail_submits:
                raise RuntimeError("chatbox down")
            self.submits.append((text, utterance_id))

    def submit_message(self, original, translations, utterance_id) -> None:
        # Recorded as the default-config joined text so assertions read the
        # same display string the chatbox would show (the pipeline tests all
        # run with a default OscConfig).
        text = format_message(original, list(translations), OscConfig())
        self.submit(text, utterance_id)

    def set_typing(self, value) -> None:
        with self._lock:
            self.typing.append(bool(value))
            self.log.append(("typing", bool(value)))


class FakeMute:
    def __init__(self, caption: bool = True) -> None:
        self._caption = caption

    def should_caption(self) -> bool:
        return self._caption


# -- helpers ---------------------------------------------------------------


def collect(bus: EventBus, event_type: type) -> list:
    events: list = []
    bus.subscribe(event_type, events.append)
    return events


_UNSET = object()


def make_pipeline(
    *,
    config: AppConfig | None = None,
    stt=None,
    mt=_UNSET,
    chatbox=None,
    mute=None,
    source=None,
    segmenter=None,
    bus: EventBus | None = None,
    captioning: bool | None = True,
) -> SimpleNamespace:
    """``captioning`` opts the pipeline in by default so the STT/MT/gating
    tests below don't each have to enable it -- the production default is
    off (user opt-in per launch). Pass ``None`` to leave the raw default
    untouched (for tests that exercise that default itself)."""
    bus = bus or EventBus()
    config = config or AppConfig()
    source = source or FakeSource()
    segmenter = segmenter or FakeSegmenter()
    stt = stt or FakeStt()
    if mt is _UNSET:
        mt = FakeMt()
    chatbox = chatbox or FakeChatbox()
    pipeline = Pipeline(config, bus, source, segmenter, stt, mt, chatbox, mute)
    if captioning is not None:
        pipeline.set_captioning(captioning)
    return SimpleNamespace(
        pipeline=pipeline,
        bus=bus,
        config=config,
        source=source,
        segmenter=segmenter,
        stt=stt,
        mt=mt,
        chatbox=chatbox,
    )


@contextlib.contextmanager
def running(pipeline: Pipeline):
    pipeline.start()
    try:
        yield pipeline
    finally:
        pipeline.stop()


def sample(n: int = 512, v: float = 0.1) -> np.ndarray:
    return np.full(n, v, dtype=np.float32)
