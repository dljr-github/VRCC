"""HeardStream: captioning other people, and never speaking for them.

The load-bearing property is the one that is easiest to break later and hardest
to notice: what other people say must never leave the machine. The room already
heard them; relaying it to the chatbox would republish their words under the
user's name. Here that is checked structurally (no chatbox anywhere in reach)
as well as behaviourally.
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
    def __init__(self, text="konnichiwa", language="Japanese"):
        from vrcc.stt.engine import SttResult

        self._result = SttResult(
            text=text, language=language, avg_logprob=-0.2, no_speech_prob=0.01
        )
        self.calls = 0
        self.concurrent = 0
        self.max_concurrent = 0

    def transcribe(self, samples):
        self.calls += 1
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


# -- the boundary -------------------------------------------------------------


def test_the_stream_holds_nothing_that_could_reach_vrchat():
    """Structural, not behavioural: no chatbox, sender or OSC object is in
    reach, so relaying other people's words is not a branch someone can flip
    on by accident later."""
    stream, _source, _bus = _stream()

    reachable = " ".join(type(v).__name__.lower() for v in vars(stream).values())
    assert "chatbox" not in reachable
    assert "sender" not in reachable
    assert "osc" not in reachable
    assert not any("chatbox" in name or "osc" in name for name in vars(stream))


def test_a_heard_utterance_publishes_only_a_heard_phrase():
    stream, source, bus = _stream()
    try:
        stream.start()
        source.feed()
        published = _wait(bus)
    finally:
        stream.stop()

    assert published
    assert all(isinstance(e, HeardPhrase) for e in published), [
        type(e).__name__ for e in published
    ]


# -- translation direction ----------------------------------------------------


def test_translation_goes_into_the_language_the_user_reads():
    """Not translate.targets. Someone speaking Japanese at an English speaker
    must be shown English; the outbound targets would send it back to Japanese.
    """
    cfg = AppConfig()
    cfg.stt.spoken_languages = ["English"]
    cfg.translate.targets = ["Japanese"]
    mt = _Mt()
    stream, source, bus = _stream(cfg=cfg, mt=mt)
    try:
        stream.start()
        source.feed()
        _wait(bus)
    finally:
        stream.stop()

    assert mt.calls, "nothing was translated"
    _text, src, targets = mt.calls[0]
    assert src == "Japanese"
    assert targets == ["English"]


def test_speech_already_in_the_users_language_is_not_translated():
    cfg = AppConfig()
    cfg.stt.spoken_languages = ["Japanese"]
    mt = _Mt()
    stream, source, bus = _stream(cfg=cfg, mt=mt)
    try:
        stream.start()
        source.feed()
        _wait(bus)
    finally:
        stream.stop()

    assert mt.calls == [], "translated Japanese into Japanese"
    assert _phrases(bus)[0].translations == []


def test_no_translation_engine_still_publishes_the_transcript():
    """Reading what was said is useful even untranslated, and the engine can be
    absent or still loading."""
    stream, source, bus = _stream(mt=None)
    try:
        stream.start()
        source.feed()
        published = _wait(bus)
    finally:
        stream.stop()

    assert published[0].text
    assert published[0].translations == []


# -- sharing the engines ------------------------------------------------------


def test_transcription_is_serialised_against_the_main_pipeline():
    """The engines are shared rather than duplicated, so a second copy of a
    2.5 GB model is not loaded. That is only safe while both callers hold the
    same lock."""
    stt = _Stt()
    lock = threading.Lock()
    stream, source, bus = _stream(stt=stt, locks=(lock, threading.Lock()))
    try:
        stream.start()
        # The main pipeline is mid-decode: the heard worker must wait.
        with lock:
            source.feed()
            time.sleep(0.1)
            assert stt.calls == 0, "decoded while the other stream held the lock"
        _wait(bus)
    finally:
        stream.stop()

    assert stt.calls == 1
    assert stt.max_concurrent <= 1


# -- robustness ---------------------------------------------------------------


def test_a_backlog_drops_the_oldest_rather_than_growing():
    """A caption of what was said a minute ago is worse than no caption."""
    stream, source, bus = _stream(segmenter=_Segmenter(per_frame=40))
    try:
        stream.start()
        source.feed()
        time.sleep(0.2)
    finally:
        stream.stop()

    assert stream.dropped > 0


def test_a_failing_transcription_does_not_end_the_stream():
    class _Boom(_Stt):
        def transcribe(self, samples):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("decode failed")
            return self._result

    stt = _Boom()
    stream, source, bus = _stream(stt=stt)
    try:
        stream.start()
        source.feed()
        time.sleep(0.1)
        source.feed()
        published = _wait(bus)
    finally:
        stream.stop()

    assert published, "the stream died on the first failure"


def test_empty_or_blank_transcriptions_publish_nothing():
    stream, source, bus = _stream(stt=_Stt(text="   "))
    try:
        stream.start()
        source.feed()
        time.sleep(0.15)
    finally:
        stream.stop()

    assert _phrases(bus) == []


def test_start_and_stop_are_idempotent():
    stream, source, _bus = _stream()
    stream.stop()
    stream.start()
    stream.start()
    stream.stop()
    stream.stop()

    assert source.stopped


def test_stopping_leaves_no_worker_thread():
    stream, source, bus = _stream()
    stream.start()
    source.feed()
    _wait(bus)
    stream.stop()

    assert [t for t in threading.enumerate() if t.name == "HeardStream"] == []


# -- not captioning the user back to themselves -------------------------------


def test_speech_heard_while_the_microphone_was_live_is_dropped():
    """Loopback captures everything the output device plays, and on many setups
    that includes the user's own voice: hardware direct monitoring, headset
    sidetone, or a virtual mix output. Captioning that back reads as the
    feature being broken."""
    stt = _Stt()
    stream, source, bus = _stream(stt=stt)
    try:
        stream.start()
        stream.note_mic_level(0.2)  # the user is talking (loud on the mic)
        source.feed()
        time.sleep(0.15)
    finally:
        stream.stop()

    assert _phrases(bus) == []
    assert stream._suppressed == 1
    assert stt.calls == 0, "it should not even be transcribed"


def test_speech_heard_while_the_microphone_is_quiet_is_captioned():
    stream, source, bus = _stream()
    try:
        stream.start()
        stream.note_mic_level(0.0001)  # silence on the mic
        source.feed()
        published = _wait(bus)
    finally:
        stream.stop()

    assert published, "a quiet microphone must not suppress other people"


def test_the_guard_lapses_after_the_user_stops():
    """Otherwise one word from the user would mute everyone else for the rest
    of the session. The lookback is the utterance's own length plus the grace,
    since the check covers the whole capture window rather than its end."""
    from vrcc.core import heard as heard_mod

    stream, source, bus = _stream()
    try:
        stream.start()
        stream.note_mic_level(0.2)
        # One second of samples per fed frame, so rewind past window + grace.
        stream._user_spoke_at -= 1.0 + heard_mod._SELF_ECHO_GRACE_S + 0.5
        source.feed()
        published = _wait(bus)
    finally:
        stream.stop()

    assert published


def test_a_level_below_the_floor_is_not_treated_as_speech():
    """Room tone must not mute the room."""
    from vrcc.core import heard as heard_mod
    from vrcc.core.config import AppConfig

    cfg = AppConfig()
    stream, source, bus = _stream(cfg=cfg)
    try:
        stream.start()
        stream.note_mic_level(heard_mod._MIC_ACTIVE_RMS / 10)
        source.feed()
        published = _wait(bus)
    finally:
        stream.stop()

    assert published


def test_every_frame_feeds_the_speaker_meter():
    """The meter has to move on sound that never becomes an utterance, because
    seeing the level is how a user tells "wrong output device" (silent) from
    "nothing is being said" (moving)."""
    from vrcc.core.events import HeardLevel

    stream, source, bus = _stream()
    try:
        stream.start()
        source.feed(3)
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline and len(
            [e for e in bus.published if isinstance(e, HeardLevel)]
        ) < 3:
            time.sleep(0.005)
    finally:
        stream.stop()

    levels = [e for e in bus.published if isinstance(e, HeardLevel)]
    assert len(levels) >= 3


def test_the_meter_still_moves_while_the_echo_guard_suppresses_captions():
    """Otherwise a user whose own voice is in the speaker output would see a
    dead meter and conclude the capture was broken, when it is working and
    being suppressed on purpose."""
    from vrcc.core.events import HeardLevel

    stream, source, bus = _stream()
    try:
        stream.start()
        stream.note_mic_level(0.9)
        source.feed(2)
        time.sleep(0.15)
    finally:
        stream.stop()

    assert _phrases(bus) == [], "captions suppressed"
    assert [e for e in bus.published if isinstance(e, HeardLevel)], "but the meter lives"


def test_the_guard_arms_even_when_the_pipeline_zeroes_the_vad():
    """The defect that made the first version of this guard dead code:
    pipeline_frames publishes MicLevel(vad_prob=0.0) whenever captioning is off
    or mute sync is holding, and those are the states this feature is used in.
    RMS is real in every state, so either signal arms it."""
    stt = _Stt()
    stream, source, bus = _stream(stt=stt)
    try:
        stream.start()
        stream.note_mic_level(0.2, 0.0)  # loud mic, VAD zeroed by the gate
        source.feed()
        time.sleep(0.15)
    finally:
        stream.stop()

    assert _phrases(bus) == []
    assert stt.calls == 0


def test_an_utterance_that_merely_ENDS_after_you_stop_is_still_dropped():
    """The segmenter holds an utterance open for finalize_silence_ms, so the
    user's own speech and a fast reply arrive welded together. Judging only by
    when the merged utterance ended would pass their own voice through."""
    stream, source, bus = _stream()
    try:
        stream.start()
        stream.note_mic_level(0.2)
        # The utterance covers 1.0 s of audio and is dequeued right after the
        # user stopped: its WINDOW overlaps them even though its end does not.
        time.sleep(0.05)
        source.feed()
        time.sleep(0.15)
    finally:
        stream.stop()

    assert _phrases(bus) == []
    assert stream._suppressed == 1
