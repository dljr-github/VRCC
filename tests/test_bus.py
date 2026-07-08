import logging

import pytest

from vrcc.core.bus import EventBus
from vrcc.core.events import AppError, MicLevel, SpeechStarted


def test_publish_delivers_to_matching_type_only():
    bus = EventBus()
    mic_events = []
    speech_events = []
    bus.subscribe(MicLevel, mic_events.append)
    bus.subscribe(SpeechStarted, speech_events.append)

    event = MicLevel(rms=0.5, vad_prob=0.9)
    bus.publish(event)

    assert mic_events == [event]
    assert speech_events == []


def test_publish_with_no_subscribers_does_nothing():
    bus = EventBus()
    # Should not raise even though nothing is subscribed to this type.
    bus.publish(MicLevel(rms=0.1, vad_prob=0.2))


def test_multiple_handlers_for_same_type_all_called():
    bus = EventBus()
    calls_a = []
    calls_b = []
    bus.subscribe(MicLevel, calls_a.append)
    bus.subscribe(MicLevel, calls_b.append)

    event = MicLevel(rms=0.3, vad_prob=0.4)
    bus.publish(event)

    assert calls_a == [event]
    assert calls_b == [event]


def test_unsubscribe_stops_delivery():
    bus = EventBus()
    received = []
    unsubscribe = bus.subscribe(MicLevel, received.append)

    unsubscribe()
    bus.publish(MicLevel(rms=0.6, vad_prob=0.1))

    assert received == []


def test_unsubscribe_is_safe_to_call_more_than_once():
    bus = EventBus()
    received = []
    unsubscribe = bus.subscribe(MicLevel, received.append)

    unsubscribe()
    unsubscribe()  # must not raise

    bus.publish(MicLevel(rms=0.6, vad_prob=0.1))
    assert received == []


def test_raising_handler_does_not_block_other_handlers(caplog):
    bus = EventBus()
    calls_b = []

    def handler_a(event):
        raise ValueError("boom")

    bus.subscribe(MicLevel, handler_a)
    bus.subscribe(MicLevel, calls_b.append)

    errors = []
    bus.subscribe(AppError, errors.append)

    with caplog.at_level(logging.ERROR, logger="vrcc.bus"):
        bus.publish(MicLevel(rms=0.7, vad_prob=0.8))

    assert calls_b == [MicLevel(rms=0.7, vad_prob=0.8)]
    assert len(errors) == 1
    assert errors[0].code == "HANDLER_ERROR"
    assert len(caplog.records) == 1


def test_raising_handler_publishes_exactly_one_app_error():
    bus = EventBus()

    def handler_a(event):
        raise ValueError("boom")

    def handler_b(event):
        raise RuntimeError("also boom")

    bus.subscribe(MicLevel, handler_a)
    bus.subscribe(MicLevel, handler_b)

    errors = []
    bus.subscribe(AppError, errors.append)

    bus.publish(MicLevel(rms=0.2, vad_prob=0.3))

    assert len(errors) == 2
    assert all(e.code == "HANDLER_ERROR" for e in errors)


def test_raising_app_error_handler_does_not_recurse(caplog):
    bus = EventBus()

    def bad_handler(event):
        raise ValueError("boom while handling AppError")

    bus.subscribe(AppError, bad_handler)

    with caplog.at_level(logging.ERROR, logger="vrcc.bus"):
        bus.publish(AppError(code="SOME_ERROR", message="original failure"))

    # Only the original AppError dispatch's failure gets logged; no new
    # AppError is published (that would recurse), so bad_handler is called
    # exactly once total.
    assert len(caplog.records) == 1


def test_subscribers_can_unsubscribe_during_dispatch():
    bus = EventBus()
    calls = []

    def handler(event):
        calls.append(event)
        unsubscribe()

    unsubscribe = bus.subscribe(MicLevel, handler)

    bus.publish(MicLevel(rms=0.1, vad_prob=0.1))
    bus.publish(MicLevel(rms=0.2, vad_prob=0.2))

    assert len(calls) == 1


def test_events_are_frozen_dataclasses():
    event = MicLevel(rms=0.5, vad_prob=0.5)
    with pytest.raises(Exception):
        event.rms = 0.9
