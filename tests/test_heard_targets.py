"""Which language other people's speech is rendered into.

Split from ``test_heard_stream`` for the file-length cap. The question here is
only the choice of target, not the capture or the echo guard, so these share
that module's fakes rather than growing a second set.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import time

from tests.test_heard_stream import _Mt, _phrases, _Stt, _stream, _wait


def test_a_chosen_language_overrides_what_you_speak():
    """Someone captioning themselves in English may still want to read the room
    in Japanese. Without this the only answer was whatever they speak."""
    from vrcc.core.config import AppConfig

    cfg = AppConfig()
    cfg.stt.spoken_languages = ["English"]
    cfg.audio.hear_others_language = "Japanese"
    stream, _source, _bus = _stream(cfg=cfg)

    assert stream._heard_targets() == ["Japanese"]


def test_no_choice_still_follows_what_you_speak():
    """The default has to keep working with nothing set, which is what every
    existing config looks like after an update."""
    from vrcc.core.config import AppConfig

    cfg = AppConfig()
    cfg.stt.spoken_languages = ["English"]
    stream, _source, _bus = _stream(cfg=cfg)

    assert stream._heard_targets() == ["English"]


def test_their_own_language_is_never_a_translation_target():
    """Japanese speech shown to a Japanese reader needs no translation row."""
    from vrcc.core.config import AppConfig

    cfg = AppConfig()
    cfg.audio.hear_others_language = "Japanese"
    # The CODE an engine reports, not the display name. With the name here the
    # test was green for the wrong reason: _translate gives up at from_whisper
    # before the source-equals-target rule it claims to guard is ever reached.
    stt = _Stt(language="ja")
    mt = _Mt()
    stream, source, bus = _stream(cfg=cfg, stt=stt, mt=mt)
    try:
        stream.start()
        source.feed()
        time.sleep(0.15)
    finally:
        stream.stop()

    assert mt.calls == [], "asked to translate Japanese into Japanese"


def test_other_peoples_speech_is_never_decoded_as_your_language():
    """One engine serves both streams, and its language kwarg comes from the
    user's configured spoken language. Someone who tells VRCC they speak
    English would otherwise have every Japanese speaker in the room decoded as
    English, which yields confident nonsense rather than an error."""
    from vrcc.core.config import AppConfig

    cfg = AppConfig()
    cfg.stt.source_language = "English"
    stt = _Stt()
    stream, source, bus = _stream(cfg=cfg, stt=stt)
    try:
        stream.start()
        source.feed()
        time.sleep(0.2)
    finally:
        stream.stop()

    assert stt.calls == 1
    assert stt.detect_language_calls == 1, (
        "the heard stream must ask the engine to detect, not inherit the "
        "user's configured spoken language"
    )


def test_a_fabricated_english_tag_silently_drops_every_translation():
    """This is what Parakeet's detect_language path used to report for
    German speech: a made-up "en". With English the only language the user
    reads, the target list empties and _translate returns [] on every
    utterance while the transcript still publishes, which reads as
    translation being slow rather than as a failure."""
    from vrcc.core.config import AppConfig

    cfg = AppConfig()
    cfg.stt.spoken_languages = ["English"]
    cfg.translate.enabled = True
    mt = _Mt()
    stream, source, bus = _stream(
        cfg=cfg, stt=_Stt(text="wie geht es dir", language="en"), mt=mt
    )
    try:
        stream.start()
        source.feed()
        _wait(bus)
    finally:
        stream.stop()

    phrases = _phrases(bus)
    assert mt.calls == []
    assert len(phrases) == 1
    assert phrases[0].translations == []


def test_an_honest_german_tag_translates_normally():
    """The positive control: the identical utterance, correctly labelled,
    must reach the translator. Nothing here changes when the source is
    honest; the defect is entirely upstream, in what the engine reports."""
    from vrcc.core.config import AppConfig

    cfg = AppConfig()
    cfg.stt.spoken_languages = ["English"]
    cfg.translate.enabled = True
    mt = _Mt()
    stream, source, bus = _stream(
        cfg=cfg, stt=_Stt(text="wie geht es dir", language="de"), mt=mt
    )
    try:
        stream.start()
        source.feed()
        _wait(bus)
    finally:
        stream.stop()

    assert mt.calls == [("wie geht es dir", "German", ["English"])]


def test_a_none_language_publishes_untranslated_without_crashing():
    """What the fixed OnnxAsrEngine actually reports for detect_language=True:
    no fabricated code at all. HeardPhrase and _translate must handle it as
    cleanly as any other language they have no registry entry for -- the
    transcript still reaches the feed, just without a translation."""
    from vrcc.core.config import AppConfig

    cfg = AppConfig()
    cfg.stt.spoken_languages = ["English"]
    cfg.translate.enabled = True
    mt = _Mt()
    stream, source, bus = _stream(
        cfg=cfg, stt=_Stt(text="wie geht es dir", language=None), mt=mt
    )
    try:
        stream.start()
        source.feed()
        _wait(bus)
    finally:
        stream.stop()

    phrases = _phrases(bus)
    assert mt.calls == []
    assert len(phrases) == 1
    assert phrases[0].translations == []


def test_a_detected_language_code_still_resolves_a_translation_source():
    """Engines report what they detected as a code ("ja"); the language
    registry keys on display names ("Japanese"). Feeding one to the other
    looked like "unknown language, skip translation" and silently produced a
    transcription with no translation for every single utterance."""
    from vrcc.core.config import AppConfig

    cfg = AppConfig()
    cfg.stt.spoken_languages = ["English"]
    mt = _Mt()
    stream, source, bus = _stream(cfg=cfg, stt=_Stt(language="ja"), mt=mt)
    try:
        stream.start()
        source.feed()
        _wait(bus, 1)
    finally:
        stream.stop()

    assert mt.calls, "a detected code must resolve to a translation source"
    _text, src, targets = mt.calls[0]
    assert src == "Japanese"
    assert targets == ["English"]
    assert _phrases(bus)[0].translations == [("English", "konnichiwa in English")]
