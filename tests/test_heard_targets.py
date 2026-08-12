"""Which language other people's speech is rendered into.

Split from ``test_heard_stream`` for the file-length cap. The question here is
only the choice of target, not the capture or the echo guard, so these share
that module's fakes rather than growing a second set.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import time

from tests.test_heard_stream import _Mt, _Stt, _stream


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
    stt = _Stt(language="Japanese")
    mt = _Mt()
    stream, source, bus = _stream(cfg=cfg, stt=stt, mt=mt)
    try:
        stream.start()
        source.feed()
        time.sleep(0.15)
    finally:
        stream.stop()

    assert mt.calls == [], "asked to translate Japanese into Japanese"
