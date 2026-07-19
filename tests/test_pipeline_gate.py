"""Tests for :mod:`vrcc.core.pipeline_jobs.forward_final` -- the send-time
re-check of `_should_caption()`. `handle_final` only gates at enqueue time;
a result that finishes transcribing AFTER the user muted or turned
captioning off must still be caught before it reaches the chatbox.
Split out of `test_pipeline.py` to keep both files under the line cap.
"""

from __future__ import annotations

from vrcc.core import pipeline_jobs
from vrcc.core.events import PhraseRecognized

from .conftest import FakeMute, collect, make_pipeline, make_result


def test_forward_final_regated_by_captioning_off_does_not_send():
    env = make_pipeline(mt=None)
    recognized = collect(env.bus, PhraseRecognized)
    env.pipeline._begin_typing(1)
    env.pipeline.set_captioning(False)  # gate closes between enqueue and send
    pipeline_jobs.forward_final(env.pipeline, 1, make_result())
    assert recognized == []
    assert env.chatbox.submits == []
    assert env.chatbox.typing[-1] is False  # typing indicator still resolves
    assert env.pipeline._spec._last_finalized >= 1  # still bounds the caches


def test_forward_final_regated_by_mute_does_not_send():
    mute = FakeMute(caption=True)
    env = make_pipeline(mt=None, mute=mute)
    recognized = collect(env.bus, PhraseRecognized)
    env.pipeline._begin_typing(1)
    mute.caption = False  # user muted between enqueue and send
    pipeline_jobs.forward_final(env.pipeline, 1, make_result())
    assert recognized == []
    assert env.chatbox.submits == []
    assert env.chatbox.typing[-1] is False
