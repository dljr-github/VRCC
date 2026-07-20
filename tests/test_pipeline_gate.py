"""Tests for :mod:`vrcc.core.pipeline_jobs.forward_final` -- the send-time
re-check of `_should_caption()`. `handle_final` only gates at enqueue time;
a result that finishes transcribing AFTER the user muted or turned
captioning off must still be caught before it reaches the chatbox.
Split out of `test_pipeline.py` to keep both files under the line cap.
"""

from __future__ import annotations

import threading

from vrcc.audio.segmenter import SegDiscard
from vrcc.core import pipeline_jobs
from vrcc.core.events import PhrasePartialCleared, PhraseRecognized
from vrcc.core.pipeline_jobs import _SttJob

from .conftest import FakeMute, collect, make_pipeline, make_result, sample


def _final_job(uid: int, s) -> _SttJob:
    return _SttJob(utterance_id=uid, samples=s, speculative=False, samples_id=id(s))


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


def test_forward_final_regated_by_captioning_off_clears_partial():
    # A live partial for this utterance may still be showing in the log with
    # no recognized/sent event coming to firm or remove it once gated here.
    env = make_pipeline(mt=None)
    cleared = collect(env.bus, PhrasePartialCleared)
    env.pipeline._begin_typing(1)
    env.pipeline.set_captioning(False)
    pipeline_jobs.forward_final(env.pipeline, 1, make_result())
    assert [e.utterance_id for e in cleared] == [1]


def test_handle_discard_terminal_publishes_partial_cleared():
    # A terminal discard (abort / too-short-finalize) ends the utterance: no
    # recognized/sent event is ever coming to firm or remove the row, so it
    # must be cleared here.
    env = make_pipeline()
    cleared = collect(env.bus, PhrasePartialCleared)
    pipeline_jobs.handle_discard(env.pipeline, SegDiscard(utterance_id=1, terminal=True))
    assert [e.utterance_id for e in cleared] == [1]


def test_handle_discard_non_terminal_keeps_the_partial_row():
    # A speech-resume discard leaves the SAME utterance active: more
    # SegPartials will follow and keep updating the row, so it must survive
    # (no PhrasePartialCleared), unlike the terminal case above.
    env = make_pipeline()
    cleared = collect(env.bus, PhrasePartialCleared)
    pipeline_jobs.handle_discard(env.pipeline, SegDiscard(utterance_id=1, terminal=False))
    assert cleared == []


def test_handle_discard_drops_cache_and_resolves_typing_regardless_of_terminal():
    # drop_discarded and _resolve_typing must run for every discard; only the
    # partial-row clearing is gated on the terminal flag.
    env = make_pipeline()
    env.pipeline._begin_typing(1)
    env.pipeline._spec.note_speculative(1, 99)
    pipeline_jobs.handle_discard(env.pipeline, SegDiscard(utterance_id=1, terminal=False))
    assert env.pipeline._spec._pending == {}
    assert env.pipeline._typing._in_flight == set()


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


def test_forward_final_quality_gated_none_clears_partial_and_resolves_typing():
    # A quality-gated (None) result sends nothing downstream, so a live partial
    # for this utterance would stay on the log with no recognized/sent event to
    # firm or remove it. The finalize step must clear it and resolve typing.
    env = make_pipeline(mt=None)
    cleared = collect(env.bus, PhrasePartialCleared)
    env.pipeline._begin_typing(1)
    pipeline_jobs.forward_final(env.pipeline, 1, None)
    assert [e.utterance_id for e in cleared] == [1]
    assert env.chatbox.typing[-1] is False


def test_final_no_engine_drop_clears_partial_and_resolves_typing():
    # The engine was swapped out while the final was in flight, so it drops
    # before forward_final. A live-partial row must still be cleared and typing
    # resolved, not left stuck listening.
    env = make_pipeline(mt=None)
    cleared = collect(env.bus, PhrasePartialCleared)
    env.pipeline._begin_typing(1)
    env.pipeline.set_stt(None)  # transcribe now returns _NO_ENGINE
    pipeline_jobs.process_stt_job(env.pipeline, _final_job(1, sample()), threading.Event())
    assert [e.utterance_id for e in cleared] == [1]
    assert env.chatbox.typing[-1] is False


def test_final_stop_set_drop_clears_partial_and_resolves_typing():
    # The run stopped mid-transcribe, so the final drops before forward_final.
    # Ids are monotonic across runs, so clearing here is safe even if a restart
    # already began; a leftover LISTENING row must not be left stuck.
    env = make_pipeline(mt=None)
    cleared = collect(env.bus, PhrasePartialCleared)
    env.pipeline._begin_typing(1)
    stop = threading.Event()
    stop.set()
    pipeline_jobs.process_stt_job(env.pipeline, _final_job(1, sample()), stop)
    assert [e.utterance_id for e in cleared] == [1]
    assert env.chatbox.typing[-1] is False
