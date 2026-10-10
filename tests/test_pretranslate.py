"""Tests for :mod:`vrcc.core.pretranslate` -- prefetching the MT call for a
speculative result so a final that reuses it skips the engine.

Jobs are driven synchronously (no worker threads): the STT and MT job
functions are called directly and the MT queue is drained by hand.
"""

from __future__ import annotations

import threading

from vrcc.audio.segmenter import SegDiscard
from vrcc.core import pretranslate
from vrcc.core.config import AppConfig, SttConfig, TranslateConfig
from vrcc.core.events import AppError, PhraseRecognized, PhraseTranslated
from vrcc.core.languages import get as get_lang
from vrcc.core.pipeline import JOB_QUEUE_MAX
from vrcc.core.pipeline_jobs import (
    _MtJob,
    _SttJob,
    handle_discard,
    process_mt_job,
    process_stt_job,
)

from .conftest import FakeMt, FakeStt, collect, make_pipeline, make_result, sample

ENGLISH = get_lang("English")


def _env(targets=("Japanese",), mt=None, stt=None):
    cfg = AppConfig(
        stt=SttConfig(source_language="auto"),
        translate=TranslateConfig(targets=list(targets)),
    )
    env = make_pipeline(config=cfg, stt=stt or FakeStt(), mt=mt or FakeMt())
    env.stop = threading.Event()
    return env


def _speculative(env, uid=1, samples=None):
    s = samples if samples is not None else sample()
    process_stt_job(env.pipeline, _SttJob(uid, s, True, id(s)), env.stop)
    return s


def _drain_mt(env):
    """Run every queued MT job the way the worker would."""
    q = env.pipeline._mt_queue
    while not q.empty():
        process_mt_job(env.pipeline, q.get_nowait(), env.stop)


def _final(env, uid, s):
    process_stt_job(env.pipeline, _SttJob(uid, s, False, id(s)), env.stop)
    _drain_mt(env)


def _events(env):
    return (
        collect(env.bus, PhraseRecognized),
        collect(env.bus, PhraseTranslated),
        collect(env.bus, AppError),
    )


def _real(uid, text):
    return _MtJob(uid, text, ENGLISH, manage_typing=False)


def test_hit_skips_engine_and_publishes_same_events_as_a_miss():
    env = _env()
    _recognized, translated, _errors = _events(env)
    s = _speculative(env)
    _drain_mt(env)
    assert len(env.mt.calls) == 1
    _final(env, 1, s)
    assert len(env.mt.calls) == 1
    assert env.pipeline._pretranslate.hit == 1

    plain = _env()
    _r, plain_translated, _e = _events(plain)
    _final(plain, 1, sample())
    assert len(plain.mt.calls) == 1
    assert plain_translated[0] == translated[0]
    assert plain.chatbox.submits[0] == env.chatbox.submits[0]


def test_changed_targets_is_a_miss():
    env = _env()
    s = _speculative(env)
    _drain_mt(env)
    env.config.translate.targets = ["Korean"]
    _final(env, 1, s)
    assert len(env.mt.calls) == 2
    assert env.mt.calls[1][2] == [get_lang("Korean")]
    assert env.pipeline._pretranslate.miss == 1


def test_real_job_without_an_entry_is_not_a_miss():
    env = _env()
    process_mt_job(env.pipeline, _real(1, "hello"), env.stop)
    assert len(env.mt.calls) == 1
    assert env.pipeline._pretranslate.miss == 0


def test_swapped_engine_is_a_miss():
    env = _env()
    s = _speculative(env)
    _drain_mt(env)
    other = FakeMt()
    env.pipeline.mt_slot.swap(other)
    _final(env, 1, s)
    assert len(other.calls) == 1
    assert env.pipeline._pretranslate.hit == 0


def test_different_final_text_is_a_miss():
    env = _env(stt=FakeStt(results=[make_result("one"), make_result("two")]))
    _speculative(env)
    _drain_mt(env)
    _final(env, 1, sample(n=256))
    assert [c[0] for c in env.mt.calls] == ["one", "two"]


def test_prefetch_publishes_nothing_and_leaves_typing_alone():
    env = _env()
    recognized, translated, errors = _events(env)
    _speculative(env)
    _drain_mt(env)
    assert env.chatbox.submits == []
    assert env.chatbox.typing == []
    assert (recognized, translated, errors) == ([], [], [])


def test_prefetch_engine_exception_is_silent():
    env = _env(mt=FakeMt(raises=RuntimeError("boom")))
    recognized, translated, errors = _events(env)
    _speculative(env)
    _drain_mt(env)
    assert env.chatbox.submits == []
    assert env.chatbox.typing == []
    assert (recognized, translated, errors) == ([], [], [])
    assert env.pipeline._pretranslate.dropped == 1


def test_prefetch_with_engine_removed_is_dropped_silently():
    env = _env()
    _recognized, translated, errors = _events(env)
    _speculative(env)
    env.pipeline.mt_slot.swap(None)
    _drain_mt(env)
    assert (translated, errors) == ([], [])
    assert env.chatbox.submits == []
    assert env.pipeline._pretranslate.dropped == 1


def test_no_offer_without_an_mt_engine_or_when_translate_is_off():
    env = _env()
    env.pipeline.mt_slot.swap(None)
    _speculative(env)
    assert env.pipeline._mt_queue.empty()

    off = _env()
    off.config.translate.enabled = False
    _speculative(off)
    assert off.pipeline._mt_queue.empty()


def test_prefetch_dropped_when_real_work_is_queued():
    env = _env()
    _speculative(env)
    q = env.pipeline._mt_queue
    prefetch = q.get_nowait()
    q.put_nowait(_real(9, "later"))
    process_mt_job(env.pipeline, prefetch, env.stop)
    assert env.mt.calls == []
    assert env.pipeline._pretranslate.dropped == 1


def test_offer_is_refused_unless_the_queue_is_empty():
    env = _env()
    q = env.pipeline._mt_queue
    for i in range(JOB_QUEUE_MAX - 1):
        q.put_nowait(_real(10 + i, f"real {i}"))
    _speculative(env)
    assert q.qsize() == JOB_QUEUE_MAX - 1  # typed text still has its slot
    assert all(isinstance(j, _MtJob) for j in list(q.queue))
    q.put_nowait(_real(20, "last"))
    _drain_mt(env)
    expected = [f"real {i}" for i in range(JOB_QUEUE_MAX - 1)] + ["last"]
    assert [c[0] for c in env.mt.calls] == expected


def test_second_offer_does_not_stack_behind_a_queued_prefetch():
    env = _env()
    _speculative(env, uid=1)
    _speculative(env, uid=1)
    assert env.pipeline._mt_queue.qsize() == 1
    assert env.pipeline._pretranslate.offered == 1


def test_stop_set_during_prefetch_writes_nothing():
    mt = FakeMt()
    env = _env(mt=mt)
    original = mt.translate

    def translate(text, src, targets):
        out = original(text, src, targets)
        env.stop.set()
        return out

    mt.translate = translate
    s = _speculative(env)
    process_mt_job(env.pipeline, env.pipeline._mt_queue.get_nowait(), env.stop)
    env.stop.clear()
    _final(env, 1, s)
    assert len(mt.calls) == 2
    assert env.pipeline._pretranslate.hit == 0


def test_cache_is_bounded_and_reset_on_restart():
    env = _env()
    for uid in range(1, pretranslate.CACHE_MAX + 3):
        _speculative(env, uid, sample(v=0.1 * uid))
        _drain_mt(env)
    cache = env.pipeline._pretranslate
    assert len(cache._entries) == pretranslate.CACHE_MAX
    assert 1 not in cache._entries
    env.pipeline.start()
    env.pipeline.stop()
    assert len(cache._entries) == 0


def test_finalize_ordering_does_not_evict_the_entry_before_the_real_job():
    env = _env()
    s = _speculative(env, uid=5)
    _drain_mt(env)
    # forward_final marks the utterance finalized right after enqueueing the
    # real job, before the MT worker has read the cache.
    process_stt_job(env.pipeline, _SttJob(5, s, False, id(s)), env.stop)
    assert 5 in env.pipeline._pretranslate._entries
    _drain_mt(env)
    assert len(env.mt.calls) == 1
    assert env.pipeline._pretranslate.hit == 1


def test_discard_drops_the_entry():
    env = _env()
    _speculative(env)
    _drain_mt(env)
    handle_discard(env.pipeline, SegDiscard(utterance_id=1))
    assert env.pipeline._pretranslate._entries == {}


def test_disabled_toggle_never_offers(monkeypatch):
    monkeypatch.setattr(pretranslate, "ENABLED", False)
    env = _env()
    s = _speculative(env)
    assert env.pipeline._mt_queue.empty()
    _final(env, 1, s)
    assert len(env.mt.calls) == 1
    assert env.pipeline._pretranslate.offered == 0
