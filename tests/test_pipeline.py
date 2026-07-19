"""Tests for :mod:`vrcc.core.pipeline` -- caching/dedup, gating, typing
indicator, send-to-vrchat, and worker-exception behavior.
"""

from __future__ import annotations

import threading
import time

import pytest

from vrcc.audio.segmenter import (
    SegDiscard,
    SegFinal,
    SegLevel,
    SegSpeculative,
    SegSpeechStart,
)
from vrcc.core import pipeline_jobs
from vrcc.core.config import AppConfig, OscConfig
from vrcc.core.events import AppError, MicLevel, PhraseRecognized, PhraseTranslated, SpeechStarted
from vrcc.core.pipeline_jobs import _SttJob

from .conftest import FakeChatbox, FakeMt, FakeMute, FakeStt, collect, make_pipeline, make_result, running, sample


def _wait_until(predicate, timeout: float = 2.0, interval: float = 0.005) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return bool(predicate())


# -- SegLevel / SegSpeechStart passthrough ---------------------------------


def test_seglevel_publishes_miclevel():
    env = make_pipeline()
    levels = collect(env.bus, MicLevel)
    with running(env.pipeline):
        env.pipeline._on_seg_event(SegLevel(rms=0.3, vad_prob=0.7))
    assert len(levels) == 1
    assert levels[0].rms == pytest.approx(0.3)
    assert levels[0].vad_prob == pytest.approx(0.7)


def test_segspeechstart_publishes_speechstarted():
    env = make_pipeline()
    started = collect(env.bus, SpeechStarted)
    with running(env.pipeline):
        env.pipeline._on_seg_event(SegSpeechStart(utterance_id=7))
    assert [e.utterance_id for e in started] == [7]


# -- behavior 1: speculative -> STT job, cached, not forwarded --------------


def test_speculative_enqueues_stt_caches_result_and_does_not_forward():
    env = make_pipeline()
    recognized = collect(env.bus, PhraseRecognized)
    s = sample()
    with running(env.pipeline):
        env.pipeline._on_seg_event(SegSpeculative(utterance_id=1, samples=s))
        key = (1, id(s))
        assert _wait_until(lambda: key in env.pipeline._spec._cache)
        # cached but NOT forwarded
        time.sleep(0.02)
        assert env.stt.calls == 1
        assert recognized == []
        assert env.chatbox.submits == []
        assert env.pipeline._spec._cache[key] == make_result()


# -- behavior 2: final reuses cached speculative (identity) -----------------


def test_final_reuses_cached_speculative_without_second_stt_call():
    env = make_pipeline()
    recognized = collect(env.bus, PhraseRecognized)
    s = sample()
    with running(env.pipeline):
        env.pipeline._on_seg_event(SegSpeculative(utterance_id=1, samples=s))
        env.pipeline._on_seg_event(SegFinal(utterance_id=1, samples=s))
        assert _wait_until(lambda: env.pipeline._spec._last_finalized >= 1)
        assert _wait_until(lambda: len(recognized) == 1)
    # exactly one transcribe: the speculative; the final reused the cache
    assert env.stt.calls == 1
    assert recognized[0].text == "hello world"


def test_final_with_different_samples_triggers_fresh_stt_call():
    env = make_pipeline()
    recognized = collect(env.bus, PhraseRecognized)
    s_spec = sample()
    s_final = sample()  # different object -> id differs -> no cache hit
    with running(env.pipeline):
        env.pipeline._on_seg_event(SegSpeculative(utterance_id=1, samples=s_spec))
        env.pipeline._on_seg_event(SegFinal(utterance_id=1, samples=s_final))
        assert _wait_until(lambda: env.pipeline._spec._last_finalized >= 1)
        assert _wait_until(lambda: len(recognized) == 1)
    assert env.stt.calls == 2


# -- behavior 3: discard drops the cached result ----------------------------


def test_discard_after_cache_drops_cached_result():
    env = make_pipeline()
    s = sample()
    with running(env.pipeline):
        env.pipeline._on_seg_event(SegSpeculative(utterance_id=1, samples=s))
        key = (1, id(s))
        assert _wait_until(lambda: key in env.pipeline._spec._cache)
        env.pipeline._on_seg_event(SegDiscard(utterance_id=1))
        assert key not in env.pipeline._spec._cache


def test_inflight_speculative_result_discarded_is_thrown_away():
    env = make_pipeline(stt=FakeStt())
    recognized = collect(env.bus, PhraseRecognized)
    env.stt.gate.clear()  # block inside transcribe
    s = sample()
    with running(env.pipeline):
        env.pipeline._on_seg_event(SegSpeculative(utterance_id=1, samples=s))
        assert env.stt.entered.wait(2.0)  # worker is inside transcribe
        env.pipeline._on_seg_event(SegDiscard(utterance_id=1))  # mark stale
        env.stt.gate.set()  # let transcribe finish
        # A later, good final for a different utterance flushes the FIFO.
        good = sample()
        env.pipeline._on_seg_event(SegFinal(utterance_id=2, samples=good))
        assert _wait_until(lambda: env.pipeline._spec._last_finalized >= 2)
    # the discarded speculative never cached and never published anything
    assert (1, id(s)) not in env.pipeline._spec._cache
    assert [e.utterance_id for e in recognized] == [2]


# -- behavior 4: gated (None) STT result -> nothing downstream --------------


def test_none_stt_result_produces_nothing_downstream():
    env = make_pipeline(stt=FakeStt(result=None))
    recognized = collect(env.bus, PhraseRecognized)
    translated = collect(env.bus, PhraseTranslated)
    with running(env.pipeline):
        env.pipeline._on_seg_event(SegFinal(utterance_id=1, samples=sample()))
        assert _wait_until(lambda: env.pipeline._spec._last_finalized >= 1)
    assert recognized == []
    assert translated == []
    assert env.chatbox.submits == []


# -- behavior 5: recognized -> translate -> chatbox -------------------------


def test_final_publishes_recognized_translates_and_submits_formatted():
    env = make_pipeline()
    recognized = collect(env.bus, PhraseRecognized)
    translated = collect(env.bus, PhraseTranslated)
    with running(env.pipeline):
        env.pipeline._on_seg_event(SegFinal(utterance_id=3, samples=sample()))
        assert _wait_until(lambda: len(env.chatbox.submits) == 1)
        assert _wait_until(lambda: len(translated) == 1)
    assert [e.text for e in recognized] == ["hello world"]
    tr = translated[0]
    assert tr.utterance_id == 3
    assert tr.original == "hello world"
    assert tr.translations == (("Japanese", "Japanese:hello world"),)
    # format_message: include_original + "\n" separator (defaults)
    text, uid = env.chatbox.submits[0]
    assert uid == 3
    assert text == "hello world\nJapanese:hello world"


def test_translation_disabled_sends_original_directly():
    env = make_pipeline(mt=None)
    recognized = collect(env.bus, PhraseRecognized)
    translated = collect(env.bus, PhraseTranslated)
    with running(env.pipeline):
        env.pipeline._on_seg_event(SegFinal(utterance_id=1, samples=sample()))
        assert _wait_until(lambda: len(env.chatbox.submits) == 1)
    assert [e.text for e in recognized] == ["hello world"]
    assert translated == []  # no MT -> no PhraseTranslated
    assert env.chatbox.submits[0] == ("hello world", 1)


# Source resolution and MT target selection (skipping a target equal to the
# resolved source, hidden-original delivery) live in test_pipeline_targets.


# -- behavior 6: mute + master-toggle gating --------------------------------


def test_mute_gating_updates_meter_but_skips_stt():
    env = make_pipeline(mute=FakeMute(caption=False))
    levels = collect(env.bus, MicLevel)
    with running(env.pipeline):
        env.pipeline._on_seg_event(SegLevel(rms=0.2, vad_prob=0.9))
        env.pipeline._on_seg_event(SegSpeculative(utterance_id=1, samples=sample()))
        env.pipeline._on_seg_event(SegFinal(utterance_id=1, samples=sample()))
        time.sleep(0.05)
    assert len(levels) == 1  # meter still flows
    assert env.stt.calls == 0  # no transcription while muted
    assert env.chatbox.submits == []


def test_mute_gated_exposes_only_the_mute_sync_gate():
    # The GUI polls this to label a mute pause; it must track the mute gate
    # alone, independent of the master toggle (which the GUI already knows).
    env = make_pipeline(mute=FakeMute(caption=False))
    assert env.pipeline.mute_gated() is True
    assert env.pipeline._should_caption() is False

    open_env = make_pipeline(mute=FakeMute(caption=True))
    open_env.pipeline.set_captioning(False)
    assert open_env.pipeline.mute_gated() is False  # toggle off, gate open


def test_mute_gated_is_false_without_mute_sync():
    env = make_pipeline()  # mute=None
    assert env.pipeline.mute_gated() is False


def test_master_toggle_gates_captioning():
    env = make_pipeline()  # make_pipeline opts in; production default is off
    assert env.pipeline.captioning_enabled is True
    env.pipeline.set_captioning(False)
    assert env.pipeline.captioning_enabled is False
    levels = collect(env.bus, MicLevel)
    with running(env.pipeline):
        env.pipeline._on_seg_event(SegLevel(rms=0.2, vad_prob=0.9))
        env.pipeline._on_seg_event(SegSpeculative(utterance_id=1, samples=sample()))
        time.sleep(0.05)
    assert len(levels) == 1
    assert env.stt.calls == 0


def test_captioning_starts_off_by_default():
    # The raw Pipeline default (untouched by make_pipeline's opt-in) is off:
    # the user enables captioning explicitly each launch.
    env = make_pipeline(captioning=None)
    assert env.pipeline.captioning_enabled is False


# forward_final's send-time re-check of this same gate (catching a mute or
# captioning-off that lands after enqueue) lives in test_pipeline_gate.py.


def test_seg_final_before_enabling_creates_no_stt_job():
    # A final segment arriving before the user ever enables captioning must
    # not create an STT job -- _should_caption() gates it just like the
    # master-toggle-off case above.
    env = make_pipeline(captioning=None)
    with running(env.pipeline):
        env.pipeline._on_seg_event(SegFinal(utterance_id=1, samples=sample()))
        time.sleep(0.05)
    assert env.stt.calls == 0


# -- behavior 7: typing indicator -------------------------------------------


def test_typing_true_on_speculative():
    env = make_pipeline()
    with running(env.pipeline):
        env.pipeline._on_seg_event(SegSpeculative(utterance_id=1, samples=sample()))
        assert _wait_until(lambda: env.chatbox.typing[:1] == [True])


def test_typing_false_after_submit():
    env = make_pipeline()
    s = sample()
    with running(env.pipeline):
        env.pipeline._on_seg_event(SegSpeculative(utterance_id=1, samples=s))
        env.pipeline._on_seg_event(SegFinal(utterance_id=1, samples=s))
        assert _wait_until(lambda: len(env.chatbox.submits) == 1)
        assert _wait_until(lambda: env.chatbox.typing[-1] is False)
    assert env.chatbox.typing[0] is True
    # Typing must stay ON through the whole MT wait: the first typing-off
    # comes strictly AFTER the chatbox submit, never during translation.
    submit_idx = next(i for i, e in enumerate(env.chatbox.log) if e[0] == "submit")
    typing_off_idx = env.chatbox.log.index(("typing", False))
    assert submit_idx < typing_off_idx


def test_typing_false_on_discard_when_nothing_in_flight():
    env = make_pipeline()
    with running(env.pipeline):
        env.pipeline._on_seg_event(SegSpeculative(utterance_id=1, samples=sample()))
        assert _wait_until(lambda: env.chatbox.typing[:1] == [True])
        env.pipeline._on_seg_event(SegDiscard(utterance_id=1))
        assert _wait_until(lambda: env.chatbox.typing[-1] is False)


def test_typing_false_when_result_gated_to_none():
    env = make_pipeline(stt=FakeStt(result=None))
    s = sample()
    with running(env.pipeline):
        env.pipeline._on_seg_event(SegSpeculative(utterance_id=1, samples=s))
        assert _wait_until(lambda: env.chatbox.typing[:1] == [True])
        env.pipeline._on_seg_event(SegFinal(utterance_id=1, samples=s))
        assert _wait_until(lambda: env.chatbox.typing[-1] is False)


def test_orphaned_speculative_typing_cleared_by_later_finalize():
    # The segmenter invariant guarantees every speculative is resolved, so
    # an orphan (no SegFinal/SegDiscard for utterance 1) can only come from
    # a segmenter regression -- _mark_finalized's defensive prune must still
    # clear it so the typing indicator can't get stuck on forever.
    env = make_pipeline()
    with running(env.pipeline):
        env.pipeline._on_seg_event(SegSpeculative(utterance_id=1, samples=sample()))
        assert _wait_until(lambda: env.chatbox.typing[:1] == [True])
        # Utterance 1 is never resolved; utterance 2 finalizes normally.
        env.pipeline._on_seg_event(SegFinal(utterance_id=2, samples=sample()))
        assert _wait_until(lambda: len(env.chatbox.submits) == 1)
        assert _wait_until(lambda: env.chatbox.typing[-1] is False)
        assert 1 not in env.pipeline._typing._in_flight


# -- behavior 8: send_to_vrchat False ---------------------------------------


def test_send_to_vrchat_false_skips_chatbox_but_publishes_events():
    cfg = AppConfig(osc=OscConfig(send_to_vrchat=False))
    env = make_pipeline(config=cfg)
    recognized = collect(env.bus, PhraseRecognized)
    translated = collect(env.bus, PhraseTranslated)
    s = sample()
    with running(env.pipeline):
        env.pipeline._on_seg_event(SegSpeculative(utterance_id=1, samples=s))
        env.pipeline._on_seg_event(SegFinal(utterance_id=1, samples=s))
        assert _wait_until(lambda: len(translated) == 1)
        time.sleep(0.02)
    assert len(recognized) == 1
    assert env.chatbox.submits == []  # nothing sent to VRChat
    assert env.chatbox.typing == []  # typing also skipped


# Live-partial dispatch, coalescing, and the partial chatbox send live in
# test_pipeline_partial.py (kept separate to stay under the line cap).


# -- behavior 9: worker exceptions ------------------------------------------


# -- behavior 10: early sentence injection at the speculative pass ----------


class _CommitRecorder:
    """Stand-in segmenter recording request_commit calls. The early-injection
    tests drive process_stt_job synchronously and never run the segmenter's
    frame loop, so request_commit is all that is exercised."""

    def __init__(self) -> None:
        self.commits: list = []

    def request_commit(self, utterance_id: int) -> None:
        self.commits.append(utterance_id)


def _spec_job(uid: int, s) -> _SttJob:
    return _SttJob(utterance_id=uid, samples=s, speculative=True, samples_id=id(s))


def _final_job(uid: int, s) -> _SttJob:
    return _SttJob(utterance_id=uid, samples=s, speculative=False, samples_id=id(s))


def _run_speculative(env, uid: int, s) -> None:
    # Mirror handle_speculative's bookkeeping (typing on, pending noted), then
    # process the job in-thread so the assertions never chase a worker.
    env.pipeline._begin_typing(uid)
    env.pipeline._spec.note_speculative(uid, id(s))
    pipeline_jobs.process_stt_job(env.pipeline, _spec_job(uid, s), threading.Event())


def test_speculative_sentence_is_sent_early_and_commits():
    env = make_pipeline(mt=None, stt=FakeStt(result=make_result(text="Hello there.")))
    seg = _CommitRecorder()
    env.pipeline._segmenter = seg
    s = sample()
    _run_speculative(env, 1, s)
    assert env.chatbox.submits == [("Hello there.", 1)]  # sent once, now
    assert seg.commits == [1]  # and the segmenter was told to commit


def test_final_after_early_send_does_not_duplicate():
    env = make_pipeline(mt=None, stt=FakeStt(result=make_result(text="Hello there.")))
    seg = _CommitRecorder()
    env.pipeline._segmenter = seg
    s = sample()
    _run_speculative(env, 1, s)
    sent_after_spec = len(env.chatbox.submits)
    assert sent_after_spec == 1
    # The race the dedupe guards: the natural 600 ms final for this utterance
    # was already queued before request_commit could cut it off.
    pipeline_jobs.process_stt_job(env.pipeline, _final_job(1, s), threading.Event())
    assert len(env.chatbox.submits) == sent_after_spec  # no second send
    assert env.stt.calls == 1  # the final neither re-sent nor re-transcribed


def test_gated_early_inject_composes_with_final_dedupe_no_double_send():
    # forward_final runs BEFORE mark_emitted_early. A gate closing right
    # before that call skips the speculative send, but the guard still ends
    # up set (mark_emitted_early runs unconditionally after), so the
    # natural final racing the commit doesn't send either.
    env = make_pipeline(mt=None, stt=FakeStt(result=make_result(text="Hello there.")))
    seg = _CommitRecorder()
    env.pipeline._segmenter = seg
    s = sample()
    env.pipeline.set_captioning(False)  # closed before the speculative resolves
    _run_speculative(env, 1, s)
    assert env.chatbox.submits == []  # gated: the early send never happened
    assert seg.commits == [1]  # sentence injection still cuts the utterance
    assert 1 in env.pipeline._spec._emitted_early  # guard still set after
    pipeline_jobs.process_stt_job(env.pipeline, _final_job(1, s), threading.Event())
    assert env.chatbox.submits == []  # dedupe path: still no send, no re-transcribe
    assert env.stt.calls == 1


def test_speculative_without_terminal_punctuation_is_not_sent_early():
    env = make_pipeline(mt=None, stt=FakeStt(result=make_result(text="hello world")))
    seg = _CommitRecorder()
    env.pipeline._segmenter = seg
    s = sample()
    _run_speculative(env, 1, s)
    assert env.chatbox.submits == []
    assert seg.commits == []
    # Cached as before, so the eventual final still reuses the transcription.
    assert (1, id(s)) in env.pipeline._spec._cache


def test_one_word_sentence_is_blocked_by_the_min_word_guard():
    # "Hi." is a terminal mark but a single word; sentence_min_words is 2, so
    # an early send here would be a false positive on an abbreviation-like word.
    env = make_pipeline(mt=None, stt=FakeStt(result=make_result(text="Hi.")))
    seg = _CommitRecorder()
    env.pipeline._segmenter = seg
    s = sample()
    _run_speculative(env, 1, s)
    assert env.chatbox.submits == []
    assert seg.commits == []


def test_stt_worker_exception_publishes_apperror_and_continues():
    # First transcribe raises; the second succeeds -> proves the worker
    # survived the exception and kept draining its queue.
    env = make_pipeline(stt=FakeStt(results=[RuntimeError("stt boom"), make_result()]))
    errors = collect(env.bus, AppError)
    with running(env.pipeline):
        env.pipeline._on_seg_event(SegFinal(utterance_id=1, samples=sample()))
        assert _wait_until(lambda: any(e.code == "STT_JOB_FAILED" for e in errors))
        env.pipeline._on_seg_event(SegFinal(utterance_id=2, samples=sample()))
        assert _wait_until(lambda: env.pipeline._spec._last_finalized >= 2)
    err = next(e for e in errors if e.code == "STT_JOB_FAILED")
    assert "stt boom" in err.message


def test_mt_translation_failure_falls_back_to_original_text():
    env = make_pipeline(mt=FakeMt(raises=RuntimeError("mt boom")))
    errors = collect(env.bus, AppError)
    recognized = collect(env.bus, PhraseRecognized)
    with running(env.pipeline):
        env.pipeline._on_seg_event(SegFinal(utterance_id=1, samples=sample()))
        assert _wait_until(lambda: len(env.chatbox.submits) == 1)
        assert _wait_until(lambda: any(e.code == "MT_JOB_FAILED" for e in errors))
    assert [e.text for e in recognized] == ["hello world"]
    # caption did not vanish: original text was sent instead
    assert env.chatbox.submits[0] == ("hello world", 1)


def test_chatbox_failure_with_translation_disabled_publishes_chatbox_error():
    # chatbox.submit raising in the STT worker's direct-submit branch must
    # be classified CHATBOX_SEND_FAILED (not STT_JOB_FAILED) and must still
    # resolve the typing indicator.
    chatbox = FakeChatbox(fail_submits=True)
    env = make_pipeline(mt=None, chatbox=chatbox)
    errors = collect(env.bus, AppError)
    recognized = collect(env.bus, PhraseRecognized)
    s = sample()
    with running(env.pipeline):
        env.pipeline._on_seg_event(SegSpeculative(utterance_id=1, samples=s))
        assert _wait_until(lambda: chatbox.typing[:1] == [True])
        env.pipeline._on_seg_event(SegFinal(utterance_id=1, samples=s))
        assert _wait_until(lambda: any(e.code == "CHATBOX_SEND_FAILED" for e in errors))
        assert _wait_until(lambda: chatbox.typing[-1] is False)
    assert not any(e.code == "STT_JOB_FAILED" for e in errors)
    assert len(recognized) == 1  # recognition still published
