"""Tests for the MT timing and engine lines of the end-of-run summary in
:mod:`vrcc.core.pipeline_stats`: one sample per fresh translate() call (never
a pretranslate hit), and the device/precision the loaded engines report.
"""

from __future__ import annotations

import logging
import threading
from types import SimpleNamespace

from vrcc.core.config import AppConfig, SttConfig, TranslateConfig
from vrcc.core.languages import get as get_lang
from vrcc.core.pipeline_jobs import _MtJob, _SttJob, process_mt_job, process_stt_job
from vrcc.core.pipeline_stats import MtCallStats, log_summary

from .conftest import FakeMt, make_pipeline, sample

_LOGGER_NAME = "vrcc.core.pipeline"


def _env(mt=None):
    cfg = AppConfig(
        stt=SttConfig(source_language="auto"),
        translate=TranslateConfig(targets=["Japanese"]),
    )
    return make_pipeline(config=cfg, mt=mt or FakeMt())


def _lines(caplog, prefix):
    return [
        r.message for r in caplog.records
        if r.name == _LOGGER_NAME and r.message.startswith(prefix)
    ]


def _translate(env, uid=1, text="hello"):
    job = _MtJob(uid, text, get_lang("English"), manage_typing=False)
    process_mt_job(env.pipeline, job, threading.Event())


def test_fresh_mt_calls_are_summarized(caplog):
    env = _env()
    _translate(env, 1)
    _translate(env, 2)
    assert len(env.mt.calls) == 2

    with caplog.at_level(logging.INFO, logger=_LOGGER_NAME):
        log_summary(env.pipeline)

    lines = _lines(caplog, "MT run:")
    assert len(lines) == 1
    assert lines[0].startswith("MT run: 2 calls.")
    assert "median" in lines[0] and "p95" in lines[0]
    assert "hello" not in lines[0]


def test_no_mt_line_without_a_fresh_call(caplog):
    env = _env()
    with caplog.at_level(logging.INFO, logger=_LOGGER_NAME):
        log_summary(env.pipeline)
    assert _lines(caplog, "MT run:") == []


def test_failed_mt_call_is_not_counted():
    env = _env(mt=FakeMt(raises=RuntimeError("boom")))
    _translate(env)
    assert env.pipeline._mt_stats.snapshot() == []


def test_pretranslate_hit_is_not_counted():
    env = _env()
    stop = threading.Event()
    s = sample()
    process_stt_job(env.pipeline, _SttJob(1, s, True, id(s)), stop)
    q = env.pipeline._mt_queue
    while not q.empty():  # the prefetch: an MT call, but not a caption's
        process_mt_job(env.pipeline, q.get_nowait(), stop)
    process_stt_job(env.pipeline, _SttJob(1, s, False, id(s)), stop)
    while not q.empty():
        process_mt_job(env.pipeline, q.get_nowait(), stop)

    assert env.pipeline._pretranslate.hit == 1
    assert len(env.mt.calls) == 1
    assert env.pipeline._mt_stats.snapshot() == []


def test_mt_call_stats_keep_wait_and_call_separately():
    stats = MtCallStats()
    stats.record_call(0.1, 0.5)
    stats.record_call(0.0, 0.3)
    assert stats.snapshot() == [(0.1, 0.5), (0.0, 0.3)]


def test_engines_line_reports_actual_device_and_precision(caplog):
    env = _env()
    env.pipeline.stt_slot.current._device = "cpu"
    env.pipeline.stt_slot.current._compute_type = "int8"
    env.pipeline.mt_slot.current._device = "cuda"
    env.pipeline.mt_slot.current._spec = SimpleNamespace(quantization="int8_float16")

    with caplog.at_level(logging.INFO, logger=_LOGGER_NAME):
        log_summary(env.pipeline)

    assert _lines(caplog, "Engines:") == ["Engines: STT cpu:int8, MT cuda:int8_float16."]


def test_engines_line_names_a_missing_or_silent_engine(caplog):
    env = make_pipeline(mt=None)
    with caplog.at_level(logging.INFO, logger=_LOGGER_NAME):
        log_summary(env.pipeline)
    assert _lines(caplog, "Engines:") == ["Engines: STT -:-, MT none."]
