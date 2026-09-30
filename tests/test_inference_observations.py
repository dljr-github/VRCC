"""Runtime observations are numeric diagnostics, never calibration rankings."""

import json
import threading

import pytest

from vrcc.core import pipeline_jobs, inference_observations, languages

from .conftest import FakeMt, FakeStt, make_pipeline, sample


def test_final_stt_report_separates_inference_and_lock_time(monkeypatch):
    env = make_pipeline()
    times = iter([10.0, 11.0, 12.0, 14.0])
    monkeypatch.setattr(pipeline_jobs.time, "monotonic", lambda: next(times))
    pipeline_jobs._call_engine(
        env.pipeline, sample(), threading.Event(), speculative=False,
    )

    report = env.pipeline.observation_report()
    assert report["kind"] == "runtime_observations"
    stt = report["stt"]
    assert stt["count"] == 1
    assert stt["service_s"]["p50"] == 2.0
    assert stt["lock_wait_s"]["p95"] == 1.0
    assert stt["dispatch_wait_s"] is None  # no queued job in this direct call
    assert stt["metadata"]["device"] == "unknown"
    assert "hello world" not in json.dumps(report)


def test_queued_final_reports_dispatch_wait_separately(monkeypatch):
    env = make_pipeline(mt=None)
    samples = sample()
    job = pipeline_jobs._SttJob(1, samples, False, id(samples), created_at=7.0)
    times = iter([10.0, 11.0, 12.0, 14.0])
    monkeypatch.setattr(pipeline_jobs.time, "monotonic", lambda: next(times))
    pipeline_jobs.process_stt_job(env.pipeline, job, threading.Event())
    stt = env.pipeline.observation_report()["stt"]
    assert stt["dispatch_wait_s"]["p50"] == 3.0
    assert stt["service_s"]["p50"] == 2.0
    assert stt["lock_wait_s"]["p50"] == 1.0


def test_mt_observation_is_numeric_and_records_actual_target_count(monkeypatch):
    env = make_pipeline(mt=FakeMt())
    env.pipeline._config.translate.targets = ["English", "Japanese", "Korean"]
    job = pipeline_jobs._MtJob(
        1, "private words", languages.get("English"), False, created_at=3.0,
    )
    times = iter([5.0, 6.0, 6.0, 8.0])
    monkeypatch.setattr(pipeline_jobs.time, "monotonic", lambda: next(times))
    pipeline_jobs.process_mt_job(env.pipeline, job, threading.Event())
    report = env.pipeline.observation_report()
    mt = report["mt"]
    assert mt["count"] == 1
    assert mt["metadata"]["target_count"] == 2
    assert mt["input_chars"]["p50"] == 13
    assert mt["dispatch_wait_s"]["p50"] == 2.0
    assert mt["lock_wait_s"]["p50"] == 1.0
    assert mt["service_s"]["p50"] == 2.0
    assert mt["real_time_factor"] is None
    assert "private words" not in json.dumps(report)


def test_stop_logs_json_observations_and_start_clears_them(caplog):
    env = make_pipeline()
    env.pipeline.start()
    pipeline_jobs._call_engine(
        env.pipeline, sample(), threading.Event(), speculative=False,
    )
    with caplog.at_level("INFO", logger="vrcc.core.pipeline"):
        env.pipeline.stop()
    lines = [r.message for r in caplog.records if r.message.startswith("Inference observations: ")]
    assert len(lines) == 1
    assert json.loads(lines[0].split(": ", 1)[1])["stt"]["count"] == 1
    env.pipeline.start()
    try:
        assert env.pipeline.observation_report()["stt"] is None
    finally:
        env.pipeline.stop()


def test_observations_keep_only_the_last_128_samples():
    env = make_pipeline()
    ctx = inference_observations.context(env.stt, env.pipeline._config.stt, "stt")
    for n in range(140):
        env.pipeline._observations.record(
            "stt", ctx, service_s=float(n), lock_wait_s=0, audio_s=2,
        )
    stt = env.pipeline.observation_report()["stt"]
    assert stt["count"] == 128
    assert stt["service_s"]["p50"] == 75.5
    assert stt["real_time_factor"]["p50"] == 37.75


@pytest.mark.parametrize("change", ["beam_size", "cpu_threads", "model", "initial_prompt"])
def test_changed_stt_configuration_invalidates_old_samples(change):
    env = make_pipeline()
    pipeline_jobs._call_engine(env.pipeline, sample(), threading.Event(), speculative=False)
    cfg = env.pipeline._config.stt
    setattr(cfg, change, {"beam_size": 5, "cpu_threads": 2, "model": "tiny",
                          "initial_prompt": "a private prompt"}[change])
    assert env.pipeline.observation_report()["stt"] is None


def test_engine_swap_invalidates_even_with_identical_configuration():
    env = make_pipeline()
    pipeline_jobs._call_engine(env.pipeline, sample(), threading.Event(), speculative=False)
    env.pipeline.set_stt(FakeStt())
    assert env.pipeline.observation_report()["stt"] is None


def test_actual_device_is_reported_and_mid_call_fallback_is_excluded():
    env = make_pipeline()
    env.stt._device, env.stt._compute_type = "cuda", "float16"
    original = env.stt.transcribe
    def fallback(samples):
        env.stt._device, env.stt._compute_type = "cpu", "int8"
        return original(samples)
    env.stt.transcribe = fallback
    pipeline_jobs._call_engine(env.pipeline, sample(), threading.Event(), speculative=False)
    assert env.pipeline.observation_report()["stt"] is None
    pipeline_jobs._call_engine(env.pipeline, sample(), threading.Event(), speculative=False)
    assert env.pipeline.observation_report()["stt"]["metadata"]["device"] == "cpu"
    assert env.pipeline.observation_report()["stt"]["metadata"]["precision"] == "int8"


def test_unknown_model_path_and_prompt_never_appear_in_report():
    env = make_pipeline()
    cfg = env.pipeline._config.stt
    cfg.model, cfg.initial_prompt = "C:/private/location", "secret prompt"
    cfg.extra_transcribe_kwargs = {"private": "secret setting"}
    pipeline_jobs._call_engine(env.pipeline, sample(), threading.Event(), speculative=False)
    report = json.dumps(env.pipeline.observation_report())
    assert "private" not in report and "secret" not in report
    assert env.pipeline.observation_report()["stt"]["metadata"]["model_id"] == "unknown"


@pytest.mark.parametrize("audio_s, service_s, wait_s", [
    (0, 1, 0), (-1, 1, 0), (1, float("nan"), 0),
    (1, float("inf"), 0), (1, -1, 0), (1, 1, -1),
])
def test_invalid_samples_are_not_recorded(audio_s, service_s, wait_s):
    env = make_pipeline()
    ctx = inference_observations.context(env.stt, env.pipeline._config.stt, "stt")
    env.pipeline._observations.record(
        "stt", ctx, service_s=service_s, lock_wait_s=wait_s, audio_s=audio_s,
    )
    assert env.pipeline.observation_report()["stt"] is None


def test_speculative_and_reused_final_do_not_claim_fresh_inference():
    env = make_pipeline()
    samples = sample()
    pipeline_jobs.process_stt_job(
        env.pipeline, pipeline_jobs._SttJob(1, samples, True, id(samples)), threading.Event(),
    )
    pipeline_jobs.process_stt_job(
        env.pipeline, pipeline_jobs._SttJob(1, samples, False, id(samples)), threading.Event(),
    )
    assert env.pipeline.observation_report()["stt"] is None


def test_failed_and_abandoned_stt_calls_do_not_record():
    env = make_pipeline(stt=FakeStt(raises=RuntimeError("failure")))
    with pytest.raises(RuntimeError):
        pipeline_jobs._call_engine(env.pipeline, sample(), threading.Event(), speculative=False)
    assert env.pipeline.observation_report()["stt"] is None
    env = make_pipeline()
    stop = threading.Event()
    stop.set()
    pipeline_jobs._call_engine(env.pipeline, sample(), stop, speculative=False)
    assert env.pipeline.observation_report()["stt"] is None


def test_configuration_change_during_stt_call_excludes_sample():
    env = make_pipeline()
    original = env.stt.transcribe
    def changed(samples):
        env.pipeline._config.stt.cpu_threads = 2
        return original(samples)
    env.stt.transcribe = changed
    pipeline_jobs._call_engine(env.pipeline, sample(), threading.Event(), speculative=False)
    assert env.pipeline.observation_report()["stt"] is None


@pytest.mark.parametrize("case", ["failure", "stopped", "no_targets", "no_engine"])
def test_mt_paths_without_completed_work_do_not_record(case):
    mt = None if case == "no_engine" else FakeMt(
        raises=RuntimeError("failure") if case == "failure" else None,
    )
    env = make_pipeline(mt=mt)
    if case == "no_targets":
        env.pipeline._config.translate.targets = ["English"]
    stop = threading.Event()
    if case == "stopped":
        stop.set()
    job = pipeline_jobs._MtJob(1, "private", languages.get("English"), False)
    pipeline_jobs.process_mt_job(env.pipeline, job, stop)
    assert env.pipeline.observation_report()["mt"] is None


def test_actual_model_identity_is_not_inferred_from_a_new_requested_model(tmp_path):
    env = make_pipeline()
    env.stt._model_dir = tmp_path / "small"
    env.pipeline._config.stt.model = "large-v3"
    pipeline_jobs._call_engine(env.pipeline, sample(), threading.Event(), speculative=False)
    meta = env.pipeline.observation_report()["stt"]["metadata"]
    assert meta["model_id"] == "small"
    assert meta["requested_model_id"] == "large-v3"


def test_reset_during_inference_cannot_repopulate_next_run_observations():
    env = make_pipeline()
    original = env.stt.transcribe
    def restart(samples):
        env.pipeline._observations.reset()
        return original(samples)
    env.stt.transcribe = restart
    pipeline_jobs._call_engine(env.pipeline, sample(), threading.Event(), speculative=False)
    assert env.pipeline.observation_report()["stt"] is None


def test_diagnostic_failure_does_not_drop_stt_or_mt_results(monkeypatch):
    env = make_pipeline()
    def broken_context(*args):
        raise RuntimeError("diagnostics failed")
    monkeypatch.setattr(inference_observations, "context", broken_context)
    result = pipeline_jobs._call_engine(
        env.pipeline, sample(), threading.Event(), speculative=False,
    )
    assert result.text == "hello world"
    job = pipeline_jobs._MtJob(1, "hello", languages.get("English"), False)
    pipeline_jobs.process_mt_job(env.pipeline, job, threading.Event())
    assert env.mt.calls[0][0] == "hello"
    assert env.chatbox.messages[0][1]  # translation wasn't lost to diagnostics


def test_mt_changes_in_actual_source_start_a_separate_window():
    env = make_pipeline()
    for language in ("English", "French"):
        pipeline_jobs.process_mt_job(env.pipeline, pipeline_jobs._MtJob(
            1, "hello", languages.get(language), True,
        ), threading.Event())
    mt = env.pipeline.observation_report()["mt"]
    assert mt["count"] == 1
    assert mt["metadata"]["source_language"] == "fra_Latn"


def test_mt_old_target_list_after_wait_is_not_attributed_to_new_configuration():
    env = make_pipeline()
    old_targets = [languages.get("Japanese")]
    env.pipeline._config.translate.targets = ["French"]
    job = pipeline_jobs._MtJob(1, "hello", languages.get("English"), True)
    inference_observations.translate(
        env.pipeline, env.mt, job, old_targets, threading.Event(), job.created_at,
    )
    assert env.pipeline.observation_report()["mt"] is None


def test_invalid_new_mt_targets_only_disable_diagnostics_for_an_old_valid_job():
    env = make_pipeline()
    env.pipeline._config.translate.targets = ["not a language"]
    job = pipeline_jobs._MtJob(1, "hello", languages.get("English"), True)
    result = inference_observations.translate(
        env.pipeline, env.mt, job, [languages.get("Japanese")],
        threading.Event(), job.created_at,
    )
    assert result == [("Japanese", "Japanese:hello")]
    assert env.pipeline.observation_report()["mt"] is None


def test_chinese_scripts_do_not_share_an_mt_source_window():
    env = make_pipeline()
    for language in ("Chinese Simplified", "Chinese Traditional"):
        pipeline_jobs.process_mt_job(env.pipeline, pipeline_jobs._MtJob(
            1, "private", languages.get(language), True,
        ), threading.Event())
    mt = env.pipeline.observation_report()["mt"]
    assert mt["count"] == 1
    assert mt["metadata"]["source_language"] == "zho_Hant"


def test_mt_target_count_describes_completed_outputs_after_family_deduplication():
    env = make_pipeline(mt=FakeMt(translations=[("Chinese Simplified", "private")]))
    env.pipeline._config.translate.targets = ["Chinese Simplified", "Chinese Traditional"]
    pipeline_jobs.process_mt_job(env.pipeline, pipeline_jobs._MtJob(
        1, "private", languages.get("English"), True,
    ), threading.Event())
    meta = env.pipeline.observation_report()["mt"]["metadata"]
    assert meta["requested_target_count"] == 2
    assert meta["target_count"] == 1
