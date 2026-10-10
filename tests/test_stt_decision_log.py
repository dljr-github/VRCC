"""The per-transcription decision record: fields, verdicts, and the rule that
caption text never reaches the log."""

from __future__ import annotations

import logging
import threading

import numpy as np

from vrcc.core import pipeline_jobs
from vrcc.core.pipeline_jobs import _SttJob
from vrcc.stt.decision_log import TAG

from .conftest import make_pipeline, make_result, sample
from .stt_fakes import _cfg, _seg
from .test_stt_language_bias import engine_for
from .test_stt_onnx_asr_transcribe import _loaded_engine
from .test_stt_sensevoice import _speech
from .test_stt_sensevoice_bias import LanguageSession, make_engine
from .test_stt_sensevoice_language_slot import model_dir  # noqa: F401

SECOND = np.zeros(16000, dtype=np.float32)


def records(caplog) -> list[dict[str, str]]:
    rows = []
    for rec in caplog.records:
        message = rec.getMessage()
        if message.startswith(TAG):
            rows.append(dict(part.split("=", 1) for part in message.split()[1:]))
    return rows


def test_whisper_record_reports_detection_bias_and_gate_inputs(caplog):
    engine, _ = engine_for([("fr", 0.45), ("en", 0.35), ("ja", 0.20)])
    with caplog.at_level(logging.DEBUG, logger="vrcc.stt"):
        engine.transcribe(SECOND)
    (row,) = records(caplog)
    assert row["engine"] == "whisper"
    assert row["audio_s"] == "1.00"
    assert (row["lang"], row["lang_p"]) == ("fr", "0.450")
    assert (row["alt"], row["alt_p"]) == ("en", "0.350")
    assert row["used"] == "en" and row["biased"] == "1"
    assert float(row["redecode_s"]) >= 0
    assert row["avg_logprob"] == "-0.100" and row["no_speech"] == "0.050"
    assert row["compression"] == "1.00"
    assert row["verdict"] == "accepted"
    assert row["chars"] == str(len("caption in en"))


def test_whisper_unbiased_decode_has_no_redecode_time(caplog):
    engine, _ = engine_for([("fr", 0.90), ("en", 0.06), ("ja", 0.04)])
    with caplog.at_level(logging.DEBUG, logger="vrcc.stt"):
        engine.transcribe(SECOND)
    (row,) = records(caplog)
    assert row["biased"] == "0" and row["redecode_s"] == "-"
    assert row["lang"] == row["used"] == "fr"


def test_whisper_gate_verdict_is_recorded(caplog):
    engine, model = engine_for([("en", 0.9), ("ja", 0.1)])
    cfg_gate = engine._cfg.avg_logprob_gate
    model_seg = _seg("secret words", 0, 1, cfg_gate - 1, 0.0)
    model.transcribe = lambda samples, **kw: (
        iter([model_seg]),
        type("Info", (), {"language": "en", "all_language_probs": [("en", 0.9)]})(),
    )
    with caplog.at_level(logging.DEBUG, logger="vrcc.stt"):
        assert engine.transcribe(SECOND) is None
    (row,) = records(caplog)
    assert row["verdict"] == "gated_avg_logprob"
    assert "secret" not in caplog.text


def test_sensevoice_record_reports_first_frame_languages_and_redecode(
    model_dir, caplog  # noqa: F811
):
    engine = make_engine(model_dir, LanguageSession())
    with caplog.at_level(logging.DEBUG, logger="vrcc.stt"):
        result = engine.transcribe(_speech())
    (row,) = records(caplog)
    assert row["engine"] == "sensevoice"
    assert (row["lang"], row["alt"]) == ("ja", "en")
    assert float(row["lang_p"]) > float(row["alt_p"]) > 0
    assert row["used"] == "en" and row["biased"] == "1"
    assert float(row["redecode_s"]) >= 0
    assert row["no_speech"] == "-" and row["verdict"] == "accepted"
    assert result.text not in caplog.text


def test_parakeet_record_has_gate_input_and_no_language(model_dir, caplog):
    engine, _ = _loaded_engine(model_dir, text="private words", logprobs=[-0.2, -0.4])
    with caplog.at_level(logging.DEBUG, logger="vrcc.stt"):
        engine.transcribe(SECOND)
    (row,) = records(caplog)
    assert row["engine"] == "parakeet" and row["lang"] == "-"
    assert row["avg_logprob"] == "-0.300" and row["chars"] == "13"
    assert "private" not in caplog.text


def test_parakeet_empty_text_records_verdict(model_dir, caplog):
    engine, _ = _loaded_engine(model_dir, text="")
    with caplog.at_level(logging.DEBUG, logger="vrcc.stt"):
        assert engine.transcribe(SECOND) is None
    assert records(caplog)[0]["verdict"] == "empty"


def test_final_reusing_a_speculative_logs_one_line_without_text(caplog):
    env = make_pipeline()
    s = sample()
    result = make_result()
    env.pipeline._spec.note_speculative(3, id(s))
    env.pipeline._spec.store_result((3, id(s)), result)
    job = _SttJob(utterance_id=3, samples=s, speculative=False, samples_id=id(s))
    with caplog.at_level(logging.DEBUG, logger="vrcc.core.pipeline"):
        pipeline_jobs.process_stt_job(env.pipeline, job, threading.Event())
    lines = [r.getMessage() for r in caplog.records if r.getMessage().startswith(TAG)]
    assert lines == [f"{TAG} final utterance=3 reused=1"]
    assert result.text not in caplog.text


def test_decode_failure_records_error_not_accepted(model_dir, caplog, monkeypatch):
    engine, _ = _loaded_engine(model_dir, text="x")

    def boom(*args):
        raise RuntimeError("cuda gone")

    monkeypatch.setattr(engine, "_recognize", boom)
    with caplog.at_level(logging.DEBUG, logger="vrcc.stt"):
        try:
            engine.transcribe(SECOND)
        except RuntimeError:
            pass
    assert [r["verdict"] for r in records(caplog)] == ["error"]


def test_fresh_final_logs_pass_marker_before_final_line(caplog):
    env = make_pipeline()
    s = sample()
    job = _SttJob(utterance_id=4, samples=s, speculative=False, samples_id=id(s))
    with caplog.at_level(logging.DEBUG, logger="vrcc.core.pipeline"):
        pipeline_jobs.process_stt_job(env.pipeline, job, threading.Event())
    lines = [r.getMessage() for r in caplog.records if r.getMessage().startswith(TAG)]
    assert lines == [
        f"{TAG} pass=final utterance=4",
        f"{TAG} final utterance=4 reused=0",
    ]
