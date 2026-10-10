"""tools/bench_stt.py: short-arm selection, fingerprint separation and the
fields every result row carries. No model is loaded; the engine is a stub."""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile

_TOOLS = Path(__file__).resolve().parent.parent / "tools"
sys.path.insert(0, str(_TOOLS))

import bench_data  # noqa: E402
import bench_stt  # noqa: E402

SR = bench_data.SAMPLE_RATE
_LENGTHS_S = {"1-1-0000": 1.0, "1-1-0001": 2.5, "1-1-0002": 2.6, "2-1-0000": 5.0}


@pytest.fixture
def corpus(tmp_path):
    lines: dict[Path, list[str]] = {}
    for utt_id, secs in _LENGTHS_S.items():
        spk, chap, _ = utt_id.split("-")
        folder = tmp_path / spk / chap
        folder.mkdir(parents=True, exist_ok=True)
        soundfile.write(folder / f"{utt_id}.flac", np.zeros(int(secs * SR)), SR)
        lines.setdefault(folder / f"{spk}-{chap}.trans.txt", []).append(
            f"{utt_id} word {utt_id[-1]}"
        )
    for path, rows in lines.items():
        path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return tmp_path


def _ids(utts):
    return [u[0] for u in utts]


def test_no_bound_keeps_everything(corpus):
    assert _ids(bench_data.load_utterances(corpus, 10)) == sorted(_LENGTHS_S)


def test_bound_keeps_only_short_utterances_inclusive(corpus):
    got = bench_data.load_utterances(corpus, 10, max_seconds=2.6)
    assert _ids(got) == ["1-1-0000", "1-1-0001", "1-1-0002"]
    assert all(len(audio) / SR <= 2.6 for _, audio, _ in got)


def test_bound_with_nothing_short_enough_is_empty(corpus):
    assert bench_data.load_utterances(corpus, 10, max_seconds=0.5) == []


def test_bound_applies_before_spacing(corpus):
    got = bench_data.load_utterances(corpus, 2, max_seconds=2.6)
    assert _ids(got) == ["1-1-0000", "1-1-0001"]


def test_fingerprint_separates_arms_with_identical_ids():
    utts = [("a", np.zeros(1), "x")]
    base = bench_data.utterance_set_id(utts)
    short = bench_data.utterance_set_id(utts, max_seconds=2.6)
    gated = bench_data.utterance_set_id(utts, gated=True)
    assert len({base, short, gated}) == 3
    assert short != bench_data.utterance_set_id(utts, max_seconds=3.0)


def test_default_fingerprint_is_the_historic_one():
    utts = [("a", np.zeros(1), "x"), ("b", np.zeros(1), "y")]
    import hashlib

    assert bench_data.utterance_set_id(utts) == (
        hashlib.sha256(b"a\nb").hexdigest()[:12]
    )


class _StubEngine:
    def __init__(self, cfg, texts):
        self.cfg = cfg
        self._texts = iter(texts)
        self._device = "cuda"
        self._compute_type = "int8_float16"

    def load(self):
        pass

    def warm_up(self):
        pass

    def unload(self):
        pass

    def transcribe(self, audio):
        text = next(self._texts, "")
        if text is None:
            return None
        return SimpleNamespace(text=text, avg_logprob=-0.1, no_speech_prob=0.0)


def _run(monkeypatch, corpus, **kwargs):
    seen = {}
    pytest.importorskip("jiwer")
    pytest.importorskip("whisper_normalizer")
    utts = bench_data.load_utterances(corpus, 10, kwargs.get("max_seconds"))

    def fake_create(cfg, path, bus):
        seen["cfg"] = cfg
        seen["engine"] = _StubEngine(cfg, ["w", "w", "word 0", None, "word 2", "word 0"])
        return seen["engine"]

    monkeypatch.setattr(bench_stt, "create_stt_engine", fake_create)
    row = bench_stt.bench_model("tiny", "cuda", utts, corpus, **kwargs)
    return row, seen["cfg"], utts


def test_row_records_arm_compute_and_per_utterance_data(monkeypatch, corpus):
    row, cfg, utts = _run(monkeypatch, corpus, max_seconds=2.6)
    assert row["compute"] == "int8_float16"
    assert row["compute_requested"] == "auto"
    assert row["max_seconds"] == 2.6
    assert row["gates"] == "open"
    assert row["n_utterances"] == len(utts) == 3
    assert set(row["hypotheses"]) == set(row["latencies_s"]) == set(_ids(utts))
    assert row["hypotheses"]["1-1-0001"] == ""
    assert row["empty"] == 1
    assert row["utterance_set"] == bench_data.utterance_set_id(utts, None, 2.6)


def test_open_gates_by_default(monkeypatch, corpus):
    _, cfg, _ = _run(monkeypatch, corpus)
    assert cfg.avg_logprob_gate < -1e6
    assert cfg.no_speech_gate >= 1.0


def test_gated_scoring_keeps_app_defaults_and_marks_the_row(monkeypatch, corpus):
    row, cfg, utts = _run(monkeypatch, corpus, gated_scoring=True)
    defaults = bench_stt.SttConfig()
    assert cfg.avg_logprob_gate == defaults.avg_logprob_gate
    assert cfg.compression_ratio_gate == defaults.compression_ratio_gate
    assert cfg.parakeet_avg_logprob_gate == defaults.parakeet_avg_logprob_gate
    assert row["gates"] == "app"
    assert row["gated"] is None
    assert row["utterance_set"] == bench_data.utterance_set_id(
        utts, None, None, True
    )


def test_export_refuses_to_mix_short_and_long_rows(tmp_path, capsys):
    import json

    for name, set_id in (("a", "111"), ("b", "222")):
        (tmp_path / f"{name}__cuda.json").write_text(
            json.dumps({"device": "cuda", "utterance_set": set_id})
        )
    assert bench_stt.export_results(tmp_path, tmp_path / "out.json") == 1
    assert "refusing to export" in capsys.readouterr().out


def test_denoise_gate_default_tracks_the_config():
    import denoise_realnoise_gate as gate
    from vrcc.core.vad_config import AudioConfig

    assert gate._SHIPPED_STRENGTH == AudioConfig().denoise_strength
