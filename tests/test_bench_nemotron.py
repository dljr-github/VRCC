"""tools/bench_nemotron.py: archive member filtering, tail padding and the
live-decode loop. No model is loaded; the recognizer is a stub."""

import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("soundfile")

_TOOLS = Path(__file__).resolve().parent.parent / "tools"
sys.path.insert(0, str(_TOOLS))

import bench_nemotron  # noqa: E402

SR = bench_nemotron.SAMPLE_RATE


@pytest.mark.parametrize("member,dest", [
    ("pkg/encoder.int8.onnx", "encoder.int8.onnx"),
    ("pkg/tokens.txt", "tokens.txt"),
    ("pkg/test_wavs/ja.wav", "test_wavs/ja.wav"),
    ("pkg/encoder.onnx", None),
    ("pkg/encoder.data", None),
    ("pkg/joiner.onnx", None),
    ("../../evil/decoder.int8.onnx", "decoder.int8.onnx"),
    ("pkg/.hidden", None),
])
def test_keep_flattens_and_drops_fp32_graphs(member, dest):
    assert bench_nemotron._keep(member) == dest


def test_archive_url_names_the_chunk():
    url = bench_nemotron.archive_url(1120)
    assert url.endswith("streaming-0.6b-1120ms-int8-2026-06-11.tar.bz2")
    assert url.startswith("https://github.com/k2-fsa/sherpa-onnx/releases/")


def test_tail_defaults_to_two_chunks():
    assert bench_nemotron.tail_for(None, 560) == pytest.approx(1.12)
    assert bench_nemotron.tail_for(0.3, 560) == 0.3


class _Stream:
    def __init__(self):
        self.samples = 0
        self.finished = False
        self.options = {}

    def set_option(self, key, value):
        self.options[key] = value

    def accept_waveform(self, sr, audio):
        assert sr == SR
        self.samples += len(audio)

    def input_finished(self):
        self.finished = True


class _Recognizer:
    """Emits one word per 0.5 s of audio it has seen; 'ready' once per call."""

    def __init__(self):
        self.streams = []
        self._decoded = {}

    def create_stream(self):
        s = _Stream()
        self.streams.append(s)
        return s

    def is_ready(self, s):
        return self._decoded.get(id(s)) != s.samples

    def decode_stream(self, s):
        self._decoded[id(s)] = s.samples

    def get_result(self, s):
        return " ".join(["w"] * (s.samples // (SR // 2)))


def test_live_feeds_every_sample_and_finishes():
    rec = _Recognizer()
    audio = np.zeros(SR * 2, np.float32)
    out = bench_nemotron.decode_live(rec, audio, "ja", 0.5, 32)
    stream = rec.streams[0]
    assert stream.options == {"language": "ja"}
    assert stream.finished
    assert stream.samples == len(audio) + SR // 2
    assert out["text"] == "w w w w w"
    assert out["partials"] == 4  # changes seen while feeding, before the tail
    assert 0 <= out["finalize_s"] <= out["compute_s"]


def test_auto_language_sets_no_option():
    rec = _Recognizer()
    text, secs = bench_nemotron.decode_offline(rec, np.zeros(SR, np.float32), "auto", 0.5)
    assert rec.streams[0].options == {}
    assert text == "w w w" and secs >= 0


def test_table_has_one_row_per_chunk():
    row = {"chunk_ms": 560, "provider": "cpu", "threads": 4, "wer_offline": 0.02,
           "wer_live": 0.021, "offline_median_s": 0.2, "offline_rtf": 0.1,
           "live_rtf": 0.12, "finalize_median_s": 0.05, "finalize_p90_s": 0.08,
           "partials_mean": 6.0}
    lines = bench_nemotron.table([row, {**row, "chunk_ms": 1120}]).splitlines()
    assert len(lines) == 4
    assert lines[2].startswith("| 560 ms | cpu | 4 | 2.0% | 2.1% |")
