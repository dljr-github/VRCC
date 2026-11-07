"""tools/bench_fleurs.py: CER normalization and clip selection. No model is
loaded."""

import sys
from pathlib import Path

import numpy as np
import pytest

soundfile = pytest.importorskip("soundfile")

_TOOLS = Path(__file__).resolve().parent.parent / "tools"
sys.path.insert(0, str(_TOOLS))

import bench_fleurs  # noqa: E402


@pytest.mark.parametrize("raw,norm", [
    ("場にふさわしい、敬意を払い。", "場にふさわしい敬意を払い"),
    ("안녕하세요, 반갑습니다!", "안녕하세요반갑습니다"),
    ("ＡＢＣ　１２３", "abc123"),
    ("你好，世界！", "你好世界"),
])
def test_normalize_drops_punctuation_space_and_width(raw, norm):
    assert bench_fleurs.normalize(raw) == norm


def test_load_clips_takes_one_recording_per_sentence(tmp_path):
    folder = tmp_path / "ja_jp"
    (folder / "dev").mkdir(parents=True)
    rows = []
    for i, sid in enumerate(["1", "1", "2", "3", "4"]):
        name = f"{i}.wav"
        soundfile.write(folder / "dev" / name, np.zeros(1600, np.float32), 16000)
        rows.append(f"{sid}\t{name}\tsentence {sid}\tnorm")
    (folder / "dev.tsv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    clips = bench_fleurs.load_clips(tmp_path, "ja_jp", 10)
    assert [ref for _, _, ref in clips] == [f"sentence {s}" for s in "1234"]
    assert [name for name, _, _ in clips][0] == "0.wav"
    assert len(bench_fleurs.load_clips(tmp_path, "ja_jp", 2)) == 2
