"""Character error rate of VRCC's STT models and Nemotron 3.5 on FLEURS (dev tool).

LibriSpeech (tools/bench_stt.py) only ranks English. This scores Japanese,
Korean and Chinese, the languages Parakeet does not cover and Nemotron 3.5
Streaming does, on Google's FLEURS read speech. Every model runs on the CPU
with the spoken language set, as a user who picked it in Settings would run
it, and with the quality gates open, as bench_stt does.

Run (FLEURS dev split unpacked under --fleurs, one folder per locale with
``dev.tsv`` and ``dev/*.wav``, from huggingface.co/datasets/google/fleurs):
    python tools/bench_fleurs.py --langs ja,ko,zh --utterances 60 \\
        --models sense-voice-small,small,large-v3-turbo,nemotron-560

``nemotron-<ms>`` runs sherpa-onnx's int8 export (tools/bench_nemotron.py
downloads it); every other id is a VRCC registry model run through
``create_stt_engine``. CER ignores punctuation, spaces and case on both
sides, since FLEURS references and the models disagree on those and
neither shows up in a caption's meaning. Chinese is scored as written:
a Traditional glyph against a Simplified reference counts as an error,
which is also what a reader of the caption would see.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import unicodedata
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import bench_nemotron  # noqa: E402
from vrcc.core.bus import EventBus  # noqa: E402
from vrcc.core.config import SttConfig  # noqa: E402
from vrcc.stt import create_stt_engine  # noqa: E402

SAMPLE_RATE = 16000
# short code -> (FLEURS locale folder, VRCC display name)
LANGS = {
    "ja": ("ja_jp", "Japanese"),
    "ko": ("ko_kr", "Korean"),
    "zh": ("cmn_hans_cn", "Chinese Simplified"),
}
OPEN_GATES = {
    "avg_logprob_gate": -1e9, "no_speech_gate": 1.0,
    "compression_ratio_gate": 1e9, "parakeet_avg_logprob_gate": -1e9,
    "sensevoice_avg_logprob_gate": -1e9,
}


def normalize(text: str) -> str:
    """Drop punctuation, symbols and whitespace; fold width and case."""
    text = unicodedata.normalize("NFKC", text).lower()
    return "".join(ch for ch in text if unicodedata.category(ch)[0] not in "PZSC")


def load_clips(root: Path, locale: str, n: int) -> list[tuple[str, np.ndarray, str]]:
    """N (id, audio, reference) clips, evenly spaced over the tsv, one per
    sentence (FLEURS records each sentence from several speakers)."""
    import soundfile

    folder = root / locale
    rows, seen = [], set()
    for line in (folder / "dev.tsv").read_text(encoding="utf-8").splitlines():
        cols = line.split("\t")
        if len(cols) < 3 or cols[0] in seen:
            continue
        seen.add(cols[0])
        rows.append((cols[1], cols[2]))
    n = min(n, len(rows))
    picked = [rows[i * len(rows) // n] for i in range(n)]
    clips = []
    for name, ref in picked:
        audio, sr = soundfile.read(folder / "dev" / name, dtype="float32",
                                   always_2d=False)
        if sr != SAMPLE_RATE:
            raise RuntimeError(f"{name}: {sr} Hz, expected {SAMPLE_RATE}")
        clips.append((name, audio, ref))
    return clips


class _VrccModel:
    def __init__(self, model_id: str, display: str, models_dir: Path, threads: int):
        cfg = SttConfig(model=model_id, device="cpu", source_language=display,
                        cpu_threads=threads, **OPEN_GATES)
        self.engine = create_stt_engine(cfg, models_dir / "whisper" / model_id,
                                        EventBus(), model_id=model_id)
        self.engine.load()
        self.engine.warm_up()

    def run(self, audio: np.ndarray) -> dict:
        t0 = time.perf_counter()
        result = self.engine.transcribe(audio)
        return {"text": result.text if result else "",
                "latency_s": time.perf_counter() - t0}


class _Nemotron:
    def __init__(self, chunk_ms: int, lang: str, models_dir: Path, threads: int):
        model_dir = bench_nemotron.ensure_model(models_dir, chunk_ms)
        self.rec = bench_nemotron.build_recognizer(model_dir, threads, "cpu")
        self.lang = lang
        self.tail = bench_nemotron.tail_for(None, chunk_ms)

    def run(self, audio: np.ndarray) -> dict:
        live = bench_nemotron.decode_live(self.rec, audio, self.lang, self.tail, 32)
        return {"text": live["text"], "latency_s": live["finalize_s"],
                "compute_s": live["compute_s"]}


def build(model: str, lang: str, models_dir: Path, threads: int):
    if model.startswith("nemotron-"):
        return _Nemotron(int(model.split("-", 1)[1]), lang, models_dir, threads)
    return _VrccModel(model, LANGS[lang][1], models_dir, threads)


def score(model: str, lang: str, clips, runner) -> dict:
    import jiwer

    runner.run(clips[0][1])  # untimed warm run
    outs = [runner.run(audio) for _, audio, _ in clips]
    refs = [normalize(ref) for _, _, ref in clips]
    hyps = [normalize(o["text"]) for o in outs]
    pairs = [(r, h) for r, h in zip(refs, hyps) if r]
    audio_s = sum(len(a) / SAMPLE_RATE for _, a, _ in clips)
    lat = [o["latency_s"] for o in outs]
    compute = sum(o.get("compute_s", o["latency_s"]) for o in outs)
    return {
        "model": model, "lang": lang, "n": len(clips),
        "audio_s": round(audio_s, 1),
        "cer": round(jiwer.cer([r for r, _ in pairs], [h for _, h in pairs]), 4),
        "wait_median_s": round(statistics.median(lat), 3),
        "rtf": round(compute / audio_s, 3),
        "hypotheses": {name: o["text"] for (name, _, _), o in zip(clips, outs)},
    }


def table(rows: list[dict]) -> str:
    lines = ["| lang | model | CER | wait after speech (median) | RTF |",
             "|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['lang']} | {r['model']} | {100 * r['cer']:.1f}% | "
                     f"{r['wait_median_s']:.3f} s | {r['rtf']:.3f} |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    import platformdirs

    data = Path(platformdirs.user_data_dir("VRCC"))
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--fleurs", type=Path, required=True)
    ap.add_argument("--langs", default="ja,ko,zh")
    ap.add_argument("--models", default="sense-voice-small,small,nemotron-560")
    ap.add_argument("--utterances", type=int, default=60)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--models-dir", type=Path, default=data / "models")
    ap.add_argument("--json", type=Path)
    args = ap.parse_args(argv)

    rows = []
    for lang in args.langs.split(","):
        clips = load_clips(args.fleurs, LANGS[lang][0], args.utterances)
        for model in args.models.split(","):
            print(f"{lang} {model}: {len(clips)} clips ...", flush=True)
            row = score(model, lang, clips,
                        build(model, lang, args.models_dir, args.threads))
            print(f"  CER {100 * row['cer']:.1f}%  wait {row['wait_median_s']:.3f}s"
                  f"  RTF {row['rtf']:.3f}", flush=True)
            rows.append(row)
    print(table(rows))
    if args.json:
        args.json.write_text(json.dumps(rows, ensure_ascii=False, indent=1),
                             encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
