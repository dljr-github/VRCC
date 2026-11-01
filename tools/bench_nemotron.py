"""Benchmark NVIDIA Nemotron 3.5 ASR Streaming 0.6B through sherpa-onnx (dev tool).

Run from the repo root (needs ``pip install -e .[bench] sherpa-onnx``):
    python tools/bench_nemotron.py --chunks 560,1120 --threads 4
    python tools/bench_nemotron.py --chunks 560 --samples   # ja/en sanity clips

Same LibriSpeech test-clean utterances as tools/bench_stt.py (English only, so
WER here says nothing about other languages; ``--samples`` decodes the
multilingual clips shipped in the model package instead). Each utterance runs
two ways:

- offline: the whole clip fed at once, then finished. Comparable to
  bench_stt's per-utterance latency for Whisper or Parakeet.
- live: the clip fed in ``--frame-ms`` pieces in order, decoding whatever is
  ready after each piece, the way a streaming engine inside VRCC would see
  the microphone. ``finalize`` is the time from the last piece to the final
  text: the wait a user has once they stop talking (VAD hangover aside).
  ``compute`` is the total decode time over the whole clip.

The model comes from sherpa-onnx's release page (one tarball per chunk size,
int8 graphs only are kept), into ``<models dir>/bench/nemotron/<package>``.
The PyPI sherpa-onnx wheel is CPU-only; ``--provider cuda`` needs sherpa's
CUDA build. Dev tool only, not packaged; tests/test_bench_nemotron.py covers
it with a stub recognizer.
"""

from __future__ import annotations

import argparse
import json
import shutil
import statistics
import sys
import tarfile
import time
import urllib.request
from pathlib import Path, PurePosixPath

import numpy as np
import platformdirs

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bench_data import SAMPLE_RATE, ensure_dataset, load_utterances  # noqa: E402

RELEASE = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models"
PACKAGE = "sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-{ms}ms-int8-2026-06-11"
MODEL_FILES = ("encoder.int8.onnx", "decoder.int8.onnx", "joiner.int8.onnx",
               "tokens.txt")


def package_name(chunk_ms: int) -> str:
    return PACKAGE.format(ms=chunk_ms)


def archive_url(chunk_ms: int) -> str:
    return f"{RELEASE}/{package_name(chunk_ms)}.tar.bz2"


def _keep(member_path: str) -> str | None:
    """Where an archive member lands inside the model dir, or None to skip.

    Paths from the archive are never used as given: model files land flat by
    basename and test clips under ``test_wavs/``, so ``../`` cannot escape.
    The fp32 graphs (and any external-data files next to them) are skipped;
    the int8 ones are what VRCC would ship."""
    parts = PurePosixPath(member_path).parts
    name = parts[-1] if parts else ""
    if not name or name.startswith("."):
        return None
    if "test_wavs" in parts[:-1] and name.endswith(".wav"):
        return f"test_wavs/{name}"
    if name.split(".")[0] in ("encoder", "decoder", "joiner") and ".int8." not in name:
        return None
    return name


def ensure_model(models_dir: Path, chunk_ms: int) -> Path:
    target = models_dir / "bench" / "nemotron" / package_name(chunk_ms)
    if all((target / f).is_file() for f in MODEL_FILES):
        return target
    partial = target.with_name(target.name + ".part")
    shutil.rmtree(partial, ignore_errors=True)
    (partial / "test_wavs").mkdir(parents=True)
    url = archive_url(chunk_ms)
    print(f"downloading {url} ...", flush=True)
    with urllib.request.urlopen(url, timeout=60) as resp, \
            tarfile.open(fileobj=resp, mode="r|bz2") as archive:
        for member in archive:
            dest = _keep(member.name) if member.isfile() else None
            source = archive.extractfile(member) if dest else None
            if source is not None:
                with open(partial / dest, "wb") as out:
                    shutil.copyfileobj(source, out)
    missing = [f for f in MODEL_FILES if not (partial / f).is_file()]
    if missing:
        raise RuntimeError(f"{url} lacked {', '.join(missing)}")
    shutil.rmtree(target, ignore_errors=True)
    partial.replace(target)
    return target


def build_recognizer(model_dir: Path, threads: int, provider: str):
    import sherpa_onnx

    return sherpa_onnx.OnlineRecognizer.from_transducer(
        encoder=str(model_dir / "encoder.int8.onnx"),
        decoder=str(model_dir / "decoder.int8.onnx"),
        joiner=str(model_dir / "joiner.int8.onnx"),
        tokens=str(model_dir / "tokens.txt"),
        num_threads=threads,
        provider=provider,
    )


def _stream(recognizer, language: str):
    stream = recognizer.create_stream()
    if language and language != "auto":
        stream.set_option("language", language)
    return stream


def _drain(recognizer, stream) -> None:
    while recognizer.is_ready(stream):
        recognizer.decode_stream(stream)


def decode_offline(recognizer, audio: np.ndarray, language: str,
                   tail_s: float, sr: int = SAMPLE_RATE) -> tuple[str, float]:
    """Whole clip at once (sherpa-onnx resamples a non-16 kHz ``sr``).
    Returns (text, seconds)."""
    t0 = time.perf_counter()
    stream = _stream(recognizer, language)
    stream.accept_waveform(sr, audio)
    stream.accept_waveform(sr, np.zeros(int(tail_s * sr), np.float32))
    stream.input_finished()
    _drain(recognizer, stream)
    text = recognizer.get_result(stream)
    return text.strip(), time.perf_counter() - t0


def decode_live(recognizer, audio: np.ndarray, language: str, tail_s: float,
                frame_ms: int) -> dict:
    """Clip fed piece by piece. Returns text, compute and finalize seconds,
    and how many times the partial text changed on the way."""
    step = max(1, SAMPLE_RATE * frame_ms // 1000)
    stream = _stream(recognizer, language)
    compute, partials, last = 0.0, 0, ""
    for start in range(0, len(audio), step):
        t0 = time.perf_counter()
        stream.accept_waveform(SAMPLE_RATE, audio[start:start + step])
        _drain(recognizer, stream)
        text = recognizer.get_result(stream)
        compute += time.perf_counter() - t0
        if text != last:
            partials, last = partials + 1, text
    t0 = time.perf_counter()
    stream.accept_waveform(SAMPLE_RATE, np.zeros(int(tail_s * SAMPLE_RATE), np.float32))
    stream.input_finished()
    _drain(recognizer, stream)
    text = recognizer.get_result(stream)
    finalize = time.perf_counter() - t0
    return {"text": text.strip(), "compute_s": compute + finalize,
            "finalize_s": finalize, "partials": partials}


def tail_for(tail_pad: float | None, chunk_ms: int) -> float:
    """Silence to append before finishing: ``tail_pad`` or two chunks."""
    return tail_pad if tail_pad is not None else 2 * chunk_ms / 1000


def _p90(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[min(int(len(ordered) * 0.9), len(ordered) - 1)]


def bench_chunk(recognizer, utts, language: str, tail_s: float,
                frame_ms: int) -> dict:
    import jiwer
    from whisper_normalizer.english import EnglishTextNormalizer

    for _ in range(2):  # untimed warm runs, as bench_stt does
        decode_offline(recognizer, utts[0][1], language, tail_s)
    off_text, off_s, live = [], [], []
    for i, (_, audio, _) in enumerate(utts):
        text, secs = decode_offline(recognizer, audio, language, tail_s)
        off_text.append(text)
        off_s.append(secs)
        live.append(decode_live(recognizer, audio, language, tail_s, frame_ms))
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(utts)}", flush=True)
    norm = EnglishTextNormalizer()
    refs = [norm(ref) for _, _, ref in utts]
    audio_s = sum(len(a) / SAMPLE_RATE for _, a, _ in utts)
    fin = [r["finalize_s"] for r in live]
    return {
        "n_utterances": len(utts),
        "audio_s": round(audio_s, 1),
        "wer_offline": round(jiwer.wer(refs, [norm(t) for t in off_text]), 4),
        "wer_live": round(jiwer.wer(refs, [norm(r["text"]) for r in live]), 4),
        "offline_median_s": round(statistics.median(off_s), 3),
        "offline_p90_s": round(_p90(off_s), 3),
        "offline_rtf": round(sum(off_s) / audio_s, 4),
        "live_rtf": round(sum(r["compute_s"] for r in live) / audio_s, 4),
        "finalize_median_s": round(statistics.median(fin), 3),
        "finalize_p90_s": round(_p90(fin), 3),
        "partials_mean": round(statistics.fmean(r["partials"] for r in live), 1),
        "hypotheses": dict(zip((u for u, _, _ in utts),
                               (r["text"] for r in live), strict=True)),
    }


def run_samples(recognizer, model_dir: Path, tail_s: float) -> None:
    """Decode the package's test clips, forced to the clip's language (from
    its file name, e.g. ``ja.wav``) and on auto."""
    import soundfile

    for wav in sorted((model_dir / "test_wavs").glob("*.wav")):
        audio, sr = soundfile.read(wav, dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        for lang in (wav.stem, "auto"):
            text, secs = decode_offline(recognizer, audio, lang, tail_s, sr)
            print(f"  {wav.name} [{lang}] {secs:.3f}s: {text}")


def table(rows: list[dict]) -> str:
    head = ("| chunk | provider | threads | WER offline | WER live | offline med | "
            "offline RTF | live RTF | finalize med | finalize p90 | partials |")
    lines = [head, "|" + "---|" * 11]
    for r in rows:
        lines.append(
            f"| {r['chunk_ms']} ms | {r['provider']} | {r['threads']} | "
            f"{100 * r['wer_offline']:.1f}% | {100 * r['wer_live']:.1f}% | "
            f"{r['offline_median_s']:.3f} s | {r['offline_rtf']:.3f} | "
            f"{r['live_rtf']:.3f} | {r['finalize_median_s']:.3f} s | "
            f"{r['finalize_p90_s']:.3f} s | {r['partials_mean']} |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    data_root = Path(platformdirs.user_data_dir("VRCC"))
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--chunks", default="560", help="comma separated chunk sizes, ms")
    ap.add_argument("--provider", default="cpu", choices=("cpu", "cuda"))
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--language", default="en", help="stream language, or auto")
    ap.add_argument("--utterances", type=int, default=100)
    ap.add_argument("--max-seconds", type=float, default=None,
                    help="keep only clips of at most N seconds (try 2.6)")
    ap.add_argument("--frame-ms", type=int, default=32,
                    help="piece size for the live run (VRCC's VAD frame is 32 ms)")
    ap.add_argument("--tail-pad", type=float, default=None,
                    help="seconds of silence appended before finishing "
                    "(default: two chunks; 0.5 s cut the last word of the "
                    "560 ms model's Vietnamese clip, 1.12 s did not)")
    ap.add_argument("--samples", action="store_true",
                    help="only decode the package's multilingual test clips")
    ap.add_argument("--out", type=Path, default=Path("bench_results/nemotron"))
    ap.add_argument("--data-dir", type=Path, default=data_root / "bench")
    ap.add_argument("--models-dir", type=Path, default=data_root / "models")
    args = ap.parse_args(argv)

    chunks = [int(c) for c in args.chunks.split(",")]
    if args.samples:
        for ms in chunks:
            model_dir = ensure_model(args.models_dir, ms)
            print(f"{package_name(ms)} [{args.provider}]")
            run_samples(build_recognizer(model_dir, args.threads, args.provider),
                        model_dir, tail_for(args.tail_pad, ms))
        return 0

    utts = load_utterances(ensure_dataset(args.data_dir), args.utterances,
                           args.max_seconds)
    if not utts:
        ap.error("no utterances at or under --max-seconds")
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for ms in chunks:
        model_dir = ensure_model(args.models_dir, ms)
        print(f"{ms} ms [{args.provider}, {args.threads} threads]: benchmarking ...",
              flush=True)
        t0 = time.perf_counter()
        recognizer = build_recognizer(model_dir, args.threads, args.provider)
        load_s = time.perf_counter() - t0
        tail_s = tail_for(args.tail_pad, ms)
        row = {"chunk_ms": ms, "provider": args.provider, "threads": args.threads,
               "language": args.language, "frame_ms": args.frame_ms,
               "tail_pad_s": tail_s, "max_seconds": args.max_seconds,
               "load_s": round(load_s, 2),
               **bench_chunk(recognizer, utts, args.language, tail_s,
                             args.frame_ms)}
        suffix = "" if args.max_seconds is None else f"__max{args.max_seconds:g}s"
        (args.out / f"nemotron-{ms}ms__{args.provider}{suffix}.json").write_text(
            json.dumps(row, indent=2, ensure_ascii=False), encoding="utf-8")
        rows.append(row)
        del recognizer
    print(table(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
