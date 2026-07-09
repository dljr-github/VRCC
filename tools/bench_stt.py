"""Benchmark every VRCC STT model for speed and accuracy (dev tool).

Measures, per model and device, through the app's own engine path
(``create_stt_engine`` -> load / warm_up / transcribe with ``SttConfig``
defaults, i.e. the Latency profile with the source language set to English):
model load time, per-utterance latency, real-time factor, and WER on a
LibriSpeech test-clean subset (Whisper-style English text normalization on
references and hypotheses).

Run from the repo root:
    python tools/bench_stt.py --device cuda
    python tools/bench_stt.py --device cpu
    python tools/bench_stt.py --export benchmarks/my-machine.json

The first run downloads the dataset (~350 MB) into ``<user data>/bench`` and
any missing models into the app's models dir (``--download-only`` does just
that). One JSON per (model, device) lands in ``--out`` (default
``bench_results/``); existing results are skipped (``--force`` re-runs).
``--export`` bundles every result with the machine's hardware info into one
shareable file -- see benchmarks/README.md for contributing yours. Dev tool
only -- not packaged, no test coverage.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import shutil
import statistics
import sys
import tarfile
import time
import urllib.request
from pathlib import Path

import numpy as np
import platformdirs
import soundfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vrcc.core.bus import EventBus  # noqa: E402
from vrcc.core.config import SttConfig  # noqa: E402
from vrcc.core.events import EngineStateChanged  # noqa: E402
from vrcc.download.manager import DownloadManager  # noqa: E402
from vrcc.stt import create_stt_engine  # noqa: E402
from vrcc.stt.registry import WHISPER_MODELS  # noqa: E402

DATASET_URL = "https://www.openslr.org/resources/12/test-clean.tar.gz"
SAMPLE_RATE = 16000

# Untimed transcriptions of a real utterance after warm_up(), so first-call
# kernel autotuning never lands in a timed measurement.
_UNTIMED_WARM_RUNS = 2


def _default_models_dir() -> Path:
    return Path(platformdirs.user_data_dir("VRCC")) / "models"


def _default_data_dir() -> Path:
    return Path(platformdirs.user_data_dir("VRCC")) / "bench"


def ensure_dataset(data_dir: Path) -> Path:
    """Download + extract LibriSpeech test-clean once; return its root.

    Interruption-safe: the archive downloads to a .part file renamed only
    when complete, extraction is trusted only once a sentinel exists, and a
    corrupt archive is deleted with a retry hint instead of failing every
    later run."""
    root = data_dir / "LibriSpeech" / "test-clean"
    sentinel = data_dir / "LibriSpeech" / ".extracted"
    if root.is_dir() and sentinel.is_file():
        return root
    data_dir.mkdir(parents=True, exist_ok=True)
    archive = data_dir / "test-clean.tar.gz"
    if not archive.is_file():
        partial = archive.parent / (archive.name + ".part")
        print(f"downloading {DATASET_URL} ...", flush=True)
        with urllib.request.urlopen(DATASET_URL) as resp, open(partial, "wb") as out:
            done = 0
            while chunk := resp.read(1 << 20):
                out.write(chunk)
                done += len(chunk)
                if done % (50 << 20) < (1 << 20):
                    print(f"  {done >> 20} MB", flush=True)
        partial.replace(archive)
    print("extracting ...", flush=True)
    tree = data_dir / "LibriSpeech"
    if tree.exists():
        # No sentinel means the tree can't be trusted (interrupted extract, or
        # one from before the sentinel existed). The tar's members are
        # read-only, so restore writability or rmtree fails on Windows.
        for path in tree.rglob("*"):
            if path.is_file():
                path.chmod(0o644)
        shutil.rmtree(tree)
    try:
        with tarfile.open(archive) as tar:
            try:
                tar.extractall(data_dir, filter="data")
            except TypeError:
                # filter= needs 3.10.12+/3.11.4+/3.12; older interpreters in
                # the supported >=3.10 range extract without it.
                tar.extractall(data_dir)
    except tarfile.ReadError as exc:
        archive.unlink()
        raise RuntimeError(
            f"{archive} was corrupt (interrupted download?); deleted it, "
            "run this command again"
        ) from exc
    sentinel.write_text("")
    return root


def load_utterances(root: Path, n: int) -> list[tuple[str, np.ndarray, str]]:
    """N (id, mono float32 16 kHz audio, reference) tuples, evenly spaced
    over the id-sorted corpus so every run picks the same speaker-diverse
    set (a plain prefix would over-represent the first speakers)."""
    refs: dict[str, str] = {}
    for trans in sorted(root.rglob("*.trans.txt")):
        for line in trans.read_text(encoding="utf-8").splitlines():
            utt_id, _, text = line.partition(" ")
            refs[utt_id] = text.strip()
    ids = sorted(refs)
    n = min(n, len(ids))
    picked = [ids[i * len(ids) // n] for i in range(n)]

    utts = []
    for utt_id in picked:
        spk, chap, _ = utt_id.split("-")
        path = root / spk / chap / f"{utt_id}.flac"
        audio, sr = soundfile.read(path, dtype="float32", always_2d=False)
        if sr != SAMPLE_RATE:
            raise RuntimeError(f"{path} has sample rate {sr}, expected {SAMPLE_RATE}")
        utts.append((utt_id, audio, refs[utt_id]))
    return utts


def machine_info() -> dict:
    import os
    import platform

    from vrcc.core import hardware

    info = {"os": platform.platform(), "python": sys.version.split()[0]}
    try:
        import winreg

        key = winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE,
            r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
        )
        info["cpu"] = winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
    except (OSError, ModuleNotFoundError):
        info["cpu"] = platform.processor() or "unknown"
    info["cpu_threads"] = os.cpu_count()
    info["gpus"] = hardware.device_names()
    info["driver"] = hardware.driver_version()
    vram = hardware.total_vram_bytes()
    info["vram_gb"] = round(vram / 1024**3, 1) if vram else None
    return info


def export_results(results_dir: Path, target: Path) -> int:
    """Bundle every per-model result in ``results_dir`` plus fresh machine
    info into one contribution-ready JSON at ``target`` (hypotheses dropped:
    they're per-run debug data, not metrics)."""
    records = []
    for path in sorted(results_dir.glob("*__*.json")):
        record = json.loads(path.read_text())
        record.pop("hypotheses", None)
        records.append(record)
    if not records:
        print(f"no results in {results_dir}; run a benchmark first")
        return 1
    sets = {record.get("utterance_set") for record in records}
    if len(sets) > 1:
        print(
            "refusing to export: results were measured on different utterance "
            "sets; re-run the odd ones with --force (or delete them) so every "
            "record used the same set"
        )
        return 1

    # Prefer the machine info captured when the benchmarks ran; a fresh
    # snapshot could describe a different driver than the one measured.
    meta_path = results_dir / "meta.json"
    machine = (
        json.loads(meta_path.read_text()) if meta_path.is_file() else machine_info()
    )

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(
            {
                "schema": 1,
                "dataset": "LibriSpeech test-clean",
                "machine": machine,
                "results": records,
            },
            indent=2,
        )
        + "\n"
    )
    devices = {r["device"] for r in records}
    beams = {r.get("beam", 1) for r in records}
    print(
        f"wrote {target} ({len(records)} results, "
        f"devices: {', '.join(sorted(devices))}, "
        f"beams: {', '.join(str(b) for b in sorted(beams))})"
    )
    return 0


def _utterance_set_id(utts: list[tuple[str, np.ndarray, str]]) -> str:
    """Short fingerprint of exactly which utterances a record measured, so
    skip/export logic can refuse to mix runs whose sets differ (a sampling
    change keeps the count at N while changing the members)."""
    joined = "\n".join(utt_id for utt_id, _, _ in utts)
    return hashlib.sha256(joined.encode()).hexdigest()[:12]


def ensure_models(model_ids: list[str], models_dir: Path) -> None:
    dm = DownloadManager(models_dir, EventBus())
    for model_id in model_ids:
        if dm.is_whisper_downloaded(model_id):
            print(f"{model_id}: present", flush=True)
        else:
            print(f"{model_id}: downloading ...", flush=True)
            dm.ensure_whisper(model_id)


def bench_model(
    model_id: str,
    device: str,
    utts: list[tuple[str, np.ndarray, str]],
    models_dir: Path,
    beam: int = 1,
) -> dict:
    """Load one model on ``device``, transcribe every utterance, and return
    the timing/WER record. The engine is fed pre-segmented utterances exactly
    as the app's VAD hands them over.

    The quality gates are opened for the run (``avg_logprob``/``no_speech``)
    so WER measures what the model recognized, not what the gates suppressed:
    a gated utterance returns no text, and scoring that as a full sentence of
    deletions would blame the model for a policy decision. Utterances the
    app's default gates *would* have dropped are counted in ``gated`` instead.
    """
    import jiwer
    from whisper_normalizer.english import EnglishTextNormalizer

    bus = EventBus()
    events: list[tuple[str, str | None]] = []
    bus.subscribe(
        EngineStateChanged, lambda e: events.append((e.state, e.detail))
    )

    defaults = SttConfig()
    cfg = SttConfig(
        model=model_id,
        device=device,
        beam_size=beam,
        avg_logprob_gate=-1e9,
        no_speech_gate=1.0,
    )
    engine = create_stt_engine(cfg, models_dir / "whisper" / model_id, bus)

    t0 = time.perf_counter()
    engine.load()
    load_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    engine.warm_up()
    warmup_s = time.perf_counter() - t0
    for _ in range(_UNTIMED_WARM_RUNS):
        engine.transcribe(utts[0][1])

    hyps: list[str] = []
    latencies: list[float] = []
    empty = 0
    gated = 0
    for i, (_, audio, _) in enumerate(utts):
        t0 = time.perf_counter()
        result = engine.transcribe(audio)
        latencies.append(time.perf_counter() - t0)
        if result is None:
            empty += 1  # no text at all: the gates are open, so this is real
        else:
            if (
                result.avg_logprob < defaults.avg_logprob_gate
                or result.no_speech_prob > defaults.no_speech_gate
            ):
                gated += 1
        hyps.append(result.text if result else "")
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(utts)}", flush=True)

    resolved_device = engine._device
    compute = getattr(engine, "_compute_type", None) or (
        WHISPER_MODELS[model_id].quantization or "fp32"
    )
    engine.unload()
    del engine
    gc.collect()

    normalize = EnglishTextNormalizer()
    refs_norm = [normalize(ref) for _, _, ref in utts]
    hyps_norm = [normalize(hyp) for hyp in hyps]
    audio_s = sum(len(audio) / SAMPLE_RATE for _, audio, _ in utts)
    proc_s = sum(latencies)

    return {
        "model": model_id,
        "device": device,
        "resolved_device": resolved_device,
        "compute": compute,
        "beam": beam,
        "utterance_set": _utterance_set_id(utts),
        "size_mb": WHISPER_MODELS[model_id].size_mb,
        "load_s": round(load_s, 2),
        "warmup_s": round(warmup_s, 2),
        "n_utterances": len(utts),
        "audio_s": round(audio_s, 1),
        "proc_s": round(proc_s, 1),
        "rtf": round(proc_s / audio_s, 4),
        "latency_median_s": round(statistics.median(latencies), 3),
        "latency_p90_s": round(
            sorted(latencies)[min(int(len(latencies) * 0.9), len(latencies) - 1)], 3
        ),
        "wer": round(jiwer.wer(refs_norm, hyps_norm), 4),
        # Utterances the app's default quality gates would have suppressed
        # (no caption shown). Scored text either way: see the docstring.
        "gated": gated,
        "empty": empty,
        "engine_events": [e for e in events if e[0] != "loading"],
        "hypotheses": dict(
            zip((utt_id for utt_id, _, _ in utts), hyps, strict=True)
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument(
        "--models", help="comma-separated model ids (default: every registry id)"
    )
    parser.add_argument("--utterances", type=int, default=100)
    parser.add_argument(
        "--beam", type=int, default=1,
        help="STT beam size; whisper models only (the Quality preset uses 5)",
    )
    parser.add_argument("--out", type=Path, default=Path("bench_results"))
    parser.add_argument("--data-dir", type=Path, default=_default_data_dir())
    parser.add_argument("--models-dir", type=Path, default=_default_models_dir())
    parser.add_argument(
        "--download-only", action="store_true",
        help="fetch the dataset and any missing models, then exit",
    )
    parser.add_argument(
        "--force", action="store_true", help="re-run models with existing results"
    )
    parser.add_argument(
        "--export", type=Path, metavar="FILE",
        help="bundle --out results + hardware info into FILE and exit",
    )
    args = parser.parse_args()

    if args.export:
        return export_results(args.out, args.export)

    if args.utterances < 1:
        parser.error("--utterances must be at least 1")
    if args.beam < 1:
        parser.error("--beam must be at least 1")

    model_ids = (
        args.models.split(",") if args.models else list(WHISPER_MODELS)
    )
    unknown = [m for m in model_ids if m not in WHISPER_MODELS]
    if unknown:
        parser.error(f"unknown model ids: {', '.join(unknown)}")

    dataset_root = ensure_dataset(args.data_dir)
    ensure_models(model_ids, args.models_dir)
    if args.download_only:
        return 0
    if not args.device:
        parser.error("--device is required to benchmark")

    if args.device == "cuda":
        from vrcc.core import hardware

        hardware.setup_cuda_dlls()

    utts = load_utterances(dataset_root, args.utterances)
    args.out.mkdir(parents=True, exist_ok=True)
    meta_path = args.out / "meta.json"
    if not meta_path.is_file():
        meta_path.write_text(json.dumps(machine_info(), indent=2))

    suffix = "" if args.beam == 1 else f"__beam{args.beam}"
    for model_id in model_ids:
        out_path = args.out / f"{model_id}__{args.device}{suffix}.json"
        if out_path.is_file() and not args.force:
            # A result from a different utterance set is stale, not done:
            # keeping it would let --export mix incomparable runs.
            existing = json.loads(out_path.read_text())
            if existing.get("utterance_set") == _utterance_set_id(utts):
                print(f"{model_id} [{args.device}]: exists, skipping", flush=True)
                continue
            print(
                f"{model_id} [{args.device}]: exists for a different "
                f"utterance set, re-running",
                flush=True,
            )
        print(f"{model_id} [{args.device}]: benchmarking ...", flush=True)
        result = bench_model(
            model_id, args.device, utts, args.models_dir, beam=args.beam
        )
        out_path.write_text(json.dumps(result, indent=2))
        print(
            f"  wer={result['wer']:.3f} rtf={result['rtf']:.3f} "
            f"median={result['latency_median_s']:.3f}s load={result['load_s']:.1f}s "
            f"on {result['resolved_device']}:{result['compute']}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
