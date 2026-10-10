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
    python tools/bench_stt.py --device cuda --noise babble --snr 10
    python tools/bench_stt.py --device cuda --max-seconds 2.6 --gated --out bench_results/short
    python tools/bench_stt.py --export benchmarks/my-machine.json

The first run downloads the dataset (~350 MB) into ``<user data>/bench`` and
any missing models into the app's models dir (``--download-only`` does just
that). One JSON per (model, device) lands in ``--out`` (default
``bench_results/``); existing results are skipped (``--force`` re-runs).
``--noise babble|white --snr N`` overlays noise (babble: several corpus
speakers overlaid; white: Gaussian) on every utterance before transcribing,
at N dB SNR against that utterance's own level; see tools/bench_noise.py.
A noisy run's filename and utterance-set fingerprint both carry the noise
condition, so it never overwrites or gets exported alongside a clean run
(point ``--out`` at its own directory, e.g. ``bench_results/babble10``).
``--max-seconds N`` keeps only utterances at or under N seconds, the lengths
the app's VAD actually hands over (much shorter than whole LibriSpeech clips).
``--gated`` leaves the app's default quality gates in place and scores a
gated utterance as an empty hypothesis, which is what the user would see.
Both change the utterance-set fingerprint and the filename, so short or
gated rows never mix with the default table in an export. Every row records
the compute type the engine resolved, and its per-utterance hypotheses and
latencies stay in the result file for later paired comparisons.
``--export`` bundles every result with the machine's hardware info into one
shareable file -- see benchmarks/README.md for contributing yours. Dev tool
only -- not packaged; tests/test_bench_stt.py covers it with a stub engine.
"""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import platformdirs

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import bench_noise  # noqa: E402
from bench_data import (  # noqa: E402
    SAMPLE_RATE,
    ensure_dataset,
    load_utterances,
    utterance_set_id,
)
from vrcc.core.bus import EventBus  # noqa: E402
from vrcc.core.config import SttConfig  # noqa: E402
from vrcc.core.events import EngineStateChanged  # noqa: E402
from vrcc.download.manager import DownloadManager  # noqa: E402
from vrcc.stt import create_stt_engine  # noqa: E402
from vrcc.stt.registry import WHISPER_MODELS  # noqa: E402

# Untimed transcriptions of a real utterance after warm_up(), so first-call
# kernel autotuning never lands in a timed measurement.
_UNTIMED_WARM_RUNS = 2


def _default_models_dir() -> Path:
    return Path(platformdirs.user_data_dir("VRCC")) / "models"


def _default_data_dir() -> Path:
    return Path(platformdirs.user_data_dir("VRCC")) / "bench"


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
    info into one contribution-ready JSON at ``target`` (hypotheses and
    per-utterance latencies dropped: they're per-run data, not metrics)."""
    records = []
    for path in sorted(results_dir.glob("*__*.json")):
        record = json.loads(path.read_text())
        record.pop("hypotheses", None)
        record.pop("latencies_s", None)
        records.append(record)
    if not records:
        print(f"no results in {results_dir}; run a benchmark first")
        return 1
    sets = {record.get("utterance_set") for record in records}
    if len(sets) > 1:
        print(
            "refusing to export: results were measured on different utterance "
            "sets (sampling, beam or noise settings differ across the files "
            f"in {results_dir}); re-run the odd ones with --force, delete "
            "them, or export a noisy run from its own --out dir, so every "
            "record used the same set"
        )
        return 1

    # Sets already agree (checked above), so every record shares one noise
    # condition; None/None for a clean run.
    noise_type, noise_snr_db = next(
        iter({(r.get("noise_type"), r.get("noise_snr_db")) for r in records})
    )
    dataset = "LibriSpeech test-clean"
    bound = next(iter({r.get("max_seconds") for r in records}))
    if bound is not None:
        dataset += f" (utterances <= {bound:g} s)"
    if {r.get("gates") for r in records} == {"app"}:
        dataset += ", app quality gates applied"
    if noise_type:
        dataset += f" + {noise_type} noise @ {noise_snr_db:g} dB SNR"

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
                "dataset": dataset,
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
    noise_type: str | None = None,
    noise_snr_db: float | None = None,
    max_seconds: float | None = None,
    gated_scoring: bool = False,
) -> dict:
    """Load one model on ``device``, transcribe every utterance, and return
    the timing/WER record. The engine is fed pre-segmented utterances exactly
    as the app's VAD hands them over. ``utts`` already carries noise if the
    caller applied it (via ``bench_noise.apply_noise``); ``noise_type`` and
    ``noise_snr_db`` are recorded here, not applied.

    Unless ``gated_scoring`` is set, all five quality gates are opened (whisper's avg_logprob, no_speech and
    compression_ratio; parakeet_avg_logprob and sensevoice_avg_logprob for
    those onnx-asr backends) so WER measures what the model recognized, not
    what a gate would suppress. Utterances the default gates *would* drop
    are counted in ``gated`` instead, against whichever threshold fits,
    except compression_ratio (no SttResult field): that drop is scored text only.

    With ``gated_scoring`` the app's default gates stay in place and a
    dropped utterance is an empty hypothesis; ``gated`` is then None because
    drops are indistinguishable from empty results and all land in ``empty``.
    """
    import jiwer
    from whisper_normalizer.english import EnglishTextNormalizer

    bus = EventBus()
    events: list[tuple[str, str | None]] = []
    bus.subscribe(
        EngineStateChanged, lambda e: events.append((e.state, e.detail))
    )

    defaults = SttConfig()
    open_gates = {} if gated_scoring else {
        "avg_logprob_gate": -1e9,
        "no_speech_gate": 1.0,
        "compression_ratio_gate": 1e9,
        "parakeet_avg_logprob_gate": -1e9,
        "sensevoice_avg_logprob_gate": -1e9,
    }
    cfg = SttConfig(model=model_id, device=device, beam_size=beam, **open_gates)
    engine = create_stt_engine(cfg, models_dir / "whisper" / model_id, bus)

    t0 = time.perf_counter()
    engine.load()
    load_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    engine.warm_up()
    warmup_s = time.perf_counter() - t0
    for _ in range(_UNTIMED_WARM_RUNS):
        engine.transcribe(utts[0][1])

    # onnx_asr/sensevoice report confidence on their own tighter scales;
    # whisper's -0.8 would misclassify almost every one of their utterances.
    backend = WHISPER_MODELS[model_id].backend
    logprob_gate = {
        "onnx_asr": defaults.parakeet_avg_logprob_gate,
        "sensevoice": defaults.sensevoice_avg_logprob_gate,
    }.get(backend, defaults.avg_logprob_gate)

    hyps: list[str] = []
    latencies: list[float] = []
    empty = 0
    gated = 0
    for i, (_, audio, _) in enumerate(utts):
        t0 = time.perf_counter()
        result = engine.transcribe(audio)
        latencies.append(time.perf_counter() - t0)
        if result is None:
            empty += 1  # open gates: no text at all; gated scoring: or dropped
        elif not gated_scoring:
            if (
                result.avg_logprob < logprob_gate
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
        "compute_requested": cfg.compute_type,
        "beam": beam,
        "gates": "app" if gated_scoring else "open",
        "max_seconds": max_seconds,
        "utterance_set": utterance_set_id(
            utts,
            (noise_type, noise_snr_db) if noise_type else None,
            max_seconds,
            gated_scoring,
        ),
        "noise_type": noise_type,
        "noise_snr_db": noise_snr_db,
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
        # Utterances the app's default gates would suppress (no caption
        # shown), avg_logprob/no_speech only: compression_ratio isn't on
        # SttResult, so a dropped repetition loop lands in neither counter.
        "gated": None if gated_scoring else gated,
        "empty": empty,
        "engine_events": [e for e in events if e[0] != "loading"],
        "hypotheses": dict(
            zip((utt_id for utt_id, _, _ in utts), hyps, strict=True)
        ),
        "latencies_s": {
            utt_id: round(lat, 4)
            for (utt_id, _, _), lat in zip(utts, latencies, strict=True)
        },
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
    parser.add_argument(
        "--noise", choices=("babble", "white"), default=None,
        help="overlay noise on every utterance before transcribing (babble: "
        "corpus speakers overlaid; white: Gaussian); requires --snr",
    )
    parser.add_argument(
        "--max-seconds", type=float, default=None, metavar="N",
        help="keep only utterances of at most N seconds (the app's VAD "
        "clips are far shorter than LibriSpeech's; try 2.6); pair with a large --utterances to take "
        "every short clip",
    )
    parser.add_argument(
        "--gated", action="store_true",
        help="score with the app's default quality gates applied; a gated "
        "utterance counts as an empty hypothesis",
    )
    parser.add_argument(
        "--snr", type=float, default=None,
        help="signal-to-noise ratio in dB for --noise, against each "
        "utterance's own level (e.g. 10)",
    )
    args = parser.parse_args()

    if args.export:
        return export_results(args.out, args.export)

    if args.utterances < 1:
        parser.error("--utterances must be at least 1")
    if args.beam < 1:
        parser.error("--beam must be at least 1")
    if args.max_seconds is not None and args.max_seconds <= 0:
        parser.error("--max-seconds must be positive")
    if bool(args.noise) != (args.snr is not None):
        parser.error("--noise and --snr must be given together")

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

    utts = load_utterances(dataset_root, args.utterances, args.max_seconds)
    if not utts:
        parser.error("no utterances at or under --max-seconds")
    if args.noise:
        print(f"synthesizing {args.noise} noise at {args.snr:g} dB SNR ...", flush=True)
        utts = bench_noise.apply_noise(utts, dataset_root, args.noise, args.snr)
    args.out.mkdir(parents=True, exist_ok=True)
    meta_path = args.out / "meta.json"
    if not meta_path.is_file():
        meta_path.write_text(json.dumps(machine_info(), indent=2))

    noise_desc = (args.noise, args.snr) if args.noise else None
    suffix = "" if args.beam == 1 else f"__beam{args.beam}"
    suffix += "" if not args.noise else f"__{args.noise}{args.snr:g}"
    suffix += "" if args.max_seconds is None else f"__max{args.max_seconds:g}s"
    suffix += "__gated" if args.gated else ""
    set_id = utterance_set_id(utts, noise_desc, args.max_seconds, args.gated)
    for model_id in model_ids:
        out_path = args.out / f"{model_id}__{args.device}{suffix}.json"
        if out_path.is_file() and not args.force:
            # A result from a different utterance set is stale, not done:
            # keeping it would let --export mix incomparable runs.
            existing = json.loads(out_path.read_text())
            if existing.get("utterance_set") == set_id:
                print(f"{model_id} [{args.device}]: exists, skipping", flush=True)
                continue
            print(
                f"{model_id} [{args.device}]: exists for a different "
                f"utterance set, re-running",
                flush=True,
            )
        print(f"{model_id} [{args.device}]: benchmarking ...", flush=True)
        result = bench_model(
            model_id, args.device, utts, args.models_dir, beam=args.beam,
            noise_type=args.noise, noise_snr_db=args.snr,
            max_seconds=args.max_seconds, gated_scoring=args.gated,
        )
        out_path.write_text(json.dumps(result, indent=2))
        print(
            f"  n={result['n_utterances']} wer={result['wer']:.3f} rtf={result['rtf']:.3f} "
            f"median={result['latency_median_s']:.3f}s load={result['load_s']:.1f}s "
            f"on {result['resolved_device']}:{result['compute']}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
