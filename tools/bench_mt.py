"""Translation regression gate: drives the real TranslateEngine over a fixed
corpus and counts the failures a caption would show.

Run: python -m tools.bench_mt --models nllb-600M-int8 --arms 1.1:3 [--json out.json]
See benchmarks/MT.md for results and what the columns mean.

This is a gate, not a ranking. Quality averaged over target languages would
weight languages a given user may never use (see core/recommend_rank.py), so
every arm is judged on the same counted failures: numbers dropped, a dash
prefixed to a short fragment, and decoder loops. Never downloads a model.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

from tools.mt_digits import (
    ADDED, DROPPED, KEPT, SPELLED, classify, leading_dash, longest_run, merged, numbers,
)
from vrcc.core.bus import EventBus
from vrcc.core.config import TranslateConfig, default_paths
from vrcc.core.languages import Language, get
from vrcc.download.manager import DownloadManager
from vrcc.translate.engine import TranslateEngine
from vrcc.translate.registry import MT_MODELS

CORPUS = Path(__file__).with_name("mt_corpus.json")
SOURCES = {"en": "English", "ja": "Japanese", "zh": "Chinese Simplified",
           "ko": "Korean", "fr": "French", "es": "Spanish", "de": "German"}
DEFAULT_TARGETS = ["Japanese", "Chinese Simplified", "Korean", "French",
                   "Spanish", "Russian"]
# Fragments are what NLLB dresses as dialogue; "full" is everything else that
# reads as a complete utterance.
FRAGMENT_GROUPS = {"fragment"}
FULL_GROUPS = {"sentence", "number"}
CHATBOX_TARGETS = 3  # the app translates at most this many targets per call
LOOP_RUN = 4         # same threshold as tests/integration/test_translate_quality.py


def load_corpus(limit: int | None = None) -> list[dict]:
    data = json.loads(CORPUS.read_text(encoding="utf-8"))
    entries = [
        {"src": code, "group": group, "text": text, "spoken": group == "spoken"}
        for code, groups in data.items()
        for group, texts in groups.items()
        for text in texts
    ]
    return entries[::max(1, len(entries) // limit)][:limit] if limit else entries


def targets_for(src: Language, names: list[str]) -> list[Language]:
    wanted = names if src.display == "English" else [*names, "English"]
    return [get(n) for n in dict.fromkeys(wanted) if n != src.display]


def run_arm(model: str, rep: float, ngram: int, beam: int, device: str,
            entries: list[dict], names: list[str]) -> dict:
    bus = EventBus()
    spec = MT_MODELS[model]
    model_dir = DownloadManager(default_paths(portable=False).models_dir, bus
                                ).mt_model_dir(spec)
    cfg = TranslateConfig(model=model, device=device, beam_size=beam,
                          repetition_penalty=rep, no_repeat_ngram_size=ngram)
    engine = TranslateEngine(spec, model_dir, cfg, bus)
    engine.load()
    engine.warm_up()
    rows, call_ms = [], []
    for entry in entries:
        src = get(SOURCES[entry["src"]])
        tgts = targets_for(src, names)
        for i in range(0, len(tgts), CHATBOX_TARGETS):
            t0 = time.perf_counter()
            outs = engine.translate(entry["text"], src, tgts[i:i + CHATBOX_TARGETS])
            call_ms.append((time.perf_counter() - t0) * 1000)
            for display, out in outs:
                tgt = get(display)
                rows.append({
                    "group": entry["group"], "src": entry["text"],
                    "src_lang": entry["src"], "target": display, "out": out,
                    "verdict": classify(entry["text"], out, entry["src"],
                                        tgt.whisper, entry["spoken"]),
                    "merged": merged(entry["text"], out, entry["src"], tgt.whisper),
                    "has_numbers": bool(numbers(entry["text"], entry["src"],
                                                entry["spoken"])),
                    "loop": longest_run(out) >= LOOP_RUN,
                    "dash": leading_dash(entry["text"], out),
                })
    device_used = f"{engine._device}:{engine._compute_type}"
    engine.unload()
    return {"model": model, "rep": rep, "ngram": ngram, "beam": beam,
            "device": device_used, "rows": rows, "call_ms": call_ms}


def summarize(arm: dict) -> dict:
    rows = arm["rows"]
    tally = Counter()
    for r in rows:
        if r["has_numbers"] or r["verdict"] == ADDED:
            tally[r["verdict"]] += 1
    frag = [r for r in rows if r["group"] in FRAGMENT_GROUPS]
    full = [r for r in rows if r["group"] in FULL_GROUPS]
    ms = arm["call_ms"]
    return {
        "model": arm["model"], "rep": arm["rep"], "ngram": arm["ngram"],
        "beam": arm["beam"], "device": arm["device"], "pairs": len(rows),
        "digit_pairs": sum(r["has_numbers"] for r in rows),
        KEPT: tally[KEPT], SPELLED: tally[SPELLED], DROPPED: tally[DROPPED],
        ADDED: tally[ADDED],
        "merged": sum(r["merged"] and r["verdict"] == DROPPED for r in rows),
        "dash_frag": sum(r["dash"] for r in frag), "n_frag": len(frag),
        "dash_full": sum(r["dash"] for r in full), "n_full": len(full),
        "loops": sum(r["loop"] for r in rows),
        "ms_median": statistics.median(ms), "ms_mean": statistics.fmean(ms),
        "calls": len(ms),
    }


def table(summaries: list[dict]) -> str:
    head = ("| model | rep | ngram | beam | device | digit/all pairs | KEPT | SPELLED | "
            "DROPPED | merged | ADDED | dash frag | dash full | loops | ms/call med | "
            "ms/call mean |")
    lines = [head, "|" + "---|" * 16]
    for s in summaries:
        lines.append(
            f"| {s['model']} | {s['rep']} | {s['ngram']} | {s['beam']} | "
            f"{s['device']} | {s['digit_pairs']}/{s['pairs']} | {s[KEPT]} | "
            f"{s[SPELLED]} | {s[DROPPED]} | {s['merged']} | {s[ADDED]} | "
            f"{s['dash_frag']}/{s['n_frag']} "
            f"({100 * s['dash_frag'] / max(1, s['n_frag']):.1f}%) | "
            f"{s['dash_full']}/{s['n_full']} "
            f"({100 * s['dash_full'] / max(1, s['n_full']):.1f}%) | "
            f"{s['loops']} | {s['ms_median']:.0f} | {s['ms_mean']:.0f} |")
    return "\n".join(lines)


def hardware() -> str:
    gpu = ""
    try:
        gpu = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return f"{platform.processor() or platform.machine()}; GPU {gpu or 'none'}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    d = TranslateConfig()
    ap.add_argument("--models", default=d.model,
                    help="comma separated MT ids (default: the shipped model)")
    ap.add_argument("--arms", default=f"{d.repetition_penalty}:{d.no_repeat_ngram_size}",
                    help="comma separated REP:NGRAM pairs (default: shipped)")
    ap.add_argument("--beam", type=int, default=d.beam_size)
    ap.add_argument("--device", default=d.device)
    ap.add_argument("--targets", default=",".join(DEFAULT_TARGETS))
    ap.add_argument("--limit", type=int, help="subsample the corpus (smoke test)")
    ap.add_argument("--show", type=int, default=0,
                    help="print up to N offending pairs per arm")
    ap.add_argument("--json", type=Path, help="write the raw per-pair rows here")
    args = ap.parse_args(argv)

    entries = load_corpus(args.limit)
    names = [n.strip() for n in args.targets.split(",")]
    arms = [(float(a.split(":")[0]), int(a.split(":")[1])) for a in args.arms.split(",")]
    dm = DownloadManager(default_paths(portable=False).models_dir, EventBus())
    summaries, raw = [], []
    print(f"corpus: {len(entries)} sentences; hardware: {hardware()}")
    for model in (m.strip() for m in args.models.split(",")):
        if not dm.is_mt_downloaded(MT_MODELS[model]):
            print(f"skip {model}: not downloaded", file=sys.stderr)
            continue
        for rep, ngram in arms:
            arm = run_arm(model, rep, ngram, args.beam, args.device, entries, names)
            summaries.append(summarize(arm))
            raw.append(arm)
            bad = [r for r in arm["rows"]
                   if r["verdict"] in (DROPPED, ADDED) or r["loop"] or r["dash"]]
            for r in bad[:args.show]:
                print(f"  [{model} {rep}:{ngram}] {r['verdict']}"
                      f"{' LOOP' if r['loop'] else ''}{' DASH' if r['dash'] else ''}"
                      f" {r['src']!r} -> {r['target']}: {r['out']!r}")
    print(table(summaries))
    if args.json:
        args.json.write_text(json.dumps(raw, ensure_ascii=False, indent=1),
                             encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
