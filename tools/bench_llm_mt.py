"""LLM translation arm for the MT regression gate: HY-MT1.5 via llama-server.

Run (Windows, from the repo root, with a llama.cpp release unzipped):
    python -m tools.bench_llm_mt --server C:\\llama\\llama-server.exe ^
        --quants Q4_K_M,Q8_0 --sampling greedy,recommended --json llm.json

CPU only, in-process (needs ``pip install llama-cpp-python``, no server):
    python -m tools.bench_llm_mt --backend inproc --quants Q4_K_M --threads 8

Scores the same corpus, targets and failure columns as tools/bench_mt.py, so
the table lines up with the NLLB rows in benchmarks/MT.md. Each arm also
records the server's load time and the VRAM it took (nvidia-smi, before load
vs after warm-up; blank without an NVIDIA GPU).

Targets HY-MT1.5 does not support are skipped and counted in the "skipped"
line; the bench's default six are all supported.

The GGUF comes from ``--gguf`` or, if absent, is downloaded once from
``--hf-repo`` (the first file whose name contains the quant tag) into
``<models dir>/llm``. That is the only download this tool makes; get
llama-server yourself from https://github.com/ggml-org/llama.cpp/releases
(the CUDA 12 build for an NVIDIA GPU, otherwise the CPU or Vulkan build).

HY-MT1.5's licence excludes the EU, the UK and South Korea; see tools/llm_mt.py.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from tools.bench_mt import (
    DEFAULT_TARGETS, SOURCES, hardware, load_corpus, score_engine, summarize,
    table, targets_for,
)
from tools.mt_digits import ADDED, DROPPED
from tools.llm_mt import SAMPLING, LlamaServerTranslator, supports
from vrcc.core.config import default_paths
from vrcc.core.languages import get

DEFAULT_REPO = "tencent/HY-MT1.5-1.8B-GGUF"


def gpu_memory_used_mb() -> int | None:
    """Total VRAM in use on GPU 0 in MiB, or None without nvidia-smi."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip().splitlines()
        return int(out[0]) if out else None
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


def pick_gguf(files: list[str], quant: str) -> str:
    """The repo file for ``quant``: a .gguf whose name contains the tag,
    case-insensitively; the shortest such name when several match."""
    tag = quant.lower()
    hits = sorted(
        (f for f in files if f.lower().endswith(".gguf") and tag in f.lower()),
        key=len,
    )
    if not hits:
        ggufs = [f for f in files if f.lower().endswith(".gguf")]
        raise SystemExit(f"no {quant} GGUF in the repo; it has: {', '.join(ggufs)}")
    return hits[0]


def resolve_gguf(repo: str, quant: str, models_dir: Path) -> Path:
    from huggingface_hub import hf_hub_download, list_repo_files

    name = pick_gguf(list_repo_files(repo), quant)
    target = models_dir / "llm" / repo.replace("/", "--")
    return Path(hf_hub_download(repo, name, local_dir=str(target)))


def skipped_targets(entries: list[dict], names: list[str]) -> int:
    """Source/target pairs the bench leaves out because HY-MT cannot do them."""
    return sum(
        not supports(t)
        for e in entries
        for t in targets_for(get(SOURCES[e["src"]]), names)
    )


def run_arm(translator, label: str, sampling: str,
            entries: list[dict], names: list[str]) -> dict:
    before = gpu_memory_used_mb()
    translator.load()
    try:
        translator.warm_up()
        after = gpu_memory_used_mb()
        rows, call_ms = score_engine(translator.translate, entries, names, supports)
    finally:
        translator.unload()
    vram = after - before if before is not None and after is not None else None
    return {
        "model": label, "rep": SAMPLING[sampling]["repeat_penalty"], "ngram": 0,
        "beam": sampling, "device": translator.device, "rows": rows,
        "call_ms": call_ms, "load_s": translator.load_s, "vram_mb": vram,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--backend", choices=("server", "inproc"), default="server",
                    help="server: llama-server child process; inproc: "
                    "llama-cpp-python in this process, CPU only, greedy, all "
                    "targets in one batch (tools/llm_mt_inproc.py)")
    ap.add_argument("--server", type=Path,
                    help="path to llama-server(.exe); needed for --backend server")
    ap.add_argument("--threads", type=int, default=4,
                    help="CPU threads for --backend inproc")
    ap.add_argument("--gguf", type=Path,
                    help="a local GGUF; skips the download and --quants")
    ap.add_argument("--hf-repo", default=DEFAULT_REPO)
    ap.add_argument("--quants", default="Q4_K_M",
                    help="comma separated quant tags to fetch from --hf-repo")
    ap.add_argument("--sampling", default="greedy",
                    help=f"comma separated, from: {', '.join(SAMPLING)}")
    ap.add_argument("--gpu-layers", type=int, default=99,
                    help="layers to offload; 0 runs on the CPU")
    ap.add_argument("--parallel", type=int, default=3,
                    help="server slots; 3 matches the chatbox's target count")
    ap.add_argument("--server-arg", action="append", default=[],
                    help="extra llama-server argument (repeatable)")
    ap.add_argument("--targets", default=",".join(DEFAULT_TARGETS))
    ap.add_argument("--limit", type=int, help="subsample the corpus (smoke test)")
    ap.add_argument("--show", type=int, default=0,
                    help="print up to N offending pairs per arm")
    ap.add_argument("--json", type=Path, help="write the raw per-pair rows here")
    ap.add_argument("--log-dir", type=Path, default=Path("bench_results"),
                    help="llama-server logs land here, one per arm")
    args = ap.parse_args(argv)

    samplings = [s.strip() for s in args.sampling.split(",")]
    unknown = [s for s in samplings if s not in SAMPLING]
    if unknown:
        ap.error(f"unknown sampling: {', '.join(unknown)}")
    if args.backend == "server" and (args.server is None or not args.server.is_file()):
        ap.error("--backend server needs --server pointing at llama-server")
    if args.backend == "inproc" and samplings != ["greedy"]:
        ap.error("--backend inproc decodes greedily only")

    models_dir = default_paths(portable=False).models_dir
    if args.gguf:
        models = [(args.gguf.stem, args.gguf)]
    else:
        models = [
            (f"hy-mt1.5-1.8b-{q.strip()}", resolve_gguf(args.hf_repo, q.strip(), models_dir))
            for q in args.quants.split(",")
        ]

    entries = load_corpus(args.limit)
    names = [n.strip() for n in args.targets.split(",")]
    print(f"corpus: {len(entries)} sentences; hardware: {hardware()}")
    skipped = skipped_targets(entries, names)
    if skipped:
        print(f"skipped: {skipped} pairs with a target HY-MT1.5 does not support")
    args.log_dir.mkdir(parents=True, exist_ok=True)

    summaries, raw = [], []
    for label, gguf in models:
        for sampling in samplings:
            if args.backend == "inproc":
                from tools.llm_mt_inproc import InProcessTranslator

                translator = InProcessTranslator(
                    gguf, threads=args.threads, parallel=args.parallel)
            else:
                translator = LlamaServerTranslator(
                    args.server, gguf, gpu_layers=args.gpu_layers,
                    parallel=args.parallel, sampling=sampling,
                    extra_args=args.server_arg,
                    log_path=args.log_dir / f"llama-server__{label}__{sampling}.log",
                )
            arm = run_arm(translator, label, sampling, entries, names)
            summaries.append(summarize(arm))
            raw.append(arm)
            vram = "n/a" if arm["vram_mb"] is None else f"{arm['vram_mb']} MiB"
            print(f"  [{label} {sampling}] load {arm['load_s']:.1f}s, VRAM {vram}")
            bad = [r for r in arm["rows"]
                   if r["verdict"] in (DROPPED, ADDED) or r["loop"] or r["dash"]]
            for r in bad[:args.show]:
                print(f"  [{label} {sampling}] {r['verdict']}"
                      f"{' LOOP' if r['loop'] else ''}{' DASH' if r['dash'] else ''}"
                      f" {r['src']!r} -> {r['target']}: {r['out']!r}")
    print(table(summaries))
    if args.json:
        args.json.write_text(json.dumps(raw, ensure_ascii=False, indent=1),
                             encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
