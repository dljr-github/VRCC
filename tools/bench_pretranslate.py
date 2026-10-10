"""Measure what pre-translating speculative results saves (dev tool).

Feeds a session of short LibriSpeech test-clean utterances, separated by
random 0.6 to 1.5 s gaps of silence, through the real threaded Pipeline at
real-time pace with real STT and MT engines (no fakes but the chatbox and the
audio source). Each arm is the same audio with vrcc.core.pretranslate.ENABLED
off or on; arms alternate order across repeats. Reports the pipeline's own
finalize-to-submission latency (SessionStats.latency_samples), the speculative
reuse count and the pretranslate counters.

    python tools/bench_pretranslate.py --stt large-v3-turbo --mt nllb-600M-int8

Dev tool only -- not packaged, no test coverage.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import bench_stt  # noqa: E402
from tests.integration._harness import (  # noqa: E402
    FRAME, SAMPLE_RATE, _FakeChatbox, _FakeSource, _scratch_config_path,
)
from vrcc.core import hardware, pipeline_stats, pretranslate  # noqa: E402
from vrcc.core.bus import EventBus  # noqa: E402
from vrcc.core.config import ConfigStore, default_paths  # noqa: E402
from vrcc.core.engine_stack import build_engine_stack  # noqa: E402
from vrcc.translate.engine import TranslateEngine  # noqa: E402
from vrcc.translate.registry import MT_MODELS  # noqa: E402


class _FineClock:
    # time.monotonic ticks at 15.6 ms on Windows, as coarse as the effect
    # being measured. pipeline_stats only calls monotonic.
    monotonic = staticmethod(time.perf_counter)


def build_session(utts, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    parts = [np.zeros(SAMPLE_RATE, np.float32)]
    for _, audio, _ in utts:
        parts.append(audio)
        parts.append(np.zeros(int(rng.uniform(0.6, 1.5) * SAMPLE_RATE), np.float32))
    parts.append(np.zeros(2 * SAMPLE_RATE, np.float32))
    return np.concatenate(parts)


def run_arm(audio, stt, mt, enabled: bool, args) -> dict:
    pretranslate.ENABLED = enabled
    pipeline_stats.time = _FineClock
    calls: list[float] = []
    real_translate = type(mt).translate

    def counted(text, src, targets):
        t0 = time.perf_counter()
        try:
            return real_translate(mt, text, src, targets)
        finally:
            calls.append(time.perf_counter() - t0)

    mt.translate = counted
    store = ConfigStore(_scratch_config_path())
    store.load()
    store.config.osc.send_to_vrchat = False
    store.config.translate.targets = [args.target]
    bus = EventBus()
    stack = build_engine_stack(
        store, bus, default_paths(portable=False), stt_engine=stt, mt_engine=mt,
        chatbox=_FakeChatbox(), source=_FakeSource(), mute=None,
    )
    p = stack.pipeline
    p.start()
    p.set_captioning(True)
    try:
        for i in range(0, len(audio) - FRAME, FRAME):
            p._on_frame(audio[i : i + FRAME].copy())
            time.sleep(FRAME / SAMPLE_RATE)
        time.sleep(args.drain)
    finally:
        p.stop()
        del mt.translate
    s, c = p._session, p._pretranslate
    return {
        "lat": list(s.latency_samples), "reuse": s.reuse_count,
        "finals": s.final_calls, "mt_calls": len(calls),
        "mt_s": statistics.median(calls) if calls else 0.0,
        "offered": c.offered, "hit": c.hit, "miss": c.miss,
        "wasted": c.wasted, "dropped": c.dropped,
    }


def summarize(name, runs):
    lat = [x for r in runs for x in r["lat"]]
    q = np.percentile(lat, [50, 90]) * 1000 if lat else (float("nan"),) * 2
    n = len(lat)
    tot = lambda k: sum(r[k] for r in runs)  # noqa: E731
    row = (
        f"| {name} | {n} | {q[0]:.1f} | {q[1]:.1f} | {tot('reuse')}/{tot('finals') + tot('reuse')} "
        f"| {tot('offered')} | {tot('hit')} | {tot('miss')} | {tot('wasted')} "
        f"| {tot('dropped')} | {tot('mt_calls') / max(n, 1):.2f} |"
    )
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stt", default="large-v3-turbo")
    ap.add_argument("--mt", default="nllb-600M-int8")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--target", default="Japanese")
    ap.add_argument("--utterances", type=int, default=15)
    ap.add_argument("--max-seconds", type=float, default=6.0)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--drain", type=float, default=4.0)
    args = ap.parse_args()

    from vrcc.core.config import SttConfig, TranslateConfig
    from vrcc.stt import create_stt_engine

    hardware.setup_cuda_dlls()
    paths = default_paths(portable=False)
    root = bench_stt.ensure_dataset(bench_stt._default_data_dir())
    pool = [u for u in bench_stt.load_utterances(root, 200)
            if len(u[1]) / SAMPLE_RATE <= args.max_seconds]
    utts = pool[: args.utterances]
    print(f"{len(utts)} utterances, {sum(len(u[1]) for u in utts) / SAMPLE_RATE:.0f}s speech")

    bus = EventBus()
    stt = create_stt_engine(
        SttConfig(model=args.stt, device=args.device, source_language="English"),
        paths.models_dir / "whisper" / args.stt, bus,
    )
    stt.load(); stt.warm_up()
    spec = MT_MODELS[args.mt]
    mt = TranslateEngine(
        spec, paths.models_dir / "mt" / spec.id,
        TranslateConfig(model=args.mt, device=args.device, targets=[args.target]), bus,
    )
    mt.load(); mt.warm_up()

    results = {False: [], True: []}
    for rep in range(args.repeats):
        audio = build_session(utts, seed=rep)
        order = (False, True) if rep % 2 == 0 else (True, False)
        for enabled in order:
            r = run_arm(audio, stt, mt, enabled, args)
            results[enabled].append(r)
            print(f"rep {rep} {'on ' if enabled else 'off'}: {r}", flush=True)

    print("\n| arm | finals | median ms | p90 ms | spec reuse | offered | hit | miss "
          "| wasted | dropped | MT calls/final |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    print(summarize("off", results[False]))
    print(summarize("on", results[True]))


if __name__ == "__main__":
    main()
