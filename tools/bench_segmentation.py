"""Replay labeled recordings through real Silero and compare segmentation.

Run: python -m tools.bench_segmentation manifest.json --output results.json
See benchmarks/SEGMENTATION.md for the manifest and measurement limits.
Requires the bench extra's soundfile; never downloads audio or models.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile
import soxr

from vrcc.audio.segmenter import FRAME, SegFinal, SegSpeechStart, Segmenter
from vrcc.audio.vad import StreamingVad
from vrcc.core.config import VadConfig

RATE = 16000


def variants(audio):
    """Signal stress cases, not emulations of particular microphone models."""
    yield "original", audio
    yield "gain_minus20db", audio * 0.1
    yield "gain_minus40db", audio * 0.01
    yield "clipped_plus20db", np.clip(audio * 10, -1, 1)
    gain = np.ones(len(audio), dtype=np.float32)
    gain[len(audio) // 2:] = 0.1
    yield "gain_step", audio * gain
    narrow = soxr.resample(soxr.resample(audio, RATE, 8000), 8000, RATE)
    yield "bandlimited_8khz", narrow[:len(audio)]
    rng = np.random.default_rng(20260925)
    level = max(float(np.sqrt(np.mean(audio.astype(np.float64) ** 2))), 0.001)
    yield "white_noise_10db", np.clip(
        audio + rng.normal(0, level / np.sqrt(10), len(audio)), -1, 1
    ).astype(np.float32)


def evaluate(audio, vad, configs):
    # Identical probabilities feed both policies. Prefix warms the VAD;
    # suffix gives both policies time to finalize. Neither is speech evidence.
    padded = np.concatenate([np.zeros(RATE), audio, np.zeros(2 * RATE)])
    padded = np.pad(padded, (0, (-len(padded)) % FRAME)).astype(np.float32)
    vad.reset()
    probabilities = [vad.prob(padded[i:i + FRAME])
                     for i in range(0, len(padded), FRAME)]
    results = {}
    for name, config in configs.items():
        sequence = iter(probabilities)
        seg = Segmenter(config, lambda _: next(sequence))
        starts, finals = [], []
        for i in range(0, len(padded), FRAME):
            for event in seg.process(padded[i:i + FRAME]):
                at = round((i + FRAME) / RATE - 1, 3)
                if isinstance(event, SegSpeechStart):
                    starts.append(at)
                elif isinstance(event, SegFinal):
                    finals.append(at)
        results[name] = {"starts_s": starts, "finals_s": finals,
                         "unfinished": seg.active}
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    entries = json.loads(args.manifest.read_text(encoding="utf-8"))
    if not isinstance(entries, list) or not entries:
        parser.error("manifest must be a nonempty list")
    configs = {
        "before_confirmation": VadConfig(speech_start_ms=0, min_utterance_ms=500),
        "candidate": VadConfig(),
    }
    vad = StreamingVad()
    rows = []
    groups = defaultdict(lambda: {"clips": 0, "missed_speech_clips": 0,
                                  "false_final_clips": 0, "unfinished_clips": 0})
    for entry in entries:
        if entry["kind"] not in ("speech", "nonspeech"):
            parser.error("kind must be speech or nonspeech")
        path = args.manifest.parent / entry["path"]
        audio, rate = soundfile.read(path, dtype="float32", always_2d=True)
        audio = audio.mean(axis=1)
        if not len(audio) or not np.isfinite(audio).all():
            parser.error(f"empty or nonfinite audio: {path}")
        if rate != RATE:
            audio = soxr.resample(audio, rate, RATE)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        for variant, signal in variants(audio):
            results = evaluate(signal, vad, configs)
            row = {"path": entry["path"], "sha256": digest,
                   "microphone": entry["microphone"], "language": entry["language"],
                   "kind": entry["kind"], "variant": variant, "results": results}
            rows.append(row)
            for policy, result in results.items():
                key = (entry["microphone"], entry["language"], variant, policy)
                group = groups[key]
                group["clips"] += 1
                group["missed_speech_clips"] += int(
                    entry["kind"] == "speech" and not result["finals_s"])
                group["false_final_clips"] += int(
                    entry["kind"] == "nonspeech" and bool(result["finals_s"]))
                group["unfinished_clips"] += int(result["unfinished"])
    report = {
        "configs": {k: v.model_dump() for k, v in configs.items()},
        "rows": rows,
        "groups": [{"microphone": k[0], "language": k[1], "variant": k[2],
                    "policy": k[3], **v} for k, v in sorted(groups.items())],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(rows)} recording variants to {args.output}")


if __name__ == "__main__":
    main()
