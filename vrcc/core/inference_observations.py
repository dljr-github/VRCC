"""Bounded numeric observations from real calls, not controlled benchmarks.

Only the latest configuration per stage is retained. Configuration digests
are private invalidation keys; reports never include prompts, text, audio,
paths, or arbitrary configuration values. Reading a report never borrows an
engine lock or runs a model. No observations feed the model recommender.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import threading
import time
from collections import deque
from dataclasses import dataclass

import numpy as np

from vrcc.stt.registry import WHISPER_MODELS
from vrcc.translate.registry import MT_MODELS
from vrcc.core import languages

WINDOW_SIZE = 128
logger = logging.getLogger("vrcc.core.pipeline")
_PRECISIONS = {
    "int8", "int8_float16", "int8_float32", "int8_bfloat16", "int16",
    "float16", "float32", "bfloat16",
}


@dataclass(frozen=True)
class Context:
    key: tuple
    metadata: dict


def context(engine, cfg, stage: str) -> Context:
    """Snapshot identity while the caller holds the engine slot, when used
    around inference. Requested settings are labeled as such: a live config
    can change before the corresponding engine is rebuilt."""
    registry = WHISPER_MODELS if stage == "stt" else MT_MODELS
    model_id = getattr(getattr(engine, "_spec", None), "id", None)
    if model_id is None:
        model_id = getattr(getattr(engine, "_model_dir", None), "name", None)
    model_id = model_id if model_id in registry else "unknown"
    spec = registry.get(model_id)
    backend = getattr(spec, "backend", "ctranslate2") if spec else "unknown"
    device = getattr(engine, "_device", None)
    device = device if device in {"cpu", "cuda"} else "unknown"
    precision = getattr(engine, "_compute_type", None)
    if precision is None:
        precision = getattr(getattr(engine, "_spec", None), "quantization", None)
    precision = precision if precision in _PRECISIONS else "unknown"
    # Hash only for equality; neither serialized config nor digest is exported.
    digest = hashlib.sha256(json.dumps(
        cfg.model_dump(), sort_keys=True, default=repr,
    ).encode()).hexdigest()
    metadata = {"model_id": model_id, "backend": backend,
                "requested_model_id": cfg.model if cfg.model in registry else "unknown",
                "device": device, "precision": precision,
                "requested_beam_size": cfg.beam_size}
    if stage == "stt":
        metadata.update(requested_cpu_threads=cfg.cpu_threads,
                        requested_num_workers=cfg.num_workers)
    else:
        metadata.update(requested_intra_threads=cfg.intra_threads,
                        requested_inter_threads=cfg.inter_threads)
    return Context((id(engine), digest, model_id, device, precision), metadata)


def _distribution(values: list[float]) -> dict | None:
    if not values:
        return None
    p50, p95 = np.percentile(values, [50, 95])
    return {"p50": float(p50), "p95": float(p95), "max": max(values)}


class InferenceObservations:
    """One bounded window each for final STT and MT, protected across workers.

    dispatch_wait_s includes producer backpressure and queue residence, ending
    before attempting the engine lock. service_s excludes that lock wait.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._generation = 0
        self._stages: dict[str, tuple[Context, deque]] = {}

    @property
    def generation(self) -> int:
        with self._lock:
            return self._generation

    def reset(self) -> None:
        with self._lock:
            self._generation += 1
            self._stages.clear()

    def record(self, stage: str, ctx: Context, *, service_s: float,
               lock_wait_s: float, dispatch_wait_s: float | None = None,
               audio_s: float | None = None, input_chars: int | None = None,
               target_count: int | None = None, generation: int | None = None) -> None:
        values = (service_s, lock_wait_s, dispatch_wait_s, audio_s, input_chars)
        if any(v is not None and (not math.isfinite(v) or v < 0) for v in values):
            return
        if audio_s is not None and audio_s <= 0:
            return
        with self._lock:
            if generation is not None and generation != self._generation:
                return
            old = self._stages.get(stage)
            if old is None or old[0] != ctx:
                self._stages[stage] = (ctx, deque(maxlen=WINDOW_SIZE))
            self._stages[stage][1].append({
                "service_s": service_s, "lock_wait_s": lock_wait_s,
                "dispatch_wait_s": dispatch_wait_s, "audio_s": audio_s,
                "input_chars": input_chars, "target_count": target_count,
            })

    def report(self, contexts: dict[str, Context]) -> dict:
        result = {"kind": "runtime_observations", "schema_version": 1,
                  "window_size": WINDOW_SIZE, "stt": None, "mt": None}
        with self._lock:
            for stage, current in contexts.items():
                stored = self._stages.get(stage)
                if stored is None:
                    continue
                ctx, samples = stored
                if ctx.key != current.key:
                    del self._stages[stage]
                    continue
                values = list(samples)
                summary = {"metadata": dict(ctx.metadata), "count": len(values)}
                for field in ("service_s", "lock_wait_s", "dispatch_wait_s",
                              "audio_s", "input_chars"):
                    summary[field] = _distribution([
                        v[field] for v in values if v[field] is not None
                    ])
                summary["real_time_factor"] = _distribution([
                    v["service_s"] / v["audio_s"] for v in values if v["audio_s"]
                ])
                result[stage] = summary
        return result


def report(p) -> dict:
    return p._observations.report({
        "stt": context(p.stt_slot.current, p._config.stt, "stt"),
        "mt": context(p.mt_slot.current, p._config.translate, "mt"),
    })


def log_report(p) -> None:
    """A separately copyable JSON report, containing no caption content."""
    try:
        logger.info("Inference observations: %s", json.dumps(report(p), allow_nan=False))
    except Exception:  # noqa: BLE001 -- diagnostics must never prevent stop
        logger.debug("Inference observations unavailable", exc_info=True)


def prepare(p, engine, stage):
    """Best-effort identity capture, outside the measured service interval."""
    try:
        generation = p._observations.generation
        cfg = p._config.stt if stage == "stt" else p._config.translate
        return generation, context(engine, cfg, stage)
    except Exception:  # noqa: BLE001 -- diagnostics must not affect captions
        return None


def finish(p, stage, token, stop, **measurements) -> None:
    """Discard cross-run/config/device samples and contain diagnostics errors."""
    if token is None or stop.is_set():
        return
    try:
        generation, ctx = token
        cfg = p._config.stt if stage == "stt" else p._config.translate
        slot = p.stt_slot if stage == "stt" else p.mt_slot
        if ctx.key != context(slot.current, cfg, stage).key:
            return
        p._observations.record(stage, ctx, generation=generation, **measurements)
    except Exception:  # noqa: BLE001 -- diagnostics must not affect captions
        return


def translate(p, engine, job, targets, stop, wait_start):
    """Time a real MT call with the engine slot held; exceptions still reach
    the existing original-caption fallback. Warm-up never enters this path."""
    acquired = time.monotonic()
    token = prepare(p, engine, "mt")
    try:
        # Targets were resolved before borrowing the slot; settings may have
        # changed during that wait. Never label an old request with new settings.
        expected = [languages.get(n) for n in p._config.translate.targets]
        if token is None or targets != [lang for lang in expected if lang != job.src]:
            token = None
        else:
            token[1].metadata.update(
                requested_target_count=len(targets), source_language=job.src.nllb,
                request_kind="microphone" if job.manage_typing else "typed",
            )
    except Exception:  # noqa: BLE001 -- stale/invalid settings only lose diagnostics
        token = None
    start = time.monotonic()
    result = engine.translate(job.text, job.src, targets)
    service_s = time.monotonic() - start
    if token is not None:
        # TranslateEngine returns one entry per completed target after model-
        # family deduplication (e.g. the two Chinese scripts in m2m100).
        if result is None:
            token = None
        else:
            token[1].metadata["target_count"] = len(result)
    finish(p, "mt", token, stop, service_s=service_s, lock_wait_s=acquired - wait_start,
           dispatch_wait_s=wait_start - job.created_at, input_chars=len(job.text),
           target_count=None if result is None else len(result))
    return result
