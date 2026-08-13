"""Measured STT benchmark tables.

Split out of :mod:`vrcc.core.recommend` so the measurements sit apart from the
ranking that reads them. ``recommend`` re-exports both names and the dicts are
the same objects, so ``recommend.STT_BENCH`` stays a valid patch target.
"""

from __future__ import annotations

# Measured with tools/bench_stt.py on LibriSpeech test-clean (100 utterances);
# reference machine: Ryzen 9 9950X3D + RTX 5090, full run recorded in
# benchmarks/rtx-5090-ryzen-9950x3d.json. WER is scored with the quality gates
# opened, so it measures what the model recognized rather than what the gates
# suppressed. A model added to the STT registry must be benchmarked with
# tools/bench_stt.py and its row added here, or it ranks behind every measured
# model in its partition. CPU latency is the load-sensitive column: it moved by
# about a quarter between runs on a busy machine, so treat the ordering as the
# signal and the absolute value as approximate.
# id -> (wer_gpu, wer_cpu, gpu_median_s, cpu_median_s)
STT_BENCH: dict[str, tuple[float, float, float, float]] = {
    "tiny": (0.074, 0.079, 0.03, 0.13),
    "base": (0.057, 0.059, 0.04, 0.25),
    "small": (0.037, 0.037, 0.09, 0.74),
    "medium": (0.027, 0.026, 0.17, 2.41),
    "large-v3": (0.017, 0.018, 0.24, 3.90),
    "large-v3-turbo": (0.017, 0.016, 0.07, 2.81),
    "distil-large-v3.5": (0.024, 0.023, 0.06, 2.78),
    "distil-small.en": (0.040, 0.040, 0.04, 0.64),
    "parakeet-tdt-0.6b-v3": (0.023, 0.023, 0.21, 0.13),
}

# Beam 5 (the Quality mode) against beam 1 (Speed), same runs. Only whisper
# models have a beam to widen: the onnx-asr decoders are greedy, which is why
# the Mode control greys out for them. Models absent from a device's row were
# not measured there because they already lag past the latency gate at beam 1.
# id -> {device: (wer_beam5, median_beam5_s)}
BEAM_BENCH: dict[str, dict[str, tuple[float, float]]] = {
    "tiny": {"gpu": (0.071, 0.04), "cpu": (0.070, 0.14)},
    "base": {"gpu": (0.047, 0.06), "cpu": (0.049, 0.26)},
    "small": {"gpu": (0.035, 0.09), "cpu": (0.038, 0.78)},
    "medium": {"gpu": (0.024, 0.18)},
    "large-v3": {"gpu": (0.017, 0.26)},
    "large-v3-turbo": {"gpu": (0.018, 0.07)},
    "distil-large-v3.5": {"gpu": (0.023, 0.06)},
    "distil-small.en": {"gpu": (0.041, 0.04), "cpu": (0.040, 0.81)},
}

# Peak GPU memory the app adds while transcribing, in MB: NVML used-memory at
# rest subtracted from the peak across load, warm-up and four transcriptions,
# so it covers the CUDA context and the decode workspace, not just the weights.
# Measured 2026-08-08 on the reference machine at int8_float16, which is what
# best_compute_type resolves to on the cards the gpu_low budget governs (the
# reference card is sm120, where int8 is dropped, so a plain run there would
# have measured float16 and overstated every row).
#
# A checkpoint size cannot stand in for this. The ratio of peak to file size
# runs from 0.89x (large-v3, whose int8 weights are the bulk of a 3090 MB file)
# to 6.76x (tiny, which is almost all fixed CUDA context), and the ordering is
# not even preserved:
# large-v3-turbo ships a bigger file than medium (1620 MB against 1530) yet
# peaks lower (1531 MB against 1597), because its four decoder layers need far
# less working memory than medium's twenty-four. Sizing on the file therefore
# rejected turbo and admitted medium on a 6 GB card, which is the better model
# on word error, on latency and on memory.
#
# The onnx-asr models are absent because they would not build a CUDA session
# here; under an "auto" device they resolve to CPU anyway (see
# hardware.resolved_device), so they hold no VRAM in the configuration the
# recommender is sizing for. An id absent from this table is never size-gated.
# id -> peak_mb
STT_VRAM_MB: dict[str, int] = {
    "tiny": 507,
    # Re-measured 2026-08-10: 571 in three independent runs, identical to
    # 0.1 MB. The old 444 was the only row off by more than 3.3%, and it put
    # base BELOW tiny, which no weight count can produce.
    "base": 571,
    "small": 826,
    "medium": 1597,
    "large-v3": 2741,
    "large-v3-turbo": 1531,
    "distil-large-v3.5": 1449,
    "distil-small.en": 699,
}

# Same probe, same models, at the float16 the engines resolve to on compute
# capability >= 12: best_compute_type drops every int8 type there, so a
# Blackwell card never runs the peaks above. Measured 2026-08-10 on the
# reference machine (RTX 5090, sm120).
#
# The two tables disagree by 1.13x to 1.67x, which is why neither a scaling
# factor nor always sizing at the worst case works. A factor understates
# large-v3 badly at the low end of that spread; the worst case invents five
# warnings on 6 to 11 GB cards where the model genuinely fits, which are the
# users who can least afford a recommender that cries wolf.
STT_VRAM_FP16_MB: dict[str, int] = {
    "tiny": 571,
    "base": 667,
    "small": 1115,
    "medium": 2523,
    "large-v3": 4379,
    "large-v3-turbo": 2555,
    "distil-large-v3.5": 2427,
    "distil-small.en": 827,
}


def stt_vram_table(compute_type: str) -> dict[str, int]:
    """Peaks for the compute type the engines will actually use.

    Keyed on the resolved compute type rather than on compute capability, so a
    card with no int8 kernels for any other reason is sized right too.
    """
    return STT_VRAM_MB if compute_type.startswith("int8") else STT_VRAM_FP16_MB


# Same probe, same session, for the translation models. Not read by the
# ranking (the MT preset is per tier, not per card), but it is the other half
# of what VRCC puts on the card and it is what _GPU_LOW_VRAM_SHARE leaves room
# for, so it is recorded here rather than in a comment. Measured at the
# reference card's own float16, so these overstate an older card slightly.
# id -> peak_mb
MT_VRAM_MB: dict[str, int] = {
    "m2m100-418M-int8": 1604,
    "m2m100-1.2B-int8": 3546,
    "nllb-600M-int8": 2188,
    "nllb-1.3B-int8": 4273,
}

# The same models at int8_float16, which is what every card below compute
# capability 12 actually runs. Measured 2026-08-10 on the reference machine.
# The gap is large (nllb-600M 2188 at float16 against 1372 here), so sizing a
# 6 GB card off the float16 row put the gpu_low preset over a budget it fits
# with room to spare.
MT_VRAM_INT8_MB: dict[str, int] = {
    "m2m100-418M-int8": 996,
    "m2m100-1.2B-int8": 2011,
    "nllb-600M-int8": 1372,
    "nllb-1.3B-int8": 2426,
}


def mt_vram_table(compute_type: str) -> dict[str, int]:
    """Translation-model peaks for the compute type in use, the counterpart of
    :func:`stt_vram_table`."""
    return MT_VRAM_INT8_MB if compute_type.startswith("int8") else MT_VRAM_MB

