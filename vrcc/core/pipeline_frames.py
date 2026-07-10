"""Per-frame work for the pipeline's segmenter worker: the listen gate and
the energy pre-gate.

Module functions take the Pipeline instance ``p``: the segmenter, bus and
config stay Pipeline attributes -- only the per-frame code lives here (the
orchestrator sits at the 500-line cap). Import direction: pipeline imports
this module (never the reverse at runtime).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from vrcc.core import energy_gate
from vrcc.core.events import AppError, MicLevel

if TYPE_CHECKING:
    from vrcc.core.pipeline import Pipeline

# Same logger as the orchestrator: one operational stream for the pipeline.
logger = logging.getLogger("vrcc.core.pipeline")


def listen_gated(p: "Pipeline") -> bool:
    """Whether the segmenter worker should drop frames instead of buffering
    them (captioning off, or mute sync holding captions). A model swap does
    not gate frames: that pause is transient and speech during it still
    captions after the swap (job-time gating covers it)."""
    return not p._captioning or p.mute_gated()


def process_frame(p: "Pipeline", frame: "np.ndarray") -> None:
    """Handle one frame from the audio queue (segmenter worker thread).

    Gating at the frame level rather than at finalize means speech while
    muted is never buffered, so unmuting mid-sentence captions only what is
    said after the unmute. The transition abort resolves a pending
    speculative so the typing indicator never sticks on; gated frames still
    publish a meter level so the GUI meter keeps tracking the mic."""
    gated = listen_gated(p)
    if gated != p._frame_gated:
        p._frame_gated = gated
        if gated:
            for event in p._segmenter.abort():
                p._on_seg_event(event)
    if gated:
        level = energy_gate.rms(np.asarray(frame, dtype=np.float32))
        p._bus.publish(MicLevel(rms=level, vad_prob=0.0))
        return
    if energy_gated(p, frame):
        return  # near-silent frame: meter published, VAD skipped
    try:
        events = p._segmenter.process(frame)
    except Exception as exc:  # noqa: BLE001 -- must not kill the thread
        logger.exception("segmenter.process raised; dropping frame")
        p._bus.publish(AppError("SEGMENTER_FAILED", str(exc)))
        return
    for event in events:
        p._on_seg_event(event)


def energy_gated(p: "Pipeline", frame: "np.ndarray") -> bool:
    """Cheap RMS pre-gate (numeric decision in :mod:`vrcc.core.energy_gate`):
    a gated near-silent frame still publishes its :class:`MicLevel` here so
    the meter keeps moving, but skips the ~1 ms ONNX VAD."""
    level = energy_gate.gated_level(frame, p._config.audio, p._segmenter.active)
    if level is None:
        return False
    p._bus.publish(MicLevel(rms=level, vad_prob=0.0))
    return True
