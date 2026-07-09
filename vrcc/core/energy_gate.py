"""Cheap RMS energy pre-gate: decide whether a near-silent audio frame may
skip the ~1 ms ONNX VAD.

Pure and Qt-free so the pipeline calls it per frame and tests drive it without
a bus. The pipeline owns the side effects (publishing the meter level); this
module only makes the numeric decision.
"""

from __future__ import annotations

import numpy as np

# The energy slider is int16-scaled (0-2000, classic mic-gate convention);
# frames are float32 in [-1, 1], so the threshold divides by the int16 range.
_INT16_FULL_SCALE = 32768.0


def rms(frame: np.ndarray) -> float:
    """Root-mean-square level of a float32 frame in [-1, 1]."""
    return float(np.sqrt(np.mean(np.square(frame))))


def gated_level(frame: np.ndarray, audio_cfg, active: bool) -> float | None:
    """Return the frame RMS when it is gated out, else ``None``.

    A frame is gated only when the gate is enabled, an utterance is not already
    active, and the level is below the configured threshold. Only utterance
    *starts* are blocked -- mid-utterance (``active``) every frame flows so a
    quiet tail isn't chopped. Config is read by the caller per frame, so
    Settings apply live.
    """
    if not audio_cfg.energy_gate_enabled:
        return None
    if active:
        return None
    level = rms(frame)
    if level >= audio_cfg.energy_threshold / _INT16_FULL_SCALE:
        return None
    return level
