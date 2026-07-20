"""Capture-stage microphone gain: a fixed dB boost or a smoothed auto-level.

Applied to the mono 16 kHz signal before rechunking, so the level meter, the
VAD and Whisper all see the boosted audio. Pure numpy, zero Qt. configure()
runs on the GUI thread while process() runs on the audio callback thread; the
parameters are plain attribute stores (GIL-atomic), so no lock is taken (same
lock-free rationale as Segmenter.reconfigure).
"""

from __future__ import annotations

import math

import numpy as np

# Auto-level target and floor in linear RMS (roughly -20 dBFS target, -46 dBFS
# floor). Below the floor a frame is treated as silence and gain is held.
_TARGET_RMS = 0.1
_NOISE_FLOOR_RMS = 0.005
_MAX_GAIN = 10.0 ** (30.0 / 20.0)  # +30 dB ceiling
_MIN_GAIN = 1.0
# Per-frame smoothing (32 ms frames): fast attack, slow release.
_ATTACK = 0.2
_RELEASE = 0.02


def _soft_clip(x: np.ndarray) -> np.ndarray:
    """Bound to [-1, 1] with a tanh knee so a large boost saturates instead of
    hard-clipping into audible crackle."""
    return np.tanh(x).astype(np.float32)


class GainProcessor:
    def __init__(self) -> None:
        self._gain_db = 0.0
        # (auto, fixed_linear) published as one tuple so a GUI-thread
        # configure() can never be observed half-applied by the audio thread
        # (single reference swap is GIL-atomic; no lock needed).
        self._params: tuple[bool, float] = (False, 1.0)
        self._auto_gain = 1.0
        # Was the previous frame above the noise floor? Drives the onset snap:
        # the first above-floor frame after silence jumps straight to target.
        self._prev_above_floor = False

    def configure(self, gain_db: float, auto: bool) -> None:
        self._gain_db = float(gain_db)
        fixed_linear = 10.0 ** (self._gain_db / 20.0)
        self._params = (bool(auto), fixed_linear)

    def reset(self) -> None:
        self._auto_gain = 1.0
        self._prev_above_floor = False

    def process(self, frame: np.ndarray) -> np.ndarray:
        frame = np.asarray(frame, dtype=np.float32)
        if frame.size == 0:
            return frame
        auto, fixed_linear = self._params
        gain = self._next_auto_gain(frame) if auto else fixed_linear
        if gain == 1.0:
            return frame
        scaled = frame * gain
        if np.max(np.abs(scaled)) > 1.0:
            return _soft_clip(scaled)
        return scaled.astype(np.float32)

    def _next_auto_gain(self, frame: np.ndarray) -> float:
        rms = float(np.sqrt(np.mean(frame ** 2)))
        if not math.isfinite(rms):
            return self._auto_gain  # never let a bad frame poison the gain
        if rms < _NOISE_FLOOR_RMS:
            self._prev_above_floor = False
            return self._auto_gain  # hold; don't amplify the noise floor
        desired = _TARGET_RMS / max(rms, 1e-9)
        desired = float(np.clip(desired, _MIN_GAIN, _MAX_GAIN))
        if not self._prev_above_floor:
            # First above-floor frame after silence: snap straight to target
            # rather than ramp, so the onset (and the pre-roll recovered
            # around it) is amplified now instead of trailing a slow attack.
            self._prev_above_floor = True
            next_gain = desired
        else:
            coeff = _ATTACK if desired > self._auto_gain else _RELEASE
            next_gain = self._auto_gain + coeff * (desired - self._auto_gain)
        if not math.isfinite(next_gain):
            return self._auto_gain
        self._auto_gain = next_gain
        return self._auto_gain
