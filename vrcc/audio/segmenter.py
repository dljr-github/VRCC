"""Utterance segmentation state machine driven by an injected VAD callback.

Turns 512-sample (32 ms @ 16 kHz) frames into boundary events (speech start,
speculative, discard, final); ``vad_fn`` is injected (tests script probs). Pure
stdlib + numpy, zero Qt. Frame-math / hysteresis / reuse-identity noted inline.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Callable

import numpy as np

from vrcc.core.config import VadConfig

FRAME = 512
HYSTERESIS_GAP = 0.15


@dataclass(frozen=True)
class SegSpeechStart:
    utterance_id: int


@dataclass(frozen=True)
class SegSpeculative:
    utterance_id: int
    samples: np.ndarray


@dataclass(frozen=True)
class SegFinal:
    utterance_id: int
    samples: np.ndarray


@dataclass(frozen=True)
class SegDiscard:
    utterance_id: int


@dataclass(frozen=True)
class SegLevel:
    rms: float
    vad_prob: float


def _concat(frames: list[np.ndarray]) -> np.ndarray:
    """Concatenate buffered frames into one 1-D float32 array."""
    if not frames:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(frames)


class Segmenter:
    """Turns a frame stream into utterance-boundary events (one 512-sample
    float32 frame per :meth:`process`, in order; ``SegLevel`` every frame, the
    rest conditional). Invariant: every ``SegSpeculative`` is resolved by
    exactly one later ``SegFinal`` or ``SegDiscard``, which downstream STT
    relies on to tie each speculative job to one resolution.
    """

    def __init__(
        self,
        cfg: VadConfig,
        vad_fn: Callable[[np.ndarray], float],
        sample_rate: int = 16000,
    ) -> None:
        self.cfg = cfg
        self._vad_fn = vad_fn
        self.sample_rate = sample_rate

        frame_ms = 1000.0 * FRAME / sample_rate
        self._speculative_frames = math.ceil(cfg.speculative_silence_ms / frame_ms)
        self._finalize_frames = math.ceil(cfg.finalize_silence_ms / frame_ms)
        self._min_utterance_frames = math.ceil(cfg.min_utterance_ms / frame_ms)
        self._preroll_frames = math.ceil(cfg.pre_roll_ms / frame_ms)
        self._max_utterance_frames = math.ceil(cfg.max_utterance_s * 1000.0 / frame_ms)

        self._preroll: deque[np.ndarray] = deque(maxlen=self._preroll_frames)
        self._active = False
        self._utterance_id = 1
        self._buffer: list[np.ndarray] = []
        self._frames_since_start = 0
        self._silence_run = 0
        self._pending_spec_samples: np.ndarray | None = None

    @property
    def active(self) -> bool:
        """Whether mid-utterance (ACTIVE). The energy pre-gate consults this so
        it only blocks utterance *starts*, never frames already in flight."""
        return self._active

    def process(self, frame: np.ndarray) -> list[object]:
        events: list[object] = []

        frame = np.asarray(frame, dtype=np.float32)
        vad_prob = float(self._vad_fn(frame))
        rms = float(np.sqrt(np.mean(frame**2)))
        events.append(SegLevel(rms=rms, vad_prob=vad_prob))

        is_speech = vad_prob >= self.cfg.threshold
        is_silence = vad_prob < (self.cfg.threshold - HYSTERESIS_GAP)
        frame_copy = frame.copy()  # frame buffers may be reused by the caller

        if not self._active:
            if is_speech:
                self._buffer = list(self._preroll)
                self._preroll.append(frame_copy)
                self._buffer.append(frame_copy)
                self._active = True
                self._frames_since_start = 1
                self._silence_run = 0
                self._pending_spec_samples = None
                events.append(SegSpeechStart(utterance_id=self._utterance_id))
                # Degenerate configs (pre-roll >= max cap) can hit the cap on
                # this very transition frame; force the final here, not late.
                if len(self._buffer) >= self._max_utterance_frames:
                    events.append(
                        SegFinal(
                            utterance_id=self._utterance_id,
                            samples=_concat(self._buffer),
                        )
                    )
                    self._reset_to_idle()
            else:
                self._preroll.append(frame_copy)
            return events

        # ACTIVE
        self._preroll.append(frame_copy)
        self._buffer.append(frame_copy)
        self._frames_since_start += 1

        if is_speech:
            self._silence_run = 0
            if self._pending_spec_samples is not None:
                self._pending_spec_samples = None
                events.append(SegDiscard(utterance_id=self._utterance_id))
        elif is_silence:
            self._silence_run += 1
        # else: dead-band frame -- leave _silence_run untouched.

        if len(self._buffer) >= self._max_utterance_frames:
            samples = (
                self._pending_spec_samples
                if self._pending_spec_samples is not None
                else _concat(self._buffer)
            )
            events.append(SegFinal(utterance_id=self._utterance_id, samples=samples))
            self._reset_to_idle()
            return events

        if (
            self._silence_run >= self._speculative_frames
            # Skip the speculative if finalize trips on this same frame
            # (equal/inverted thresholds): the final is already here.
            and self._silence_run < self._finalize_frames
            and self._pending_spec_samples is None
        ):
            # Snapshot the buffer ONCE; SegFinal later reuses this exact object
            # (identity, not ==) if no speech intervenes, so STT can reuse its
            # speculative transcription instead of re-running inference.
            self._pending_spec_samples = _concat(self._buffer)
            events.append(
                SegSpeculative(
                    utterance_id=self._utterance_id,
                    samples=self._pending_spec_samples,
                )
            )

        if self._silence_run >= self._finalize_frames:
            if self._frames_since_start >= self._min_utterance_frames:
                samples = (
                    self._pending_spec_samples
                    if self._pending_spec_samples is not None
                    else _concat(self._buffer)
                )
                events.append(
                    SegFinal(utterance_id=self._utterance_id, samples=samples)
                )
            elif self._pending_spec_samples is not None:
                # Too short for a final but a speculative is in flight: discard
                # it so the STT worker drops the job (resolve-every invariant).
                events.append(SegDiscard(utterance_id=self._utterance_id))
            self._reset_to_idle()

        return events

    def _reset_to_idle(self) -> None:
        self._active = False
        self._buffer = []
        self._frames_since_start = 0
        self._silence_run = 0
        self._pending_spec_samples = None
        self._utterance_id += 1
