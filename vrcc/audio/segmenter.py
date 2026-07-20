"""Utterance segmentation state machine driven by an injected VAD callback.

Turns 512-sample (32 ms @ 16 kHz) frames into boundary events (speech start,
speculative, discard, final); ``vad_fn`` is injected (tests script probs). Pure
stdlib + numpy, zero Qt. Frame-math / hysteresis / reuse-identity noted inline.
"""

from __future__ import annotations

import math
import threading
from collections import deque
from dataclasses import dataclass
from typing import Callable

import numpy as np

from vrcc.core.config import VadConfig

FRAME = 512
# Minimum gap kept between the speech and silence thresholds so a dead band
# always exists and the silence bar can never invert past the speech bar.
MIN_GAP = 0.05


@dataclass(frozen=True)
class SegSpeechStart:
    utterance_id: int


@dataclass(frozen=True)
class SegSpeculative:
    utterance_id: int
    samples: np.ndarray


@dataclass(frozen=True)
class SegPartial:
    """A periodic buffer snapshot while an utterance is still active. Not part
    of the speculative/final resolve contract: additive, and never resolved by
    a SegFinal/SegDiscard."""

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
        self._vad_fn = vad_fn
        self.sample_rate = sample_rate

        self._apply_config(cfg)

        self._preroll: deque[np.ndarray] = deque(maxlen=self._preroll_frames)
        self._active = False
        self._utterance_id = 1
        self._buffer: list[np.ndarray] = []
        self._frames_since_start = 0
        self._silence_run = 0
        self._pending_spec_samples: np.ndarray | None = None
        self._frames_since_partial = 0
        self._partial_emitted = False
        self._commit_lock = threading.Lock()
        self._commit_requested: int | None = None

    def _apply_config(self, cfg: VadConfig) -> None:
        """Precompute the frame-count thresholds from ``cfg`` + the frame
        duration. Shared by __init__ and :meth:`reconfigure`; the latter runs on
        the GUI thread while the audio thread is inside :meth:`process`. No lock
        is taken: ``cfg`` is a single reference store and each threshold is a
        plain int, so the GIL makes every write atomic. Each value is computed
        into a local first and assigned exactly once, so :meth:`process` never
        observes a half-updated threshold -- at worst one call reads a mix of
        old/new, which just means the *next* utterance (not one in flight)
        adopts the new timings."""
        frame_ms = 1000.0 * FRAME / self.sample_rate
        speculative = math.ceil(cfg.speculative_silence_ms / frame_ms)
        finalize = math.ceil(cfg.finalize_silence_ms / frame_ms)
        min_utterance = math.ceil(cfg.min_utterance_ms / frame_ms)
        preroll = math.ceil(cfg.pre_roll_ms / frame_ms)
        max_utterance = math.ceil(cfg.max_utterance_s * 1000.0 / frame_ms)
        partial = math.ceil(cfg.partial_interval_ms / frame_ms)
        self.cfg = cfg
        self._speculative_frames = speculative
        self._finalize_frames = finalize
        self._min_utterance_frames = min_utterance
        # Invariant: pre-roll can never exceed the speculative-silence
        # window. The commit path (_reset_to_idle) keeps the pre-roll ring
        # across an early commit; if pre-roll held more audio than the
        # speculative window, it could still contain end-of-sentence speech
        # at commit time and prepend a stale word onto the next utterance.
        self._preroll_frames = min(preroll, speculative)
        self._max_utterance_frames = max_utterance
        self._partial_frames = partial

    def reconfigure(self, cfg: VadConfig) -> None:
        """Apply new VAD timings/threshold live (GUI thread) without dropping an
        in-flight utterance: the current utterance keeps its state and the next
        one adopts the new timings (see :meth:`_apply_config` for the lock-free
        rationale). The idle pre-roll ring is only resized when ``pre_roll_ms``
        changed -- it holds recent idle frames (not in-flight state), so a fresh
        ring just refills within ``pre_roll_ms``."""
        self._apply_config(cfg)
        if self._preroll.maxlen != self._preroll_frames:
            self._preroll = deque(maxlen=self._preroll_frames)

    def reset(self) -> None:
        """Drop all in-flight state: the open utterance, its buffer, the idle
        pre-roll ring and any pending speculative snapshot. For restarts where
        buffered audio belongs to a previous run (a device swap must not
        prefix the old microphone's audio onto the new one's first caption).
        Call only while no audio thread is feeding :meth:`process`; the
        utterance id advances so a dropped utterance never shares its id."""
        self._reset_to_idle()
        self._preroll.clear()

    def abort(self) -> list[object]:
        """Discard the in-flight utterance and the pre-roll immediately,
        returning the ``SegDiscard`` needed to keep the resolve-every-
        speculative invariant when the pipeline stops listening mid-utterance
        (VRChat mute via mute sync, or the captioning toggle). Same threading
        contract as :meth:`reset`: call only from the thread that feeds
        :meth:`process`."""
        events: list[object] = []
        if self._pending_spec_samples is not None or self._partial_emitted:
            events.append(SegDiscard(utterance_id=self._utterance_id))
        self.reset()
        return events

    @property
    def active(self) -> bool:
        """Whether mid-utterance (ACTIVE). The energy pre-gate consults this so
        it only blocks utterance *starts*, never frames already in flight."""
        return self._active

    def request_commit(self, utterance_id: int) -> None:
        """Ask the segmenter to end the current utterance and start a fresh one
        on the next frame (the STT worker detected a finished sentence and has
        already sent it). Thread-safe: called from the STT worker while the
        audio thread runs process(); a plain flag store under a short lock."""
        with self._commit_lock:
            self._commit_requested = utterance_id

    def process(self, frame: np.ndarray) -> list[object]:
        events: list[object] = []

        frame = np.asarray(frame, dtype=np.float32)
        vad_prob = float(self._vad_fn(frame))
        rms = float(np.sqrt(np.mean(frame**2)))
        events.append(SegLevel(rms=rms, vad_prob=vad_prob))

        with self._commit_lock:
            commit_id = self._commit_requested
            self._commit_requested = None
        if commit_id is not None and self._active and commit_id == self._utterance_id:
            # STT already emitted this sentence; drop the buffer and start a
            # fresh utterance. Keep the pre-roll ring so the next sentence's
            # onset is not clipped. No SegFinal (would double-send). Return
            # now instead of falling into the idle branch below: this exact
            # frame is not fed into the state machine, so it cannot both
            # close the old utterance and open the new one in one call --
            # the next process() call starts the next utterance cleanly.
            self._reset_to_idle()
            return events

        is_speech = vad_prob >= self.cfg.threshold
        silence_bar = min(self.cfg.silence_threshold, self.cfg.threshold - MIN_GAP)
        is_silence = vad_prob < silence_bar
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
                self._frames_since_partial = 0
                self._partial_emitted = False
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

        # Live partials are additive: a periodic buffer snapshot while the
        # utterance is active, independent of the speculative/final resolve
        # contract below. Counted in real frames, not silence, so it fires
        # throughout continuous speech, not just once silence starts.
        self._frames_since_partial += 1
        if (
            self.cfg.live_partials
            and self._frames_since_partial >= self._partial_frames
            and len(self._buffer) > self._preroll_frames
        ):
            events.append(
                SegPartial(
                    utterance_id=self._utterance_id,
                    samples=_concat(self._buffer),
                )
            )
            self._frames_since_partial = 0
            self._partial_emitted = True

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
            elif self._pending_spec_samples is not None or self._partial_emitted:
                # Too short for a final but a speculative or a live partial is
                # in flight: discard so the STT worker drops the job and the
                # LISTENING row is cleared (resolve-every invariant).
                events.append(SegDiscard(utterance_id=self._utterance_id))
            self._reset_to_idle()

        return events

    def _reset_to_idle(self) -> None:
        self._active = False
        self._buffer = []
        self._frames_since_start = 0
        self._silence_run = 0
        self._pending_spec_samples = None
        self._frames_since_partial = 0
        self._partial_emitted = False
        self._utterance_id += 1
