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
# Minimum gap kept between the speech and silence thresholds so a dead band
# always exists and the silence bar can never invert past the speech bar.
MIN_GAP = 0.05

# Room-floor probe (see _update_room_floor). A noisy room can hold the
# segmenter ACTIVE almost continuously -- measured: six-speaker babble at
# 24.4 dB SNR triggers a false SegSpeechStart within 6 frames and the
# segmenter never returns to idle within a 47-frame (1.5 s) trace -- so the
# probe cannot wait for idle frames the way the pre-roll ring does; it runs
# every frame and separates room from speaker by the segmenter's own
# in-utterance peak, not by RMS or by active/idle state. _PEAK_MARGIN: a
# frame counts as the speaker's own voice (excluded) once its VAD reading is
# within this fraction of the peak seen so far; every other ACTIVE frame, and
# every idle frame (peak is 0, nothing is excluded), is a room candidate.
# Swept 0.60-0.95 (margin_sweep.py, scratchpad) against two real measurements
# at ratio 0.7: missed finals on the six-speaker-babble harness at 24.4 dB SNR
# (segmenter_noise_type.py, scratchpad) and clean-speech WER through the real
# Segmenter + whisper small/CPU over 25 LibriSpeech test-clean utterances at
# RMS 0.05 (wer_through_segmenter.py, scratchpad). The gate DOES open mid-
# utterance on ordinary clean speech at every margin tried (traced at 0.85,
# scratchpad: all 5 utterances checked, opening around 2-3 s in) -- margin
# does not keep it shut. What it changes is whether is_drop ever SUSTAINS: a
# below-peak dip resets on the next full-confidence frame (process(), "is_
# speech and not is_drop"), so a handful of scattered drop frames never reach
# finalize_silence_ms and cost nothing. At 0.90/0.95 more frames qualify as
# room evidence, and on at least one of the 25 utterances that is enough for
# a genuine dip to sustain long enough to finalize early: 0.50 points of WER
# (4.15% -> 4.65%). 0.85 is the highest margin measured with zero WER cost
# while still cutting missed finals to single digits (23/25 at ratio 0.0 ->
# 8/25); nothing here rules out that some untested utterance still sustains
# at 0.85, only that none of these 25 did.
_PEAK_MARGIN = 0.85
# Room-candidate SAMPLES (not elapsed frames) needed before the estimate is
# trusted enough to gate relative_silence_ratio open (see VadConfig.
# relative_silence_ratio). Below this count the room is unknown and the
# relative test stays off, same as a clean room -- the safe default for the
# very first utterance after launch or after reset() (device swap). In a
# genuinely quiet room this warms up within the first ~16 idle frames (every
# idle frame qualifies); a false-triggered noisy room reaches it slower, since
# only the ACTIVE frames that read below the false utterance's own peak count
# (traced on the 24.4 dB babble harness, scratchpad: 7 of 16 by the 47th
# frame, the rest arriving once real speech is underway) -- the room's own
# missed-final count (see _PEAK_MARGIN) is measured with that slower warm-up
# already priced in, so it is not a hidden cost.
_FLOOR_WARM_FRAMES = 16


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
    relies on to tie each speculative job to one resolution. "Silence" is
    either branch of :attr:`VadConfig.silence_threshold` (absolute) or
    :attr:`VadConfig.relative_silence_ratio` (a sustained fall from the
    in-utterance peak, for a loud room that never crosses the absolute bar).
    The relative branch only applies while :meth:`_update_room_floor`'s
    running estimate says the room itself reads above the absolute bar, so a
    clean room reproduces the absolute-only behavior exactly.
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
        self._peak_vad = 0.0

        # Room-floor probe state (see _update_room_floor); independent of the
        # in-flight utterance, so none of this is touched by _reset_to_idle.
        self._room_vad_floor = 0.0
        self._room_floor_n = 0

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
        self.cfg = cfg
        self._speculative_frames = speculative
        self._finalize_frames = finalize
        self._min_utterance_frames = min_utterance
        self._preroll_frames = preroll
        self._max_utterance_frames = max_utterance

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
        utterance id advances so a dropped utterance never shares its id.

        Also drops the room-floor estimate: a device swap changes the
        microphone and its coupling to the room, so the old reading is not
        evidence about the new one -- back to cold start, same as launch."""
        self._reset_to_idle()
        self._preroll.clear()
        self._room_vad_floor = 0.0
        self._room_floor_n = 0

    def abort(self) -> list[object]:
        """Discard the in-flight utterance and the pre-roll immediately,
        returning the ``SegDiscard`` needed to keep the resolve-every-
        speculative invariant when the pipeline stops listening mid-utterance
        (VRChat mute via mute sync, or the captioning toggle). Same threading
        contract as :meth:`reset`: call only from the thread that feeds
        :meth:`process`."""
        events: list[object] = []
        if self._pending_spec_samples is not None:
            events.append(SegDiscard(utterance_id=self._utterance_id))
        self.reset()
        return events

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
        self._update_room_floor(vad_prob)

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
                self._peak_vad = vad_prob
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

        # Silero keys on speech STRUCTURE, not level: a loud room can read
        # solidly above both absolute bars even with the speaker silent (see
        # VadConfig.relative_silence_ratio). Track the in-utterance ceiling and
        # treat a sustained fall to a fraction of it as silence too, so a
        # speaker who stops is detected even when the room does not. Peak
        # updates before the drop test so the frame that sets a new peak never
        # reads as its own drop. Gated on is_speech so a dead-band frame (never
        # is_speech) can never read as a drop: the absolute dead band stays a
        # true invariant regardless of relative_silence_ratio. Also gated on
        # room_gate_open: a clean room's estimate stays low for the early part
        # of an utterance, and even once genuine mid-utterance dips carry it
        # past the silence bar, is_drop resets on the next full-confidence
        # frame rather than sustaining -- see _PEAK_MARGIN above for the
        # measurement behind that margin.
        self._peak_vad = max(self._peak_vad, vad_prob)
        room_gate_open = (
            self._room_floor_n >= _FLOOR_WARM_FRAMES and self._room_vad_floor >= silence_bar
        )
        is_drop = (
            is_speech
            and room_gate_open
            and vad_prob < self._peak_vad * self.cfg.relative_silence_ratio
        )

        if is_speech and not is_drop:
            self._silence_run = 0
            if self._pending_spec_samples is not None:
                self._pending_spec_samples = None
                events.append(SegDiscard(utterance_id=self._utterance_id))
        elif is_silence or is_drop:
            self._silence_run += 1
        # else: dead-band frame (above silence_bar, below threshold) -- leave
        # _silence_run untouched. is_drop cannot land here; it requires
        # is_speech.

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
        self._peak_vad = 0.0
        self._utterance_id += 1

    def _update_room_floor(self, vad_prob: float) -> None:
        """Slow estimate of the room's own VAD reading, independent of
        active/idle: a noisy room can hold the segmenter ACTIVE almost
        continuously (a false SegSpeechStart re-triggers as soon as one
        finalizes), so waiting for idle frames the way the pre-roll ring does
        starves in exactly the room this exists to detect (measured: 24.4 dB
        babble never returns to idle within a 47-frame trace).

        Runs every frame instead, and separates room from speaker by the
        segmenter's own in-utterance peak rather than by RMS or by
        active/idle state: every idle frame qualifies (peak is 0, nothing is
        near it), and while ACTIVE, a frame qualifies unless its VAD reading
        is within _PEAK_MARGIN of self._peak_vad -- the speaker's own voice,
        excluded, while everything else (a false babble trigger's own
        fluctuation, a pause) feeds the estimate. Peak is read as of the
        START of this frame (this method runs before process() updates it),
        so the one frame that sets a NEW peak is never mistaken for room
        evidence either way: it is higher than the old peak, so it fails
        "within margin of the old peak" on its own. Qualifying samples feed a
        variable-then-fixed running average (1/(n+1) until
        _FLOOR_WARM_FRAMES samples, the fixed step after) so it converges to
        the plain average of what little evidence exists yet instead of
        crawling up from a zero-initialized bias."""
        is_candidate = not self._active or vad_prob < self._peak_vad * _PEAK_MARGIN
        if not is_candidate:
            return

        n = self._room_floor_n
        alpha = 1.0 / (n + 1) if n < _FLOOR_WARM_FRAMES else 1.0 / _FLOOR_WARM_FRAMES
        self._room_vad_floor += alpha * (vad_prob - self._room_vad_floor)
        if n < _FLOOR_WARM_FRAMES:
            self._room_floor_n = n + 1
