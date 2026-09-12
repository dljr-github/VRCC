"""AudioConfig and VadConfig, split out of config.py (config_migrate.py is
the sibling precedent for a config.py satellite module). config.py sits at
the 500-line cap (tests/test_structure.py) with no room for the gating
comment relative_silence_ratio needs, so both classes moved here verbatim
and config.py imports and re-exports them. Every existing call site
(`from vrcc.core.config import VadConfig`, in segmenter.py, live_apply.py
and the test suite) keeps working unchanged.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class AudioConfig(BaseModel):
    device: str = "auto"
    energy_gate_enabled: bool = False
    energy_threshold: int = 300
    # GTCRN noise suppression before the VAD/STT. Off by default: it corrupts
    # short words on quiet clean speech (SenseVoice decodes "testing" as
    # "Investesting" at 0.5) and clean-clip accuracy drops as strength rises,
    # while its win is only on genuinely noisy input, so a noisy-room user opts
    # in rather than every user paying the cost. strength is a dry/wet blend in
    # [0,1]; a gentle 0.25 when enabled, since 0.5 was where the short-word
    # damage set in.
    denoise_enabled: bool = False
    denoise_strength: float = Field(default=0.25, ge=0.0, le=1.0)
    # Caption what the SPEAKERS play, so other people in VRChat can be read as
    # well as heard. Off by default: it is a second transcription stream, and
    # what it captures is the whole output device rather than VRChat's voice
    # channel, so it is opt-in rather than a surprise. Empty device means the
    # current default speaker, resolved at capture time so a headset swap is
    # followed.
    hear_others_enabled: bool = False
    hear_others_device: str = ""
    # Empty means whatever the user speaks, which is right for almost everyone.
    # Set explicitly by someone who wants to read others in a language they are
    # not captioning themselves in.
    hear_others_language: str = ""


class VadConfig(BaseModel):
    # Silero speech probability to start an utterance. Errs sensitive on
    # purpose: a missed utterance is silent and reads as a broken app, while a
    # false trigger is visible and easy to turn down. Clean speech sits only
    # just above 0.5, so a lower bar also catches soft or unclear speech.
    threshold: float = 0.35
    # Silence bar, decoupled from the speech threshold so raising sensitivity
    # (lowering the speech threshold) never raises the silence bar and chops
    # words mid-utterance. Clamped below the speech threshold at use.
    silence_threshold: float = 0.25
    speculative_silence_ms: int = 250
    finalize_silence_ms: int = 600
    min_utterance_ms: int = 500
    pre_roll_ms: int = 150
    max_utterance_s: float = 28.0
    # Silero reads speech STRUCTURE, not level: a loud room (six-speaker
    # babble) can sit above both the speech and silence bars even with the
    # speaker silent, so silence_threshold alone never trips (segmenter_noise_
    # type harness, scratchpad: 24.4 dB babble read mean VAD 0.797 on room
    # noise alone, well past silence_threshold 0.25). A sustained fall to this
    # fraction of the in-utterance peak counts as silence too.
    #
    # Applying that fall test unconditionally costs clean-speech accuracy once
    # the discriminator is loose enough to let it: a mid-utterance dip during
    # real speech can itself read as a fall from peak, finalizing early. So
    # Segmenter only applies this ratio while its own running estimate of the
    # room's ambient VAD reading (see Segmenter._update_room_floor,
    # _PEAK_MARGIN) says the absolute silence_threshold cannot fire; a clean
    # room's estimate stays near 0 and the ratio is inert, reproducing
    # ratio-0.0 behavior exactly. 0.7 is the value that estimate gates.
    # Measured through the real Segmenter + whisper small/CPU on 25 clean
    # LibriSpeech test-clean utterances at RMS 0.05 (wer_through_segmenter.py,
    # scratchpad): WER 4.15% at ratio 0.0 and 4.15% at ratio 0.7 gated (no
    # regression); the same babble harness (segmenter_noise_type.py,
    # scratchpad, run unmodified) cut missed finals at the 24.4 dB floor from
    # 23/25 (ratio 0.0) to 8/25, while the 44.4 and 34.0 dB clean floors and
    # the spectrum-matched (no speech structure) control held 0 missed at
    # every floor. 14 dB and below stay unfixable: no relative fall to detect
    # there once the room itself sits at ~0.83 VAD. An earlier ungated sweep
    # (coverage_wer.txt, scratchpad) had put the unconditional cost at 4.81%
    # -> 5.88%; this measurement could not reproduce that on the current
    # tree (ungated ratio 0.7 measured 4.15%, unchanged) -- the mechanism is
    # still real (margin_sweep.py, scratchpad: loosening the gate to where it
    # lets mid-utterance dips through does cost 0.50 points, 4.15% -> 4.65%),
    # just smaller today than that figure recorded.
    relative_silence_ratio: float = 0.7
