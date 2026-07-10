"""Tests for :mod:`vrcc.core.pipeline_frames` -- the segmenter worker's
listen gate (frame-level mute sync / captioning toggle) and the energy
pre-gate it hosts. Frames are driven through ``process_frame`` directly on
the test thread; no worker threads are involved.
"""

from __future__ import annotations

import numpy as np
import pytest

from vrcc.audio.segmenter import SegFinal, Segmenter
from vrcc.core import pipeline_frames
from vrcc.core.config import AppConfig, AudioConfig, VadConfig
from vrcc.core.events import MicLevel

from .conftest import FakeMute, collect, make_pipeline, sample


class ScriptedVad:
    """Pops one scripted probability per call; errors loudly if exhausted."""

    def __init__(self, probs: list[float]) -> None:
        self._probs = list(probs)
        self.calls = 0

    def __call__(self, frame: np.ndarray) -> float:
        self.calls += 1
        if not self._probs:
            raise AssertionError(
                f"ScriptedVad exhausted after {self.calls} calls but process() "
                "was called again"
            )
        return self._probs.pop(0)


# Quick-finalize VAD timings so a real-segmenter utterance completes in a
# handful of frames (speculatives disabled unless a test enables them).
_FAST_CFG = dict(
    pre_roll_ms=64,                 # 2 frames
    speculative_silence_ms=64_000,  # disabled
    finalize_silence_ms=64,         # 2 frames
    min_utterance_ms=32,
)


def _feed(env, value: float, count: int) -> None:
    for _ in range(count):
        pipeline_frames.process_frame(env.pipeline, sample(v=value))


# -- the regression: speech while muted must never be captured ---------------


def test_unmuting_mid_sentence_captions_only_post_unmute_audio():
    # The user's scenario: talk while muted in VRChat, then unmute mid
    # sentence. The final STT job must contain only audio from after the
    # unmute (fill 0.5), none of the muted speech (fill 0.25).
    vad = ScriptedVad([0.9] * 3 + [0.2, 0.2])
    seg = Segmenter(VadConfig(**_FAST_CFG), vad)
    env = make_pipeline(segmenter=seg, mute=FakeMute(caption=False))
    env.config.audio.energy_gate_enabled = False

    _feed(env, 0.25, 4)  # speech while muted
    assert env.pipeline._stt_queue.empty()
    assert seg.active is False
    assert vad.calls == 0  # frames never reached the segmenter

    env.mute.caption = True
    _feed(env, 0.5, 3)  # speech after the unmute
    _feed(env, 0.5, 2)  # silence tail -> SegFinal

    job = env.pipeline._stt_queue.get_nowait()
    assert env.pipeline._stt_queue.empty()
    assert job.speculative is False
    assert not np.any(job.samples == np.float32(0.25))
    assert np.all(job.samples == np.float32(0.5))


def test_mute_closing_mid_utterance_aborts_it():
    # A speculative is in flight (typing on, job queued) when mute closes:
    # the transition abort must resolve the typing indicator, and no final
    # job may ever appear for that utterance.
    cfg = VadConfig(**dict(_FAST_CFG, speculative_silence_ms=64, finalize_silence_ms=640))
    vad = ScriptedVad([0.9] * 3 + [0.2, 0.2])
    seg = Segmenter(cfg, vad)
    env = make_pipeline(segmenter=seg, mute=FakeMute(caption=True))
    env.config.audio.energy_gate_enabled = False

    _feed(env, 0.5, 3)  # speech
    _feed(env, 0.5, 2)  # silence -> speculative fires
    assert env.chatbox.typing[:1] == [True]

    env.mute.caption = False
    _feed(env, 0.5, 1)  # transition frame: gate closes, utterance aborted
    _feed(env, 0.5, 3)  # more silence/speech while muted changes nothing

    assert env.chatbox.typing[-1] is False
    job = env.pipeline._stt_queue.get_nowait()
    assert job.speculative is True  # the speculative job is all there ever was
    assert env.pipeline._stt_queue.empty()


def test_second_mute_cycle_still_drops_pre_unmute_speech():
    # Close -> open -> close -> open: the SECOND close must abort the open
    # utterance exactly like the first. A gate that only remembers its first
    # transition would keep first-cycle speech (fill 0.7) buffered across the
    # second mute and stitch it into the caption finalized after the second
    # unmute, reintroducing the original bug one cycle later.
    vad = ScriptedVad([0.9] * 3 + [0.9] * 3 + [0.2] * 2)
    seg = Segmenter(VadConfig(**_FAST_CFG), vad)
    env = make_pipeline(segmenter=seg, mute=FakeMute(caption=False))
    env.config.audio.energy_gate_enabled = False

    _feed(env, 0.25, 2)  # cycle 1: speech while muted, dropped
    env.mute.caption = True
    _feed(env, 0.7, 3)  # cycle 1: utterance opens, never finalized
    env.mute.caption = False
    _feed(env, 0.25, 2)  # cycle 2: gate closes mid-utterance, aborts it
    env.mute.caption = True
    _feed(env, 0.5, 3)  # cycle 2: speech after the second unmute
    _feed(env, 0.5, 2)  # silence tail -> SegFinal

    job = env.pipeline._stt_queue.get_nowait()
    assert env.pipeline._stt_queue.empty()
    assert job.speculative is False
    assert not np.any(job.samples == np.float32(0.7))
    assert not np.any(job.samples == np.float32(0.25))
    assert np.all(job.samples == np.float32(0.5))


# -- gated frames still feed the GUI meter -----------------------------------


def test_meter_stays_alive_while_mute_gated():
    env = make_pipeline(mute=FakeMute(caption=False))
    levels = collect(env.bus, MicLevel)
    _feed(env, 0.5, 3)
    assert len(levels) == 3
    for level in levels:
        assert level.rms == pytest.approx(0.5, abs=1e-6)
        assert level.vad_prob == 0.0
    assert env.segmenter.frames == []  # frames never reach the segmenter
    # abort fires once, on the close transition, not on every gated frame
    assert env.segmenter.aborts == 1


# -- the captioning toggle joins the same gate --------------------------------


def test_captioning_toggle_gates_frames_and_reopens():
    env = make_pipeline(captioning=False)  # mute None
    levels = collect(env.bus, MicLevel)
    _feed(env, 0.5, 2)
    assert env.segmenter.frames == []
    assert len(levels) == 2  # no level lost while gated

    env.pipeline.set_captioning(True)
    _feed(env, 0.5, 2)
    assert len(env.segmenter.frames) == 2
    # one abort for the initial close; the reopen transition must not abort
    assert env.segmenter.aborts == 1


# -- a model swap does not gate frames ----------------------------------------


def test_swap_does_not_frame_gate():
    # The swap pause is transient: frames keep flowing into the segmenter so
    # speech across a swap still captions once the swap lands. Job-time
    # gating (_should_caption) still blocks caption creation meanwhile.
    env = make_pipeline()
    env.pipeline.set_swapping(True)
    _feed(env, 0.5, 2)
    assert len(env.segmenter.frames) == 2

    env.pipeline._on_seg_event(SegFinal(utterance_id=1, samples=sample()))
    assert env.pipeline._stt_queue.empty()  # gated at job time, not dropped mid-air


# -- energy pre-gate behavior survives the move --------------------------------


def test_energy_gate_blocks_quiet_frames_and_passes_loud_ones():
    # energy_threshold=1000 -> float32 rms cutoff 1000/32768 ~= 0.0305.
    cfg = AppConfig(audio=AudioConfig(energy_gate_enabled=True, energy_threshold=1000))
    env = make_pipeline(config=cfg)
    levels = collect(env.bus, MicLevel)

    pipeline_frames.process_frame(env.pipeline, sample(v=0.001))  # gated
    assert env.segmenter.frames == []
    assert len(levels) == 1  # meter still flows
    assert levels[0].rms == pytest.approx(0.001, abs=1e-6)
    assert levels[0].vad_prob == 0.0

    pipeline_frames.process_frame(env.pipeline, sample(v=0.1))  # passes
    assert len(env.segmenter.frames) == 1
