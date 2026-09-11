"""Bus-handler hygiene for :mod:`vrcc.core.pipeline`.

Level values are recorded at their publish sites (SegLevel passthrough,
the segmenter-worker listen gate, the energy pre-gate) rather than through a
``MicLevel`` bus subscription, so constructing a Pipeline never adds a
handler to the shared bus.
"""

from __future__ import annotations

import pytest

from vrcc.audio.segmenter import SegLevel
from vrcc.core import pipeline_frames
from vrcc.core.bus import EventBus
from vrcc.core.config import AppConfig, AudioConfig
from vrcc.core.events import MicLevel

from .conftest import make_pipeline, sample


def test_constructing_pipelines_leaves_no_accumulating_miclevel_handler():
    bus = EventBus()
    for _ in range(5):
        make_pipeline(bus=bus)
    assert bus._handlers.get(MicLevel, []) == []


def test_seglevel_path_records_input_level():
    env = make_pipeline()
    env.pipeline._on_seg_event(SegLevel(rms=0.4, vad_prob=0.6))
    assert env.pipeline._input._rms_samples == [0.4]


def test_frame_gated_path_records_input_level():
    # captioning=None leaves the pipeline's off-by-default state, so every
    # frame takes the listen-gated branch (pipeline_frames.py's line 49-52).
    env = make_pipeline(captioning=None)
    pipeline_frames.process_frame(env.pipeline, sample(v=0.2))
    assert env.pipeline._input._rms_samples == [pytest.approx(0.2)]


def test_energy_pre_gate_path_records_input_level():
    # Rarely exercised: the energy gate ships disabled (config.py default).
    cfg = AppConfig(audio=AudioConfig(energy_gate_enabled=True, energy_threshold=1000))
    env = make_pipeline(config=cfg)
    pipeline_frames.process_frame(env.pipeline, sample(v=0.001))  # below threshold: gated
    assert env.pipeline._input._rms_samples == [pytest.approx(0.001, abs=1e-6)]
