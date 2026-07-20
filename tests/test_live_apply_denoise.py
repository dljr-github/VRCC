"""Tests for the denoise live-apply path: Pipeline.set_source_denoise and
LiveApply.apply_audio_denoise, mirroring the equivalent gain-path coverage in
test_live_apply.py.
"""

from __future__ import annotations

from vrcc.core.config import AudioConfig
from vrcc.core.live_apply import LiveApply

from .conftest import make_pipeline


class _DenoiseSource:
    def __init__(self) -> None:
        self.denoise_calls = []

    def start(self, on_frame) -> None:
        pass

    def stop(self) -> None:
        pass

    def set_denoise(self, enabled, strength) -> None:
        self.denoise_calls.append((enabled, strength))


class _FakePipeline:
    def __init__(self) -> None:
        self.denoise_calls = []

    def set_source_denoise(self, enabled, strength) -> None:
        self.denoise_calls.append((enabled, strength))


def test_set_source_denoise_delegates_to_source_set_denoise():
    source = _DenoiseSource()
    env = make_pipeline(source=source)
    env.pipeline.set_source_denoise(True, 0.8)
    assert source.denoise_calls == [(True, 0.8)]


def test_set_source_denoise_is_a_noop_without_a_denoise_processor():
    env = make_pipeline()  # default FakeSource has no set_denoise
    env.pipeline.set_source_denoise(True, 0.8)  # must not raise


def test_apply_audio_denoise_delegates_to_pipeline_set_source_denoise():
    pipe = _FakePipeline()
    live = LiveApply(
        pipeline=pipe,
        segmenter=object(),
        chatbox=object(),
        bus=object(),
        reload_engine=lambda kind: None,
        make_source=lambda device_cfg: None,
        make_mute=lambda: None,
    )
    cfg = AudioConfig(denoise_enabled=True, denoise_strength=0.65)
    live.apply_audio_denoise(cfg)
    assert pipe.denoise_calls == [(True, 0.65)]
