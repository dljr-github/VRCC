"""Tests for :class:`vrcc.core.live_apply.LiveApply` -- the Qt-free façade the
GUI calls to push Settings changes into the running stack. Each method must
delegate to the matching live-reconfigure entry point (and, for audio, contain
a failed device open as a published ``MIC_OPEN_FAILED`` rather than raising).
"""

from __future__ import annotations

from types import SimpleNamespace

from vrcc.core.bus import EventBus
from vrcc.core.config import OscConfig, VadConfig
from vrcc.core.events import AppError
from vrcc.core.live_apply import LiveApply


class _FakePipeline:
    def __init__(self, running: bool = True, boom: bool = False) -> None:
        self._running = running
        self._boom = boom
        self.restarted_with = None
        self.mute = None

    def restart_source(self, new_source):
        self.restarted_with = new_source
        if self._boom:
            raise RuntimeError("device unavailable")
        return self._running

    def set_mute(self, mute):
        self.mute = mute


class _FakeSegmenter:
    def __init__(self) -> None:
        self.reconfigured = []

    def reconfigure(self, cfg) -> None:
        self.reconfigured.append(cfg)


class _FakeChatbox:
    def __init__(self) -> None:
        self.client = []
        self.rate = []

    def reconfigure(self, ip, port) -> None:
        self.client.append((ip, port))

    def reconfigure_rate(self, burst, min_interval_s) -> None:
        self.rate.append((burst, min_interval_s))


class _FakeMute:
    def __init__(self) -> None:
        self.events = []

    def start(self) -> None:
        self.events.append("start")

    def stop(self) -> None:
        self.events.append("stop")


def _make(*, pipeline=None, mute=None, sources=None):
    bus = EventBus()
    seg = _FakeSegmenter()
    chatbox = _FakeChatbox()
    pipe = pipeline or _FakePipeline()
    reloads = []
    built = []
    made_mutes = []

    def make_source(device_cfg):
        src = SimpleNamespace(device=device_cfg)
        built.append(device_cfg)
        return src

    def make_mute():
        built_mute = _FakeMute()
        made_mutes.append(built_mute)
        return built_mute

    live = LiveApply(
        pipeline=pipe,
        segmenter=seg,
        chatbox=chatbox,
        bus=bus,
        reload_engine=reloads.append,
        make_source=make_source,
        make_mute=make_mute,
        mute=mute,
    )
    return SimpleNamespace(
        live=live, bus=bus, seg=seg, chatbox=chatbox, pipeline=pipe,
        reloads=reloads, built=built, made_mutes=made_mutes,
    )


def test_apply_vad_delegates_to_segmenter_reconfigure():
    env = _make()
    cfg = VadConfig(finalize_silence_ms=700)
    env.live.apply_vad(cfg)
    assert env.seg.reconfigured == [cfg]


def test_apply_osc_reconfigures_client_and_rate():
    env = _make()
    cfg = OscConfig(ip="10.0.0.9", port=9002, burst=7, min_interval_s=0.9)
    env.live.apply_osc(cfg)
    assert env.chatbox.client == [("10.0.0.9", 9002)]
    assert env.chatbox.rate == [(7, 0.9)]


def test_apply_audio_device_restarts_source_and_returns_running():
    pipe = _FakePipeline(running=True)
    env = _make(pipeline=pipe)
    assert env.live.apply_audio_device("Some Mic") is True
    assert env.built == ["Some Mic"]
    assert pipe.restarted_with.device == "Some Mic"


def test_apply_audio_device_publishes_mic_open_failed_on_error():
    pipe = _FakePipeline(boom=True)
    env = _make(pipeline=pipe)
    errors: list[AppError] = []
    env.bus.subscribe(AppError, errors.append)

    assert env.live.apply_audio_device("Bad Mic") is False  # contained, not raised
    assert [e.code for e in errors] == ["MIC_OPEN_FAILED"]
    assert "device unavailable" in errors[0].detail


def test_apply_mute_sync_starts_and_stops_existing_coordinator():
    mute = _FakeMute()
    env = _make(mute=mute)
    env.live.apply_mute_sync(True)
    env.live.apply_mute_sync(False)
    assert mute.events == ["start", "stop"]


def test_apply_mute_sync_builds_one_when_none_existed():
    # Mute sync off at launch means no coordinator was ever built. Enabling it
    # must build and install one rather than persist a setting that does
    # nothing until the next launch.
    env = _make(mute=None)

    env.live.apply_mute_sync(True)

    assert len(env.made_mutes) == 1
    assert env.made_mutes[0].events == ["start"]
    assert env.pipeline.mute is env.made_mutes[0]


def test_apply_mute_sync_reuses_the_coordinator_it_built():
    env = _make(mute=None)

    env.live.apply_mute_sync(True)
    env.live.apply_mute_sync(False)
    env.live.apply_mute_sync(True)

    assert len(env.made_mutes) == 1
    assert env.made_mutes[0].events == ["start", "stop", "start"]


def test_apply_mute_sync_disabled_without_a_coordinator_builds_nothing():
    env = _make(mute=None)
    env.live.apply_mute_sync(False)
    assert env.made_mutes == []


def test_reload_engine_delegates_to_injected_closure():
    env = _make()
    env.live.reload_engine("stt")
    env.live.reload_engine("mt")
    assert env.reloads == ["stt", "mt"]
