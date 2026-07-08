"""Wiring tests for the app composition root (``vrcc.app``): the headless
engine stack / background ``EngineLoader``, and the guarded pipeline start.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path

from vrcc.app import build_engine_stack
from vrcc.core.bus import EventBus
from vrcc.core.config import ConfigStore, default_paths
from vrcc.core.events import AppError
from vrcc.core.reloading import EngineLoader
from vrcc.translate.registry import MT_MODELS


# -- headless fakes ---------------------------------------------------------


class _FakeSource:
    def __init__(self) -> None:
        self.started = False
        self.on_frame = None

    def start(self, on_frame) -> None:
        self.started = True
        self.on_frame = on_frame

    def stop(self) -> None:
        self.started = False


class _FakeStt:
    def __init__(self) -> None:
        self.loaded = False
        self.warmed = False

    def load(self) -> None:
        self.loaded = True

    def warm_up(self) -> None:
        self.warmed = True

    def transcribe(self, samples):
        return None


class _FakeMt:
    def __init__(self) -> None:
        self.loaded = False
        self.warmed = False

    def load(self) -> None:
        self.loaded = True

    def warm_up(self) -> None:
        self.warmed = True

    def translate(self, text, src, targets):
        return []


class _FakeChatbox:
    def __init__(self) -> None:
        self.typing: list[bool] = []
        self.started = False
        self.stopped = False

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True

    def submit(self, text, utterance_id) -> None:
        pass

    def set_typing(self, value) -> None:
        self.typing.append(value)


class _FakeMute:
    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def should_caption(self) -> bool:
        return True


def _paths(tmp_path: Path):
    return default_paths(portable=True, app_dir=tmp_path)


def _store(tmp_path: Path) -> ConfigStore:
    return ConfigStore(_paths(tmp_path).config_file)


# -- build_engine_stack -----------------------------------------------------


def test_build_stack_with_all_fakes_starts_and_stops_clean(tmp_path):
    store = _store(tmp_path)
    bus = EventBus()
    source = _FakeSource()
    chatbox = _FakeChatbox()

    stack = build_engine_stack(
        store,
        bus,
        _paths(tmp_path),
        stt_engine=_FakeStt(),
        mt_engine=_FakeMt(),
        chatbox=chatbox,
        mute=_FakeMute(),
        source=source,
    )

    assert stack.pipeline is not None
    assert stack.mt is not None
    assert stack.mute is not None

    stack.pipeline.start()
    assert source.started is True
    stack.pipeline.stop()
    assert source.started is False


def test_translate_disabled_yields_no_mt_engine(tmp_path):
    store = _store(tmp_path)
    store.config.translate.enabled = False
    bus = EventBus()

    stack = build_engine_stack(
        store,
        bus,
        _paths(tmp_path),
        stt_engine=_FakeStt(),
        chatbox=_FakeChatbox(),
        mute=_FakeMute(),
        source=_FakeSource(),
    )

    assert stack.mt is None


def test_mute_disabled_yields_no_mute(tmp_path):
    store = _store(tmp_path)
    store.config.mute_sync.enabled = False
    bus = EventBus()

    stack = build_engine_stack(
        store,
        bus,
        _paths(tmp_path),
        stt_engine=_FakeStt(),
        mt_engine=_FakeMt(),
        chatbox=_FakeChatbox(),
        source=_FakeSource(),
    )

    assert stack.mute is None


def test_auto_audio_device_resolves_to_none(tmp_path):
    store = _store(tmp_path)
    store.config.audio.device = "auto"
    bus = EventBus()

    # source NOT injected -> a real MicSource is built.
    stack = build_engine_stack(
        store,
        bus,
        _paths(tmp_path),
        stt_engine=_FakeStt(),
        mt_engine=_FakeMt(),
        chatbox=_FakeChatbox(),
        mute=_FakeMute(),
    )

    assert stack.source._device is None


def test_stack_uses_configured_mt_spec_dir(tmp_path):
    store = _store(tmp_path)
    store.config.translate.model = "nllb-600M-int8"
    bus = EventBus()

    stack = build_engine_stack(
        store,
        bus,
        _paths(tmp_path),
        stt_engine=_FakeStt(),
        chatbox=_FakeChatbox(),
        mute=_FakeMute(),
        source=_FakeSource(),
    )

    # A real TranslateEngine was built for the configured spec.
    assert stack.mt is not None
    spec = MT_MODELS["nllb-600M-int8"]
    expected = _paths(tmp_path).models_dir / "mt" / spec.id
    assert stack.mt._model_dir == expected


# -- guarded pipeline start (mic failure must not be silent) ----------------


def test_start_pipeline_guarded_publishes_mic_open_failed(tmp_path):
    from vrcc.app import _start_pipeline_guarded

    class _BoomSource(_FakeSource):
        def start(self, on_frame) -> None:
            raise RuntimeError("PortAudio: device unavailable")

    store = _store(tmp_path)
    bus = EventBus()
    errors: list[AppError] = []
    bus.subscribe(AppError, errors.append)

    stack = build_engine_stack(
        store,
        bus,
        _paths(tmp_path),
        stt_engine=_FakeStt(),
        mt_engine=_FakeMt(),
        chatbox=_FakeChatbox(),
        mute=_FakeMute(),
        source=_BoomSource(),
    )

    assert _start_pipeline_guarded(stack.pipeline, bus) is False
    assert [e.code for e in errors] == ["MIC_OPEN_FAILED"]
    assert "device unavailable" in errors[0].detail
    stack.pipeline.stop()  # safe no-op: the failed start unwound itself


def test_start_pipeline_guarded_success_returns_true(tmp_path):
    from vrcc.app import _start_pipeline_guarded

    store = _store(tmp_path)
    bus = EventBus()
    errors: list[AppError] = []
    bus.subscribe(AppError, errors.append)
    source = _FakeSource()

    stack = build_engine_stack(
        store,
        bus,
        _paths(tmp_path),
        stt_engine=_FakeStt(),
        mt_engine=_FakeMt(),
        chatbox=_FakeChatbox(),
        mute=_FakeMute(),
        source=source,
    )

    assert _start_pipeline_guarded(stack.pipeline, bus) is True
    assert source.started is True
    assert errors == []
    stack.pipeline.stop()


# -- EngineLoader -----------------------------------------------------------


def test_engine_loader_calls_back_success(tmp_path):
    bus = EventBus()
    results: list[bool] = []
    stt, mt = _FakeStt(), _FakeMt()

    loader = EngineLoader(stt, mt, bus, on_complete=results.append)
    loader.start()
    loader.join(timeout=5.0)

    assert results == [True]
    assert stt.loaded and stt.warmed
    assert mt.loaded and mt.warmed


def test_engine_loader_success_with_no_mt(tmp_path):
    bus = EventBus()
    results: list[bool] = []
    stt = _FakeStt()

    loader = EngineLoader(stt, None, bus, on_complete=results.append)
    loader.start()
    loader.join(timeout=5.0)

    assert results == [True]
    assert stt.loaded and stt.warmed


def test_engine_loader_reports_failure_and_publishes_error(tmp_path):
    class _BoomStt(_FakeStt):
        def load(self) -> None:
            raise RuntimeError("boom")

    bus = EventBus()
    errors: list[AppError] = []
    bus.subscribe(AppError, errors.append)
    results: list[bool] = []

    loader = EngineLoader(_BoomStt(), _FakeMt(), bus, on_complete=results.append)
    loader.start()
    loader.join(timeout=5.0)

    assert results == [False]
    assert len(errors) == 1


def test_engine_loader_tracks_failed_kind_and_still_loads_other_engine(tmp_path):
    """Per-engine try blocks: an STT failure is recorded in ``failed_kinds``
    and must not skip the MT load (previously one try wrapped both)."""

    class _BoomStt(_FakeStt):
        def load(self) -> None:
            raise RuntimeError("boom")

    bus = EventBus()
    results: list[bool] = []
    mt = _FakeMt()

    loader = EngineLoader(_BoomStt(), mt, bus, on_complete=results.append)
    loader.start()
    loader.join(timeout=5.0)

    assert results == [False]
    assert loader.failed_kinds == {"stt"}
    assert mt.loaded and mt.warmed  # the other engine still loaded


def test_engine_loader_both_failures_report_each_kind(tmp_path):
    """Both engines failing publishes one ENGINE_LOAD_FAILED per engine and
    records both kinds -- per-engine reporting is intentional."""

    class _BoomStt(_FakeStt):
        def load(self) -> None:
            raise RuntimeError("stt boom")

    class _BoomMt(_FakeMt):
        def load(self) -> None:
            raise RuntimeError("mt boom")

    bus = EventBus()
    errors: list[AppError] = []
    bus.subscribe(AppError, errors.append)
    results: list[bool] = []

    loader = EngineLoader(_BoomStt(), _BoomMt(), bus, on_complete=results.append)
    loader.start()
    loader.join(timeout=5.0)

    assert results == [False]
    assert loader.failed_kinds == {"stt", "mt"}
    assert [e.code for e in errors] == ["ENGINE_LOAD_FAILED", "ENGINE_LOAD_FAILED"]


# -- startup-failure seeding (run()'s recovery contract, Qt-free) ------------
# _seeding_setup mirrors run(): the reloader is seeded from the startup config
# ids, and seed() applies _Starter._on_done's guarded _FAILED marking exactly.

_STARTUP_IDS = {"stt": "small", "mt": None}


def _seeding_setup():
    """Returns ``(reloader, pipe, builds, seed)``: a synchronous reloader
    seeded with the startup ids, plus ``seed(failed_kinds)`` mirroring the
    guard in ``_Starter._on_done`` (only a kind still on its startup id is
    marked ``_FAILED``; a user swap that already replaced it is kept)."""
    from vrcc.core.reloading import _FAILED, _Reloader

    class _Pipe:
        installed = None

        def detach_stt(self):
            return None

        def set_stt(self, e):
            self.installed = e

        def detach_mt(self):
            return None

        def set_mt(self, e):
            pass

    builds: list[str] = []
    pipe = _Pipe()

    def build(kind, target):
        builds.append(target)
        return (_FakeStt(), target)

    reloader = _Reloader(
        pipeline=pipe,
        build=build,
        load=lambda e: (e.load(), e.warm_up()),
        set_swapping=lambda v: None,
        set_status=lambda ok, reason="": None,
        marshal=lambda fn: fn(),
        spawn=lambda fn: fn(),
        bus=EventBus(),
        loaded=dict(_STARTUP_IDS),
    )
    reloader._on_pending = lambda kind: None

    def seed(failed_kinds):  # mirrors _Starter._on_done's guarded seeding
        for kind in failed_kinds:
            if reloader._loaded.get(kind) == _STARTUP_IDS[kind]:
                reloader._loaded[kind] = _FAILED

    return reloader, pipe, builds, seed


def test_startup_failure_seeding_lets_same_model_reselection_swap(tmp_path):
    """No user interaction during the load: the loader reports WHICH kind
    failed, the completion path seeds _FAILED (kind still on its startup id),
    and a later request() for the SAME configured id runs a real swap instead
    of no-opping into the dead engine."""

    class _BoomStt(_FakeStt):
        def load(self) -> None:
            raise RuntimeError("boom")

    reloader, pipe, builds, seed = _seeding_setup()

    def on_complete(success: bool) -> None:
        if not success:
            seed(loader.failed_kinds)

    loader = EngineLoader(_BoomStt(), None, EventBus(), on_complete=on_complete)
    loader.start()
    loader.join(timeout=5.0)

    reloader.request("stt", "small")  # same configured id -> must really swap
    assert builds == ["small"]
    assert pipe.installed is not None and pipe.installed.loaded


def test_late_startup_failure_does_not_clobber_successful_user_swap(tmp_path):
    """One engine can fail seconds before the other finishes loading. If the
    user swapped to a different model (successfully) inside that window, the
    late completion must NOT overwrite the working engine's id with _FAILED
    -- and the live engine stays a no-op on re-selection."""
    reloader, pipe, builds, seed = _seeding_setup()

    reloader.request("stt", "medium")  # user swap wins the race
    assert builds == ["medium"]

    seed({"stt"})  # late loader completion reports the startup failure
    assert reloader._loaded["stt"] == "medium"  # kept, not clobbered

    reloader.request("stt", "medium")  # still installed -> still a no-op
    assert builds == ["medium"]
