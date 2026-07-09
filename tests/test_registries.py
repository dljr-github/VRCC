"""Tests for the download manager: unit tests fake the network downloads,
with one integration-marked real download.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from vrcc.core.bus import EventBus
from vrcc.core.events import DownloadProgress
from vrcc.download.manager import DownloadManager
from vrcc.translate.registry import MT_MODELS


# --------------------------------------------------------------------------
# DownloadManager: paths + presence checks
# --------------------------------------------------------------------------

@pytest.fixture()
def bus() -> EventBus:
    return EventBus()


@pytest.fixture()
def manager(tmp_path: Path, bus: EventBus) -> DownloadManager:
    return DownloadManager(tmp_path / "models", bus)


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x")


def test_mt_model_dir_layout(manager: DownloadManager, tmp_path: Path):
    spec = MT_MODELS["nllb-600M-int8"]
    assert manager.mt_model_dir(spec) == tmp_path / "models" / "mt" / spec.id


def test_whisper_model_dir_layout(manager: DownloadManager, tmp_path: Path):
    assert manager.whisper_model_dir("small") == tmp_path / "models" / "whisper" / "small"


def test_is_mt_downloaded_false_on_empty(manager: DownloadManager):
    spec = MT_MODELS["nllb-600M-int8"]
    assert manager.is_mt_downloaded(spec) is False


def test_is_mt_downloaded_false_when_spm_missing(manager: DownloadManager):
    spec = MT_MODELS["nllb-600M-int8"]
    _touch(manager.mt_model_dir(spec) / "model.bin")
    assert manager.is_mt_downloaded(spec) is False


def test_is_mt_downloaded_false_when_model_bin_missing(manager: DownloadManager):
    spec = MT_MODELS["nllb-600M-int8"]
    _touch(manager.mt_model_dir(spec) / spec.spm_file)
    assert manager.is_mt_downloaded(spec) is False


def test_is_mt_downloaded_true_when_both_present(manager: DownloadManager):
    spec = MT_MODELS["nllb-600M-int8"]
    d = manager.mt_model_dir(spec)
    _touch(d / "model.bin")
    _touch(d / spec.spm_file)
    assert manager.is_mt_downloaded(spec) is True


def test_is_whisper_downloaded_false_on_empty(manager: DownloadManager):
    assert manager.is_whisper_downloaded("small") is False


def test_is_whisper_downloaded_true_after_touch(manager: DownloadManager):
    _touch(manager.whisper_model_dir("small") / "model.bin")
    assert manager.is_whisper_downloaded("small") is True


# --------------------------------------------------------------------------
# DownloadManager: delete
# --------------------------------------------------------------------------

def test_delete_removes_mt_dir(manager: DownloadManager):
    spec = MT_MODELS["nllb-600M-int8"]
    d = manager.mt_model_dir(spec)
    _touch(d / "model.bin")
    assert d.exists()
    manager.delete("mt", spec.id)
    assert not d.exists()


def test_delete_removes_whisper_dir(manager: DownloadManager):
    d = manager.whisper_model_dir("small")
    _touch(d / "model.bin")
    assert d.exists()
    manager.delete("whisper", "small")
    assert not d.exists()


def test_delete_missing_dir_is_noop(manager: DownloadManager):
    # deleting something that was never downloaded must not raise
    manager.delete("mt", "nllb-600M-int8")


def test_delete_unknown_kind_raises_value_error(manager: DownloadManager):
    with pytest.raises(ValueError):
        manager.delete("bogus", "small")


# --------------------------------------------------------------------------
# DownloadManager: ensure_mt
# --------------------------------------------------------------------------

def _run_realistic_snapshot(tqdm_class, local_dir: Path, spec, *, disable_bytes_bar=False):
    """Drive ``tqdm_class`` the way huggingface_hub's snapshot_download does.

    The real implementation instantiates the class TWICE: an aggregated
    BYTES bar (unit="B", total=0 initially, total grows as per-file
    metadata arrives via direct attribute mutation) and a FILE-COUNT bar
    inside thread_map (default unit, update(1) per finished file).
    """
    kwargs = {"disable": True} if disable_bytes_bar else {}
    bytes_bar = tqdm_class(
        desc="Downloading (incomplete total...)",
        total=0,
        initial=0,
        unit="B",
        unit_scale=True,
        **kwargs,
    )
    files_bar = tqdm_class(total=2, desc="Fetching 2 files")

    # chunk arrives before any file metadata set a total: must not publish
    bytes_bar.update(5)
    # file 1 metadata arrives, then its bytes
    bytes_bar.total += 100
    bytes_bar.refresh()
    bytes_bar.update(40)
    bytes_bar.update(55)
    files_bar.update(1)
    # file 2 metadata arrives (total grows), then its bytes
    bytes_bar.total += 50
    bytes_bar.refresh()
    bytes_bar.update(50)
    files_bar.update(1)

    bytes_bar.close()
    files_bar.close()

    local_dir.mkdir(parents=True, exist_ok=True)
    _touch(local_dir / "model.bin")
    _touch(local_dir / spec.spm_file)


def test_ensure_mt_calls_snapshot_and_publishes_progress(
    manager: DownloadManager, bus: EventBus, monkeypatch: pytest.MonkeyPatch
):
    spec = MT_MODELS["nllb-600M-int8"]
    events: list[DownloadProgress] = []
    bus.subscribe(DownloadProgress, events.append)

    calls: dict[str, object] = {}

    def fake_snapshot_download(repo_id, local_dir, tqdm_class=None, **kwargs):
        calls["repo_id"] = repo_id
        calls["local_dir"] = local_dir
        assert tqdm_class is not None
        _run_realistic_snapshot(tqdm_class, Path(local_dir), spec)
        return local_dir

    monkeypatch.setattr(
        "vrcc.download.manager.snapshot_download", fake_snapshot_download
    )

    result = manager.ensure_mt(spec)

    assert result == manager.mt_model_dir(spec)
    assert calls["repo_id"] == spec.repo
    assert Path(calls["local_dir"]) == manager.mt_model_dir(spec)

    progress = [e for e in events if not e.done]
    # only byte-derived events: the file-count bar (updates of 1, total 2)
    # and the pre-metadata chunk (total still 0) must not leak through
    assert [(e.downloaded, e.total) for e in progress] == [
        (45, 100),
        (100, 100),
        (150, 150),
    ]
    # no total=0 noise
    assert all(e.total > 0 for e in progress)
    # cumulative bytes are monotonic non-decreasing
    downloaded = [e.downloaded for e in progress]
    assert downloaded == sorted(downloaded)
    # exactly one terminal done event, tagged with the spec id
    done_events = [e for e in events if e.done]
    assert len(done_events) == 1
    assert done_events[0].model_id == spec.id
    for e in events:
        assert e.model_id == spec.id


def test_ensure_mt_survives_none_stderr(
    manager: DownloadManager, bus: EventBus, monkeypatch: pytest.MonkeyPatch
):
    """Regression: the windowed VRCC.exe (console=False) runs with
    ``sys.stderr is None``; tqdm's progress bar wrote there and raised
    ``'NoneType' object has no attribute 'write'`` on every download. The
    real ``_ProgressTqdm`` (passed here as ``tqdm_class``) must render nothing
    and still publish byte progress even with stderr gone."""
    import sys

    spec = MT_MODELS["nllb-600M-int8"]
    events: list[DownloadProgress] = []
    bus.subscribe(DownloadProgress, events.append)

    def fake_snapshot_download(repo_id, local_dir, tqdm_class=None, **kwargs):
        # _run_realistic_snapshot calls .refresh()/.close(), which would touch
        # stderr if the bar weren't disabled -- that is exactly the crash.
        _run_realistic_snapshot(tqdm_class, Path(local_dir), spec)
        return local_dir

    monkeypatch.setattr(
        "vrcc.download.manager.snapshot_download", fake_snapshot_download
    )

    saved_err, saved_out = sys.stderr, sys.stdout
    sys.stderr = None
    sys.stdout = None
    try:
        manager.ensure_mt(spec)  # must not raise AttributeError
    finally:
        sys.stderr, sys.stdout = saved_err, saved_out

    progress = [(e.downloaded, e.total) for e in events if not e.done]
    assert progress == [(45, 100), (100, 100), (150, 150)]
    assert any(e.done for e in events)


def test_ensure_mt_publishes_bytes_even_when_bar_disabled(
    manager: DownloadManager, bus: EventBus, monkeypatch: pytest.MonkeyPatch
):
    # In non-TTY contexts tqdm may disable the bar, and a disabled tqdm's
    # update() does not advance self.n -- the manager must still report
    # real cumulative bytes, not (0, total) noise.
    spec = MT_MODELS["nllb-600M-int8"]
    events: list[DownloadProgress] = []
    bus.subscribe(DownloadProgress, events.append)

    def fake_snapshot_download(repo_id, local_dir, tqdm_class=None, **kwargs):
        _run_realistic_snapshot(
            tqdm_class, Path(local_dir), spec, disable_bytes_bar=True
        )
        return local_dir

    monkeypatch.setattr(
        "vrcc.download.manager.snapshot_download", fake_snapshot_download
    )

    manager.ensure_mt(spec)

    progress = [e for e in events if not e.done]
    assert [(e.downloaded, e.total) for e in progress] == [
        (45, 100),
        (100, 100),
        (150, 150),
    ]
    assert sum(1 for e in events if e.done) == 1


def test_ensure_mt_short_circuits_when_downloaded(
    manager: DownloadManager, bus: EventBus, monkeypatch: pytest.MonkeyPatch
):
    spec = MT_MODELS["nllb-600M-int8"]
    d = manager.mt_model_dir(spec)
    _touch(d / "model.bin")
    _touch(d / spec.spm_file)

    events: list[DownloadProgress] = []
    bus.subscribe(DownloadProgress, events.append)

    def boom(*args, **kwargs):
        raise AssertionError("snapshot_download must not be called when present")

    monkeypatch.setattr("vrcc.download.manager.snapshot_download", boom)

    result = manager.ensure_mt(spec)

    assert result == d
    assert len(events) == 1
    assert events[0].done is True
    assert events[0].model_id == spec.id


def test_ensure_mt_propagates_errors(
    manager: DownloadManager, monkeypatch: pytest.MonkeyPatch
):
    spec = MT_MODELS["nllb-600M-int8"]

    def boom(*args, **kwargs):
        raise RuntimeError("network down")

    monkeypatch.setattr("vrcc.download.manager.snapshot_download", boom)

    with pytest.raises(RuntimeError):
        manager.ensure_mt(spec)


# --------------------------------------------------------------------------
# DownloadManager: ensure_whisper
# --------------------------------------------------------------------------

def test_ensure_whisper_calls_download_and_publishes_done(
    manager: DownloadManager, bus: EventBus, monkeypatch: pytest.MonkeyPatch
):
    events: list[DownloadProgress] = []
    bus.subscribe(DownloadProgress, events.append)

    calls: dict[str, object] = {}

    def fake_download_model(model_id, output_dir=None, **kwargs):
        calls["model_id"] = model_id
        calls["output_dir"] = output_dir
        d = Path(output_dir)
        d.mkdir(parents=True, exist_ok=True)
        _touch(d / "model.bin")
        return output_dir

    monkeypatch.setattr(
        "vrcc.download.manager.download_model", fake_download_model
    )

    result = manager.ensure_whisper("tiny")

    assert result == manager.whisper_model_dir("tiny")
    assert calls["model_id"] == "tiny"
    assert Path(calls["output_dir"]) == manager.whisper_model_dir("tiny")

    done_events = [e for e in events if e.done]
    assert len(done_events) == 1
    assert done_events[0].model_id == "tiny"


def test_ensure_whisper_short_circuits_when_downloaded(
    manager: DownloadManager, bus: EventBus, monkeypatch: pytest.MonkeyPatch
):
    _touch(manager.whisper_model_dir("tiny") / "model.bin")

    events: list[DownloadProgress] = []
    bus.subscribe(DownloadProgress, events.append)

    def boom(*args, **kwargs):
        raise AssertionError("download_model must not be called when present")

    monkeypatch.setattr("vrcc.download.manager.download_model", boom)

    result = manager.ensure_whisper("tiny")

    assert result == manager.whisper_model_dir("tiny")
    assert len(events) == 1
    assert events[0].done is True
    assert events[0].model_id == "tiny"


def test_ensure_whisper_propagates_errors(
    manager: DownloadManager, monkeypatch: pytest.MonkeyPatch
):
    def boom(*args, **kwargs):
        raise RuntimeError("network down")

    monkeypatch.setattr("vrcc.download.manager.download_model", boom)

    with pytest.raises(RuntimeError):
        manager.ensure_whisper("tiny")


# --------------------------------------------------------------------------
# DownloadManager: Parakeet (onnx-asr backend)
# --------------------------------------------------------------------------

_PARAKEET_ID = "parakeet-tdt-0.6b-v3"
_PARAKEET_FILES = (
    "config.json",
    "vocab.txt",
    "encoder-model.int8.onnx",
    "decoder_joint-model.int8.onnx",
)


def test_parakeet_lives_under_the_whisper_dir(manager: DownloadManager, tmp_path: Path):
    assert (
        manager.whisper_model_dir(_PARAKEET_ID)
        == tmp_path / "models" / "whisper" / _PARAKEET_ID
    )


def test_is_parakeet_downloaded_needs_every_file(manager: DownloadManager):
    d = manager.whisper_model_dir(_PARAKEET_ID)
    assert manager.is_whisper_downloaded(_PARAKEET_ID) is False
    for name in _PARAKEET_FILES[:-1]:
        _touch(d / name)
    assert manager.is_whisper_downloaded(_PARAKEET_ID) is False
    _touch(d / _PARAKEET_FILES[-1])
    assert manager.is_whisper_downloaded(_PARAKEET_ID) is True


def test_is_parakeet_downloaded_ignores_whisper_model_bin(manager: DownloadManager):
    _touch(manager.whisper_model_dir(_PARAKEET_ID) / "model.bin")
    assert manager.is_whisper_downloaded(_PARAKEET_ID) is False


def test_ensure_parakeet_snapshots_repo_with_needed_files_only(
    manager: DownloadManager, bus: EventBus, monkeypatch: pytest.MonkeyPatch
):
    events: list[DownloadProgress] = []
    bus.subscribe(DownloadProgress, events.append)
    calls: dict[str, object] = {}

    def fake_snapshot(repo_id, local_dir=None, allow_patterns=None, tqdm_class=None):
        calls["repo_id"] = repo_id
        calls["local_dir"] = local_dir
        calls["allow_patterns"] = allow_patterns
        for name in allow_patterns:
            _touch(Path(local_dir) / name)
        return local_dir

    monkeypatch.setattr("vrcc.download.manager.snapshot_download", fake_snapshot)

    def boom(*args, **kwargs):
        raise AssertionError("faster-whisper download_model must not run for parakeet")

    monkeypatch.setattr("vrcc.download.manager.download_model", boom)

    result = manager.ensure_whisper(_PARAKEET_ID)

    assert result == manager.whisper_model_dir(_PARAKEET_ID)
    assert calls["repo_id"] == "istupakov/parakeet-tdt-0.6b-v3-onnx"
    assert Path(calls["local_dir"]) == manager.whisper_model_dir(_PARAKEET_ID)
    assert set(calls["allow_patterns"]) == set(_PARAKEET_FILES)
    assert manager.is_whisper_downloaded(_PARAKEET_ID) is True
    done_events = [e for e in events if e.done]
    assert len(done_events) == 1
    assert done_events[0].model_id == _PARAKEET_ID


def test_ensure_parakeet_short_circuits_when_downloaded(
    manager: DownloadManager, bus: EventBus, monkeypatch: pytest.MonkeyPatch
):
    d = manager.whisper_model_dir(_PARAKEET_ID)
    for name in _PARAKEET_FILES:
        _touch(d / name)

    def boom(*args, **kwargs):
        raise AssertionError("snapshot_download must not be called when present")

    monkeypatch.setattr("vrcc.download.manager.snapshot_download", boom)

    result = manager.ensure_whisper(_PARAKEET_ID)

    assert result == d


def test_delete_removes_parakeet_dir(manager: DownloadManager):
    d = manager.whisper_model_dir(_PARAKEET_ID)
    _touch(d / "encoder-model.int8.onnx")
    manager.delete("whisper", _PARAKEET_ID)
    assert not d.exists()


# --------------------------------------------------------------------------
# Real download (manual verification only; excluded by default)
# --------------------------------------------------------------------------

@pytest.mark.integration
def test_integration_download_tiny_whisper(tmp_path: Path):
    manager = DownloadManager(tmp_path / "models", EventBus())
    out = manager.ensure_whisper("tiny")
    assert (out / "model.bin").exists()
    assert manager.is_whisper_downloaded("tiny") is True
