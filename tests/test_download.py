"""Tests for the hf-mirror.com retry fallback in ``DownloadManager``:
a primary download failure retries once against the mirror, restoring the
endpoint state (env var and ``huggingface_hub.constants.ENDPOINT``) whatever
the outcome.
"""

from __future__ import annotations

import os
from pathlib import Path

import huggingface_hub
import pytest

from vrcc.core.bus import EventBus
from vrcc.download.manager import _HF_MIRROR, DownloadManager
from vrcc.translate.registry import MT_MODELS


@pytest.fixture()
def bus() -> EventBus:
    return EventBus()


@pytest.fixture()
def manager(tmp_path: Path, bus: EventBus) -> DownloadManager:
    return DownloadManager(tmp_path / "models", bus)


@pytest.fixture(autouse=True)
def _restore_hf_endpoint():
    """Every case must leave the real endpoint state untouched, but a case
    that asserts BLOCKED still needs a clean slate going in and coming out."""
    prior_env = os.environ.get("HF_ENDPOINT")
    prior_const = huggingface_hub.constants.ENDPOINT
    yield
    if prior_env is None:
        os.environ.pop("HF_ENDPOINT", None)
    else:
        os.environ["HF_ENDPOINT"] = prior_env
    huggingface_hub.constants.ENDPOINT = prior_const


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x")


def _mt_spec():
    return MT_MODELS["nllb-600M-int8"]


# --------------------------------------------------------------------------
# ensure_mt: mirror fallback
# --------------------------------------------------------------------------

def test_ensure_mt_falls_back_to_mirror_on_primary_failure(
    manager: DownloadManager, monkeypatch: pytest.MonkeyPatch
):
    spec = _mt_spec()
    prior_env = os.environ.get("HF_ENDPOINT")
    prior_const = huggingface_hub.constants.ENDPOINT

    calls: list[str] = []
    seen_endpoint_on_success: dict[str, str] = {}

    def fake_snapshot_download(repo_id, local_dir, tqdm_class=None, **kwargs):
        calls.append(repo_id)
        if len(calls) == 1:
            raise RuntimeError("primary host unreachable")
        seen_endpoint_on_success["endpoint"] = huggingface_hub.constants.ENDPOINT
        d = Path(local_dir)
        d.mkdir(parents=True, exist_ok=True)
        _touch(d / "model.bin")
        _touch(d / spec.spm_file)
        return local_dir

    monkeypatch.setattr(
        "vrcc.download.manager.snapshot_download", fake_snapshot_download
    )

    result = manager.ensure_mt(spec)

    assert result == manager.mt_model_dir(spec)
    assert len(calls) == 2
    assert seen_endpoint_on_success["endpoint"] == _HF_MIRROR

    assert huggingface_hub.constants.ENDPOINT == prior_const
    assert os.environ.get("HF_ENDPOINT") == prior_env


def test_ensure_mt_both_attempts_fail_chains_and_restores(
    manager: DownloadManager, monkeypatch: pytest.MonkeyPatch
):
    spec = _mt_spec()
    prior_env = os.environ.get("HF_ENDPOINT")
    prior_const = huggingface_hub.constants.ENDPOINT

    calls: list[str] = []

    def fake_snapshot_download(repo_id, local_dir, tqdm_class=None, **kwargs):
        calls.append(repo_id)
        if len(calls) == 1:
            raise RuntimeError("primary host unreachable")
        raise RuntimeError("mirror host unreachable")

    monkeypatch.setattr(
        "vrcc.download.manager.snapshot_download", fake_snapshot_download
    )

    with pytest.raises(RuntimeError) as excinfo:
        manager.ensure_mt(spec)

    assert len(calls) == 2
    assert str(excinfo.value) == "mirror host unreachable"
    assert excinfo.value.__cause__ is not None
    assert str(excinfo.value.__cause__) == "primary host unreachable"

    assert huggingface_hub.constants.ENDPOINT == prior_const
    assert os.environ.get("HF_ENDPOINT") == prior_env


def test_ensure_mt_primary_success_no_retry_no_mutation(
    manager: DownloadManager, monkeypatch: pytest.MonkeyPatch
):
    spec = _mt_spec()
    prior_const = huggingface_hub.constants.ENDPOINT

    calls: list[str] = []
    seen_endpoints: list[str] = []

    def fake_snapshot_download(repo_id, local_dir, tqdm_class=None, **kwargs):
        calls.append(repo_id)
        seen_endpoints.append(huggingface_hub.constants.ENDPOINT)
        d = Path(local_dir)
        d.mkdir(parents=True, exist_ok=True)
        _touch(d / "model.bin")
        _touch(d / spec.spm_file)
        return local_dir

    monkeypatch.setattr(
        "vrcc.download.manager.snapshot_download", fake_snapshot_download
    )

    manager.ensure_mt(spec)

    assert len(calls) == 1
    assert seen_endpoints == [prior_const]
    assert huggingface_hub.constants.ENDPOINT == prior_const


def test_ensure_mt_no_retry_when_endpoint_already_mirror(
    manager: DownloadManager, monkeypatch: pytest.MonkeyPatch
):
    spec = _mt_spec()
    huggingface_hub.constants.ENDPOINT = _HF_MIRROR

    calls: list[str] = []

    def fake_snapshot_download(repo_id, local_dir, tqdm_class=None, **kwargs):
        calls.append(repo_id)
        raise RuntimeError("mirror already active, still fails")

    monkeypatch.setattr(
        "vrcc.download.manager.snapshot_download", fake_snapshot_download
    )

    with pytest.raises(RuntimeError, match="mirror already active"):
        manager.ensure_mt(spec)

    assert len(calls) == 1


# --------------------------------------------------------------------------
# ensure_whisper (faster-whisper / download_model branch): mirror fallback
# --------------------------------------------------------------------------

def test_ensure_whisper_falls_back_to_mirror_on_primary_failure(
    manager: DownloadManager, monkeypatch: pytest.MonkeyPatch
):
    prior_env = os.environ.get("HF_ENDPOINT")
    prior_const = huggingface_hub.constants.ENDPOINT

    calls: list[str] = []
    seen_endpoint_on_success: dict[str, str] = {}

    def fake_download_model(model_id, output_dir=None, **kwargs):
        calls.append(model_id)
        if len(calls) == 1:
            raise RuntimeError("primary host unreachable")
        seen_endpoint_on_success["endpoint"] = huggingface_hub.constants.ENDPOINT
        d = Path(output_dir)
        d.mkdir(parents=True, exist_ok=True)
        _touch(d / "model.bin")
        return output_dir

    monkeypatch.setattr(
        "vrcc.download.manager.download_model", fake_download_model
    )

    result = manager.ensure_whisper("medium")

    assert result == manager.whisper_model_dir("medium")
    assert len(calls) == 2
    assert seen_endpoint_on_success["endpoint"] == _HF_MIRROR

    assert huggingface_hub.constants.ENDPOINT == prior_const
    assert os.environ.get("HF_ENDPOINT") == prior_env


def test_ensure_whisper_both_attempts_fail_chains_and_restores(
    manager: DownloadManager, monkeypatch: pytest.MonkeyPatch
):
    prior_env = os.environ.get("HF_ENDPOINT")
    prior_const = huggingface_hub.constants.ENDPOINT

    calls: list[str] = []

    def fake_download_model(model_id, output_dir=None, **kwargs):
        calls.append(model_id)
        if len(calls) == 1:
            raise RuntimeError("primary host unreachable")
        raise RuntimeError("mirror host unreachable")

    monkeypatch.setattr(
        "vrcc.download.manager.download_model", fake_download_model
    )

    with pytest.raises(RuntimeError) as excinfo:
        manager.ensure_whisper("medium")

    assert len(calls) == 2
    assert str(excinfo.value) == "mirror host unreachable"
    assert excinfo.value.__cause__ is not None
    assert str(excinfo.value.__cause__) == "primary host unreachable"

    assert huggingface_hub.constants.ENDPOINT == prior_const
    assert os.environ.get("HF_ENDPOINT") == prior_env


def test_ensure_whisper_primary_success_no_retry_no_mutation(
    manager: DownloadManager, monkeypatch: pytest.MonkeyPatch
):
    prior_const = huggingface_hub.constants.ENDPOINT

    calls: list[str] = []
    seen_endpoints: list[str] = []

    def fake_download_model(model_id, output_dir=None, **kwargs):
        calls.append(model_id)
        seen_endpoints.append(huggingface_hub.constants.ENDPOINT)
        d = Path(output_dir)
        d.mkdir(parents=True, exist_ok=True)
        _touch(d / "model.bin")
        return output_dir

    monkeypatch.setattr(
        "vrcc.download.manager.download_model", fake_download_model
    )

    manager.ensure_whisper("medium")

    assert len(calls) == 1
    assert seen_endpoints == [prior_const]
    assert huggingface_hub.constants.ENDPOINT == prior_const


def test_ensure_whisper_no_retry_when_endpoint_already_mirror(
    manager: DownloadManager, monkeypatch: pytest.MonkeyPatch
):
    huggingface_hub.constants.ENDPOINT = _HF_MIRROR

    calls: list[str] = []

    def fake_download_model(model_id, output_dir=None, **kwargs):
        calls.append(model_id)
        raise RuntimeError("mirror already active, still fails")

    monkeypatch.setattr(
        "vrcc.download.manager.download_model", fake_download_model
    )

    with pytest.raises(RuntimeError, match="mirror already active"):
        manager.ensure_whisper("medium")

    assert len(calls) == 1
