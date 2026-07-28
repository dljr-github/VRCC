"""Tests for the archive download route in ``DownloadManager`` (specs carrying
an ``archive_url``, i.e. SenseVoice) and for the per-backend required-files
list that gates both downloading and the "is it installed?" check.

That list is the sharp edge: it doubles as the snapshot allow_patterns, the
archive member allow-list AND the presence check, so a backend it gets wrong
fetches the wrong files and then reports as never-downloaded forever.
"""

from __future__ import annotations

import io
import tarfile
from pathlib import Path

import pytest

from vrcc.core.bus import EventBus
from vrcc.core.events import DownloadProgress
from vrcc.download.manager import DownloadManager, _required_files
from vrcc.stt.registry import WHISPER_MODELS

SENSEVOICE_ID = "sense-voice-small"
SENSEVOICE = WHISPER_MODELS[SENSEVOICE_ID]
PARAKEET = WHISPER_MODELS["parakeet-tdt-0.6b-v3"]


@pytest.fixture()
def bus() -> EventBus:
    return EventBus()


@pytest.fixture()
def manager(tmp_path: Path, bus: EventBus) -> DownloadManager:
    return DownloadManager(tmp_path / "models", bus)


def _tarball(members: dict[str, bytes], prefix: str = "sherpa-onnx-sense-voice") -> bytes:
    """A .tar.bz2 shaped like the sherpa-onnx release asset: every file sits
    under one top-level directory."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:bz2") as archive:
        for name, payload in members.items():
            info = tarfile.TarInfo(f"{prefix}/{name}")
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
    return buffer.getvalue()


_FULL = {
    "model.int8.onnx": b"onnx-bytes" * 100,
    "tokens.txt": b"<unk> 0\n",
    "README.md": b"not needed",
    "test_wavs/en.wav": b"not needed either",
}


class _FakeResponse(io.BytesIO):
    """urlopen's context-manager response over fixed bytes."""

    def __init__(self, payload: bytes, content_length: bool = True) -> None:
        super().__init__(payload)
        self.headers = {"Content-Length": str(len(payload))} if content_length else {}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def _serve(monkeypatch, payload: bytes, **kw):
    seen: list[str] = []

    def _urlopen(url, timeout=None):
        seen.append(url)
        return _FakeResponse(payload, **kw)

    monkeypatch.setattr("vrcc.download.manager.urllib.request.urlopen", _urlopen)
    return seen


# -- required files --------------------------------------------------------

def test_required_files_for_sensevoice_is_the_ctc_layout():
    assert _required_files(SENSEVOICE) == ["model.int8.onnx", "tokens.txt"]


def test_required_files_for_onnx_asr_keeps_the_transducer_layout():
    assert _required_files(PARAKEET) == [
        "config.json",
        "vocab.txt",
        "encoder-model.int8.onnx",
        "decoder_joint-model.int8.onnx",
    ]


def test_presence_check_uses_the_specs_own_layout(manager):
    target = manager.whisper_model_dir(SENSEVOICE_ID)
    target.mkdir(parents=True)
    assert not manager.is_whisper_downloaded(SENSEVOICE_ID)

    (target / "model.int8.onnx").write_bytes(b"x")
    assert not manager.is_whisper_downloaded(SENSEVOICE_ID)  # tokens.txt missing

    (target / "tokens.txt").write_bytes(b"x")
    assert manager.is_whisper_downloaded(SENSEVOICE_ID)


def test_presence_check_ignores_the_faster_whisper_model_bin(manager):
    """A model.bin in the directory must not fool a non-whisper spec into
    reporting installed -- that is the failure the shared layout invited."""
    target = manager.whisper_model_dir(SENSEVOICE_ID)
    target.mkdir(parents=True)
    (target / "model.bin").write_bytes(b"x")

    assert not manager.is_whisper_downloaded(SENSEVOICE_ID)


# -- archive download ------------------------------------------------------

def test_ensure_whisper_unpacks_only_the_required_files(manager, monkeypatch):
    seen = _serve(monkeypatch, _tarball(_FULL))

    target = manager.ensure_whisper(SENSEVOICE_ID)

    assert seen == [SENSEVOICE.archive_url]
    assert sorted(p.name for p in target.iterdir()) == ["model.int8.onnx", "tokens.txt"]
    assert (target / "model.int8.onnx").read_bytes() == _FULL["model.int8.onnx"]
    assert manager.is_whisper_downloaded(SENSEVOICE_ID)


def test_archive_members_land_flat_not_under_their_top_level_directory(
    manager, monkeypatch
):
    _serve(monkeypatch, _tarball(_FULL, prefix="some/deep/prefix"))

    target = manager.ensure_whisper(SENSEVOICE_ID)

    assert (target / "model.int8.onnx").is_file()
    assert not (target / "some").exists()


def test_archive_paths_that_try_to_escape_are_ignored(manager, monkeypatch):
    """Only the basename of a member is ever used, so a traversal path cannot
    write outside the model directory. The member still lands, flattened."""
    _serve(monkeypatch, _tarball(_FULL, prefix="../../../../etc"))

    target = manager.ensure_whisper(SENSEVOICE_ID)

    assert (target / "model.int8.onnx").is_file()
    assert not (target.parent.parent / "etc").exists()


def test_incomplete_archive_raises_and_names_what_is_missing(manager, monkeypatch):
    _serve(monkeypatch, _tarball({"model.int8.onnx": b"x"}))

    with pytest.raises(RuntimeError, match="tokens.txt"):
        manager.ensure_whisper(SENSEVOICE_ID)

    assert not manager.is_whisper_downloaded(SENSEVOICE_ID)


def test_archive_download_publishes_byte_progress_then_done(manager, bus, monkeypatch):
    events: list[DownloadProgress] = []
    bus.subscribe(DownloadProgress, events.append)
    payload = _tarball({"model.int8.onnx": b"z" * (4 * 1024 * 1024), "tokens.txt": b"t"})
    _serve(monkeypatch, payload)

    manager.ensure_whisper(SENSEVOICE_ID)

    assert events, "no progress published"
    assert all(e.model_id == SENSEVOICE_ID for e in events)
    assert events[-1].done is True
    byte_events = [e for e in events if not e.done]
    assert byte_events, "no byte-level progress"
    assert all(e.total == len(payload) for e in byte_events)
    # Monotonic and bounded by the total.
    assert byte_events == sorted(byte_events, key=lambda e: e.downloaded)
    assert byte_events[-1].downloaded <= len(payload)


def test_archive_download_without_content_length_still_completes(manager, monkeypatch):
    # No Content-Length means no denominator, so progress stays silent rather
    # than publishing a bogus total -- but the download must still work.
    _serve(monkeypatch, _tarball(_FULL), content_length=False)

    target = manager.ensure_whisper(SENSEVOICE_ID)

    assert manager.is_whisper_downloaded(SENSEVOICE_ID)
    assert (target / "tokens.txt").is_file()


def test_already_downloaded_archive_model_skips_the_fetch(manager, monkeypatch):
    target = manager.whisper_model_dir(SENSEVOICE_ID)
    target.mkdir(parents=True)
    for name in _required_files(SENSEVOICE):
        (target / name).write_bytes(b"x")

    def _boom(*a, **k):  # pragma: no cover - must never run
        raise AssertionError("re-downloaded an installed model")

    monkeypatch.setattr("vrcc.download.manager.urllib.request.urlopen", _boom)

    assert manager.ensure_whisper(SENSEVOICE_ID) == target


def test_delete_removes_an_archive_model(manager, monkeypatch):
    _serve(monkeypatch, _tarball(_FULL))
    manager.ensure_whisper(SENSEVOICE_ID)

    manager.delete("whisper", SENSEVOICE_ID)

    assert not manager.is_whisper_downloaded(SENSEVOICE_ID)
