"""Dataset access for tools/bench_stt.py: LibriSpeech test-clean download,
utterance selection and the utterance-set fingerprint."""

from __future__ import annotations

import hashlib
import shutil
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
import soundfile

DATASET_URL = "https://www.openslr.org/resources/12/test-clean.tar.gz"
SAMPLE_RATE = 16000


def ensure_dataset(data_dir: Path) -> Path:
    """Download + extract LibriSpeech test-clean once; return its root.

    Interruption-safe: the archive downloads to a .part file renamed only
    when complete, extraction is trusted only once a sentinel exists, and a
    corrupt archive is deleted with a retry hint instead of failing every
    later run."""
    root = data_dir / "LibriSpeech" / "test-clean"
    sentinel = data_dir / "LibriSpeech" / ".extracted"
    if root.is_dir() and sentinel.is_file():
        return root
    data_dir.mkdir(parents=True, exist_ok=True)
    archive = data_dir / "test-clean.tar.gz"
    if not archive.is_file():
        partial = archive.parent / (archive.name + ".part")
        print(f"downloading {DATASET_URL} ...", flush=True)
        with urllib.request.urlopen(DATASET_URL) as resp, open(partial, "wb") as out:
            done = 0
            while chunk := resp.read(1 << 20):
                out.write(chunk)
                done += len(chunk)
                if done % (50 << 20) < (1 << 20):
                    print(f"  {done >> 20} MB", flush=True)
        partial.replace(archive)
    print("extracting ...", flush=True)
    tree = data_dir / "LibriSpeech"
    if tree.exists():
        # No sentinel means the tree can't be trusted (interrupted extract, or
        # one from before the sentinel existed). The tar's members are
        # read-only, so restore writability or rmtree fails on Windows.
        for path in tree.rglob("*"):
            if path.is_file():
                path.chmod(0o644)
        shutil.rmtree(tree)
    try:
        with tarfile.open(archive) as tar:
            try:
                tar.extractall(data_dir, filter="data")
            except TypeError:
                # filter= needs 3.10.12+/3.11.4+/3.12; older interpreters in
                # the supported >=3.10 range extract without it.
                tar.extractall(data_dir)
    except tarfile.ReadError as exc:
        archive.unlink()
        raise RuntimeError(
            f"{archive} was corrupt (interrupted download?); deleted it, "
            "run this command again"
        ) from exc
    sentinel.write_text("")
    return root


def load_utterances(
    root: Path, n: int, max_seconds: float | None = None
) -> list[tuple[str, np.ndarray, str]]:
    """N (id, mono float32 16 kHz audio, reference) tuples, evenly spaced
    over the id-sorted corpus so every run picks the same speaker-diverse
    set (a plain prefix would over-represent the first speakers).

    ``max_seconds`` keeps only utterances at or under that length before
    spacing, so the short arm covers the clip lengths the app's VAD hands
    over (median ~1.8 s) rather than LibriSpeech's ~7.5 s."""
    refs: dict[str, str] = {}
    for trans in sorted(root.rglob("*.trans.txt")):
        for line in trans.read_text(encoding="utf-8").splitlines():
            utt_id, _, text = line.partition(" ")
            refs[utt_id] = text.strip()
    ids = sorted(refs)
    if max_seconds is not None:
        ids = [i for i in ids if _duration_s(_flac_path(root, i)) <= max_seconds]
    n = min(n, len(ids))
    picked = [ids[i * len(ids) // n] for i in range(n)]

    utts = []
    for utt_id in picked:
        path = _flac_path(root, utt_id)
        audio, sr = soundfile.read(path, dtype="float32", always_2d=False)
        if sr != SAMPLE_RATE:
            raise RuntimeError(f"{path} has sample rate {sr}, expected {SAMPLE_RATE}")
        utts.append((utt_id, audio, refs[utt_id]))
    return utts


def _flac_path(root: Path, utt_id: str) -> Path:
    spk, chap, _ = utt_id.split("-")
    return root / spk / chap / f"{utt_id}.flac"


def _duration_s(path: Path) -> float:
    info = soundfile.info(path)
    return info.frames / info.samplerate


def utterance_set_id(
    utts: list[tuple[str, np.ndarray, str]],
    noise: tuple[str, float] | None = None,
    max_seconds: float | None = None,
    gated: bool = False,
) -> str:
    """Short fingerprint of exactly which utterances a record measured, so
    skip/export logic can refuse to mix runs whose sets differ (a sampling
    change keeps the count at N while changing the members; a noise
    condition leaves the ids and count alone but changes every sample).

    The length bound and gated scoring change what a row means without
    always changing the ids, so each is folded in when set. A default run
    hashes exactly as it did before these options existed, which keeps
    already-exported files comparable."""
    joined = "\n".join(utt_id for utt_id, _, _ in utts)
    if noise is not None:
        joined += f"\n__noise__{noise[0]}:{noise[1]:g}"
    if max_seconds is not None:
        joined += f"\n__max_seconds__{max_seconds:g}"
    if gated:
        joined += "\n__gated__"
    return hashlib.sha256(joined.encode()).hexdigest()[:12]
