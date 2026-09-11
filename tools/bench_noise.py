"""Deterministic noise synthesis for tools/bench_stt.py --noise.

Babble and white noise are each built once per run from a seeded RNG
(babble additionally draws material from the LibriSpeech corpus itself),
then mixed into every utterance at the requested SNR. Same seed, same
corpus, same noise type and SNR produce identical noisy audio on one
machine (two calls compared with np.array_equal); float summation order
is not guaranteed bit-identical across different machines or BLAS builds.

Dev tool only -- not packaged, no test coverage.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile

SAMPLE_RATE = 16000

# Fixed so every run and every machine builds an identical noise track;
# this is a reproducibility anchor, not a tuned value.
_SEED = 20260912

# Speakers overlaid into the babble bed. Not a tuned value, but changing it
# invalidates comparison against any existing noisy result: how many
# competing voices a model hears changes its WER independent of the SNR.
_BABBLE_SPEAKERS = 6

# The track must exceed the longest utterance in the corpus: apply_noise
# draws one slice per utterance and raises rather than loop or pad a short
# track. Measured (max(soundfile.info(p).duration for p in
# test-clean/**/*.flac) over all 2620 files): 34.955 s; 40 s leaves
# headroom for a different LibriSpeech split having a longer outlier.
_TRACK_SECONDS = 40.0


def _rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(audio, dtype=np.float64)))) if audio.size else 0.0


def _concat_speaker_audio(
    spk_dir: Path, rng: np.random.Generator, min_samples: int, exclude_ids: set[str]
) -> np.ndarray:
    """One speaker's utterances, minus anything in `exclude_ids`, read in a
    random order drawn from `rng` and concatenated until there are at least
    `min_samples` samples (the order repeats if the speaker doesn't have
    that much material alone)."""
    flacs = [p for p in sorted(spk_dir.rglob("*.flac")) if p.stem not in exclude_ids]
    if not flacs:
        # build_babble_track only chooses speakers with a non-excluded flac;
        # reaching this means that filter was bypassed, not that one ran dry.
        raise RuntimeError(f"{spk_dir} has no utterances outside the excluded set")
    order = rng.permutation(len(flacs))
    clip = np.zeros(0, dtype=np.float64)
    while clip.size < min_samples:
        for idx in order:
            audio, sr = soundfile.read(flacs[idx], dtype="float32", always_2d=False)
            if sr != SAMPLE_RATE:
                raise RuntimeError(f"{flacs[idx]} has sample rate {sr}, expected {SAMPLE_RATE}")
            clip = np.concatenate([clip, audio.astype(np.float64)])
            if clip.size >= min_samples:
                break
    return clip


def build_babble_track(
    root: Path, rng: np.random.Generator, exclude_ids: set[str]
) -> np.ndarray:
    """`_BABBLE_SPEAKERS` speakers overlaid into one `_TRACK_SECONDS` track.

    `exclude_ids` is the set of utterance ids this run will benchmark, so
    the babble bed never contains the signal it is about to be mixed with.
    Not normalized here: apply_noise normalizes the slice it actually uses,
    since a 40 s track's average level is not the level of any one slice.
    """
    n_samples = int(_TRACK_SECONDS * SAMPLE_RATE)
    candidates = [
        p.name
        for p in sorted(root.iterdir())
        if p.is_dir() and any(f.stem not in exclude_ids for f in p.rglob("*.flac"))
    ]
    if len(candidates) < _BABBLE_SPEAKERS:
        raise RuntimeError(
            f"only {len(candidates)} speakers have utterances outside this "
            f"run's {len(exclude_ids)}-utterance set; need _BABBLE_SPEAKERS="
            f"{_BABBLE_SPEAKERS} for babble (use fewer --utterances, or a "
            "smaller _BABBLE_SPEAKERS)"
        )
    chosen = rng.choice(candidates, size=_BABBLE_SPEAKERS, replace=False)

    track = np.zeros(n_samples, dtype=np.float64)
    for spk in chosen:
        clip = _concat_speaker_audio(root / spk, rng, n_samples, exclude_ids)
        offset = int(rng.integers(0, clip.size - n_samples + 1)) if clip.size > n_samples else 0
        track += clip[offset : offset + n_samples]
    return track


def build_white_track(rng: np.random.Generator) -> np.ndarray:
    """`_TRACK_SECONDS` of unit-variance Gaussian noise, unnormalized like
    the babble track (apply_noise normalizes the slice it uses)."""
    return rng.standard_normal(int(_TRACK_SECONDS * SAMPLE_RATE))


def apply_noise(
    utts: list[tuple[str, np.ndarray, str]],
    root: Path,
    noise_type: str,
    snr_db: float,
) -> list[tuple[str, np.ndarray, str]]:
    """Mix `noise_type` noise into every utterance at `snr_db`, each against
    that utterance's own RMS: a slice of the noise track is drawn, scaled
    to unit RMS, scaled again to the level `snr_db` implies, added, then
    hard-clipped to [-1, 1]. The mixed audio is deterministic on one machine
    (same corpus, noise_type and snr_db every time, see the module
    docstring); the WER a model gets on that audio is a separate measurement
    and is not claimed to repeat identically.
    """
    if noise_type not in ("babble", "white"):
        raise ValueError(f"unknown noise type: {noise_type}")

    rng = np.random.default_rng(_SEED)
    exclude_ids = {utt_id for utt_id, _, _ in utts}
    track = (
        build_babble_track(root, rng, exclude_ids)
        if noise_type == "babble"
        else build_white_track(rng)
    )

    mixed = []
    for utt_id, audio, ref in utts:
        n = len(audio)
        if len(track) < n:
            raise RuntimeError(
                f"noise track ({len(track)} samples) shorter than utterance "
                f"{utt_id} ({n} samples); raise _TRACK_SECONDS"
            )
        offset = int(rng.integers(0, len(track) - n + 1))
        segment = track[offset : offset + n].astype(np.float64)
        seg_rms = _rms(segment)
        if seg_rms == 0:
            # build_babble_track excludes speakers with no non-excluded
            # material and white noise is never all-zero; a silent segment
            # here would mean a "noisy" file is quietly clean audio instead.
            raise RuntimeError(f"noise segment for {utt_id} is silent (seg_rms=0)")
        segment = segment / seg_rms

        target_rms = _rms(audio) / (10 ** (snr_db / 20))
        noisy = audio.astype(np.float64) + segment * target_rms
        mixed.append((utt_id, np.clip(noisy, -1.0, 1.0).astype(np.float32), ref))
    return mixed
