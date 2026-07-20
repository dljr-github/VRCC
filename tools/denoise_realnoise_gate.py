"""Pre-default-on real-noise gate for the GTCRN denoiser.

The shipped noise-suppression toggle (``AudioConfig.denoise_enabled``) ships
off by default because its win was only ever measured on synthetic noise
(see ``docs/superpowers/specs/2026-07-21-noise-suppression-design.md``).
This script is the promised follow-up check: it mixes the clean speech
fixtures committed under ``tests/integration/audio/`` with REAL recorded
noise (cafe, fan, keyboard, TV/crowd babble, ...) at a set of target SNRs,
transcribes both the raw mix and the denoised mix with the real cached
Whisper model, and prints whether the gentle denoise still beats or matches
raw transcription on real noise. If it does, the toggle default may be
flipped to on in a follow-up change; if it does not, the feature stays
shipped opt-in and the limitation gets recorded in the spec.

This is deliberately not a pytest: it needs real noise recordings that are
not committed to the repo (licensing is unclear and/or the files are too
large) and a Whisper model already downloaded on this machine, so it runs
as a standalone dev tool instead of a CI-gated test.

Usage:
    python tools/denoise_realnoise_gate.py --noise-dir path/to/real_noise
    python tools/denoise_realnoise_gate.py --noise-dir path/to/real_noise \\
        --snr -5,0,5,10 --strength 0.5

``--noise-dir`` holds one or more real recorded noise WAVs (any sample
rate, resampled to 16 kHz mono; 16-bit PCM, the same constraint every other
fixture loader in this repo has). Cafe, fan, keyboard clatter and TV/crowd
babble are the scenarios the design spec calls out. Each noise clip is
mixed with every clean speech fixture found directly under
``tests/integration/audio/`` (its ``noisy/`` subdirectory holds fixtures
already mixed with synthetic noise for a different test, and is skipped)
at every ``--snr`` value, scaled so the ratio of the noise RMS to the
speech's active RMS gives the target SNR. Each mix is transcribed raw and
after the shipped ``Denoiser.process`` at ``--strength`` (default 0.5, the
shipped default), fed in 512-sample frames exactly as ``MicSource`` feeds
capture frames.

Exit codes: 0 the gentle denoise held (PASS); 2 it regressed on real noise
(FAIL, a measured result, not a crash); 1 the environment is not ready (no
cached Whisper model, or no noise files found).
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tests.integration._harness import (  # noqa: E402
    FRAME,
    _smoke,
    find_cached_whisper,
    load_fixture,
    norm_words,
    wer,
)
from vrcc.audio.denoise import Denoiser  # noqa: E402
from vrcc.core.bus import EventBus  # noqa: E402
from vrcc.core.config import SttConfig  # noqa: E402

_SPEECH_DIR = _REPO_ROOT / "tests" / "integration" / "audio"
_ACTIVE_FRAME = 320  # 20 ms at 16 kHz


def _active_rms(signal: np.ndarray, frame_len: int = _ACTIVE_FRAME) -> float:
    """RMS over frames within 20 dB of the loudest frame in ``signal``, an
    ITU-T P.56-style approximation of the active-speech level without a full
    voice activity detector, so leading or trailing silence in a fixture
    does not dilute the target SNR."""
    n = signal.size
    if n == 0:
        return 0.0
    pad = (-n) % frame_len
    padded = np.concatenate([signal, np.zeros(pad, dtype=np.float32)]) if pad else signal
    frames = padded.reshape(-1, frame_len).astype(np.float64)
    frame_rms = np.sqrt(np.mean(frames ** 2, axis=1))
    peak = float(frame_rms.max())
    if peak <= 0.0:
        return 0.0
    active = frame_rms[frame_rms >= 0.1 * peak]
    if active.size == 0:
        active = frame_rms
    return float(np.sqrt(np.mean(active ** 2)))


def mix_at_snr(
    speech: np.ndarray, noise: np.ndarray, snr_db: float, rng: np.random.Generator
) -> np.ndarray:
    """Loop/crop ``noise`` to ``speech``'s length at a random offset, scale
    it so the ratio of the speech's active RMS to the noise RMS matches
    ``snr_db``, and sum. Standard SNR mixing:
    ``noise *= speech_rms * 10**(-snr/20) / noise_rms``. Clips the result
    only if the sum would exceed full scale (a rare, loud combination),
    dividing by its own peak so the ratio between the two signals is kept."""
    speech = np.asarray(speech, dtype=np.float32)
    noise = np.asarray(noise, dtype=np.float32)
    if noise.size == 0 or speech.size == 0:
        return speech.copy()
    if noise.size < speech.size:
        reps = -(-speech.size // noise.size)
        noise = np.tile(noise, reps)
    span = noise.size - speech.size
    start = int(rng.integers(0, span + 1)) if span > 0 else 0
    segment = noise[start:start + speech.size]

    speech_rms = _active_rms(speech)
    noise_rms = float(np.sqrt(np.mean(segment.astype(np.float64) ** 2)))
    if speech_rms <= 0.0 or noise_rms <= 0.0:
        return speech.copy()

    scale = speech_rms * (10.0 ** (-snr_db / 20.0)) / noise_rms
    mixed = speech + segment * scale
    peak = float(np.max(np.abs(mixed))) if mixed.size else 0.0
    if peak > 1.0:
        mixed = mixed / peak
    return mixed.astype(np.float32)


def _denoise(sig: np.ndarray, strength: float) -> np.ndarray:
    """Run ``sig`` through a fresh shipped ``Denoiser`` in 512-sample
    frames, exactly as ``MicSource._emit`` feeds capture frames and as
    ``tests/integration/test_denoise_accuracy.py`` measures the
    synthetic-noise win. A fresh instance per clip keeps recurrent state
    from leaking between unrelated mixes."""
    denoiser = Denoiser()
    denoiser.configure(enabled=True, strength=strength)
    frames = [
        denoiser.process(sig[i:i + FRAME])
        for i in range(0, len(sig) - FRAME + 1, FRAME)
    ]
    return np.concatenate(frames) if frames else sig[:0]


def _word_label(filename: str) -> str:
    """Ground-truth word for a ``word_*.wav`` fixture, read from its
    filename: the segment before the first ``__`` (or the whole stem), with
    a leading ``word_`` stripped. Matches
    ``tests/integration/audio/noisy/manifest.json``'s naming convention
    (``word_hello__babble__snr0.wav`` gives ``hello``)."""
    stem = Path(filename).stem
    label = stem.split("__", 1)[0]
    if label.startswith("word_"):
        label = label[len("word_"):]
    return label


def discover_speech_clips(speech_dir: Path) -> tuple[list[str], list[str]]:
    """Clean speech fixtures directly under ``speech_dir`` (skips its
    ``noisy/`` subdirectory, which holds fixtures already mixed with
    synthetic noise for a different test). Files named ``word_*.wav`` score
    as single words; everything else scores as a sentence."""
    words: list[str] = []
    sentences: list[str] = []
    for path in sorted(speech_dir.glob("*.wav")):
        (words if path.name.startswith("word_") else sentences).append(path.name)
    return words, sentences


def discover_noise_files(noise_dir: Path) -> list[Path]:
    return sorted(p for p in noise_dir.iterdir() if p.suffix.lower() == ".wav")


@dataclass
class _Tally:
    word_raw_ok: int = 0
    word_mix_ok: int = 0
    word_total: int = 0
    sent_raw_wer: list = field(default_factory=list)
    sent_mix_wer: list = field(default_factory=list)


@dataclass
class GateResult:
    totals: _Tally = field(default_factory=_Tally)
    per_noise: dict = field(default_factory=dict)
    skipped_sentences: list = field(default_factory=list)


def run_gate(
    stt,
    speech_dir: Path,
    noise_dir: Path,
    snrs: list[float],
    strength: float,
    seed: int = 0,
) -> GateResult:
    """Mix every clean speech fixture in ``speech_dir`` with every noise
    clip in ``noise_dir`` at every SNR in ``snrs``, transcribe raw and
    denoised, and tally correctness (words) / WER (sentences), both overall
    and per noise file. Deterministic for fixed inputs (``seed`` drives the
    noise crop offsets)."""
    words, sentences = discover_speech_clips(speech_dir)
    noise_paths = discover_noise_files(noise_dir)
    noise_clips = [(p.name, _smoke.load_wav(p)) for p in noise_paths]

    rng = np.random.default_rng(seed)
    result = GateResult()
    for name, _ in noise_clips:
        result.per_noise[name] = _Tally()

    for word_name in words:
        label = _word_label(word_name)
        speech = load_fixture(word_name)
        for noise_name, noise in noise_clips:
            tally = result.per_noise[noise_name]
            for snr in snrs:
                mixed = mix_at_snr(speech, noise, snr, rng)
                raw = stt.transcribe(mixed)
                mix = stt.transcribe(_denoise(mixed, strength))
                raw_ok = int(raw is not None and " ".join(norm_words(raw.text)) == label)
                mix_ok = int(mix is not None and " ".join(norm_words(mix.text)) == label)
                for t in (result.totals, tally):
                    t.word_raw_ok += raw_ok
                    t.word_mix_ok += mix_ok
                    t.word_total += 1

    for sent_name in sentences:
        speech = load_fixture(sent_name)
        reference = stt.transcribe(speech)
        ref_text = reference.text if reference is not None else ""
        if not ref_text.strip():
            result.skipped_sentences.append(sent_name)
            continue
        for noise_name, noise in noise_clips:
            tally = result.per_noise[noise_name]
            for snr in snrs:
                mixed = mix_at_snr(speech, noise, snr, rng)
                raw = stt.transcribe(mixed)
                mix = stt.transcribe(_denoise(mixed, strength))
                raw_wer = wer(ref_text, raw.text if raw is not None else "")
                mix_wer = wer(ref_text, mix.text if mix is not None else "")
                for t in (result.totals, tally):
                    t.sent_raw_wer.append(raw_wer)
                    t.sent_mix_wer.append(mix_wer)

    return result


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def print_report(
    result: GateResult, model_id: str, snrs: list[float], strength: float, noise_dir: Path
) -> bool:
    """Print the comparison and a PASS/FAIL line. PASS means the denoised
    arm was at least as good as raw on both measures: single-word
    correctness (count) and mean sentence WER (lower is better). Returns
    whether it passed."""
    totals = result.totals
    print("== VRCC real-noise denoiser gate ==")
    print(f"noise dir:   {noise_dir}")
    print(f"noise files: {len(result.per_noise)}")
    print(f"SNRs (dB):   {', '.join(str(s) for s in snrs)}")
    print(f"strength:    {strength}")
    print(f"whisper:     {model_id} (cached, cpu)")
    print()

    if result.skipped_sentences:
        print(
            "skipped (clean audio produced no reference transcript): "
            + ", ".join(result.skipped_sentences)
        )
        print()

    word_pass = True
    print("Single-word correctness:")
    if totals.word_total:
        raw_pct = 100.0 * totals.word_raw_ok / totals.word_total
        mix_pct = 100.0 * totals.word_mix_ok / totals.word_total
        print(f"  raw:      {totals.word_raw_ok}/{totals.word_total} ({raw_pct:.1f}%)")
        print(f"  denoised: {totals.word_mix_ok}/{totals.word_total} ({mix_pct:.1f}%)")
        word_pass = totals.word_mix_ok >= totals.word_raw_ok
    else:
        print(
            "  no clean single-word fixtures found under tests/integration/audio "
            "(only word_*.wav files count); word scoring skipped"
        )
    print()

    sent_pass = True
    print("Sentence WER (mean; reference is each clip's own clean transcript):")
    if totals.sent_raw_wer:
        raw_wer_mean = _mean(totals.sent_raw_wer)
        mix_wer_mean = _mean(totals.sent_mix_wer)
        print(f"  raw:      {raw_wer_mean:.2%} over {len(totals.sent_raw_wer)} trials")
        print(f"  denoised: {mix_wer_mean:.2%} over {len(totals.sent_mix_wer)} trials")
        sent_pass = mix_wer_mean <= raw_wer_mean
    else:
        print("  no sentence clips scored; sentence WER check skipped")
    print()

    if len(result.per_noise) > 1:
        print("Per noise file:")
        for name, tally in result.per_noise.items():
            bits = []
            if tally.word_total:
                bits.append(f"words {tally.word_raw_ok}->{tally.word_mix_ok}/{tally.word_total}")
            if tally.sent_raw_wer:
                bits.append(
                    f"sentence WER {_mean(tally.sent_raw_wer):.1%}"
                    f"->{_mean(tally.sent_mix_wer):.1%}"
                )
            print(f"  {name}: {'; '.join(bits) if bits else 'no trials'}")
        print()

    passed = word_pass and sent_pass
    verb = "held or beat" if passed else "regressed against"
    print(f"RESULT: {'PASS' if passed else 'FAIL'} (denoised {verb} raw on real noise)")
    return passed


def _parse_snr_list(value: str) -> list[float]:
    try:
        parsed = [float(x.strip()) for x in value.split(",") if x.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid --snr list: {value!r}") from exc
    if not parsed:
        raise argparse.ArgumentTypeError(f"invalid --snr list: {value!r}")
    return parsed


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Exit codes: 0 PASS, 2 FAIL (measured regression), 1 environment "
            "not ready (no cached whisper model, or no noise files found)."
        ),
    )
    parser.add_argument(
        "--noise-dir", type=Path, required=True,
        help="directory of real recorded noise WAVs (16-bit PCM, any sample rate)",
    )
    parser.add_argument(
        "--snr", type=_parse_snr_list, default="0,5,10",
        help="comma-separated target SNRs in dB (default: 0,5,10)",
    )
    parser.add_argument(
        "--strength", type=float, default=0.5,
        help="denoiser dry/wet strength, 0..1 (default: 0.5, the shipped default)",
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="RNG seed for noise crop offsets, for reproducible runs (default: 0)",
    )
    args = parser.parse_args()

    if not args.noise_dir.is_dir():
        parser.error(f"--noise-dir is not a directory: {args.noise_dir}")
    if not 0.0 <= args.strength <= 1.0:
        parser.error("--strength must be between 0 and 1")

    if not discover_noise_files(args.noise_dir):
        print(f"no .wav files found in {args.noise_dir}", file=sys.stderr)
        return 1

    found = find_cached_whisper()
    if found is None:
        print(
            "no cached Whisper model found on this machine; run the app once "
            "(or `python tools/bench_stt.py --download-only`) so a model is "
            "cached, then re-run this gate",
            file=sys.stderr,
        )
        return 1
    model_id, whisper_dir = found

    from vrcc.stt.engine import SttEngine

    cfg = SttConfig(model=model_id, device="cpu", source_language="English")
    engine = SttEngine(cfg, whisper_dir, EventBus())
    engine.load()
    try:
        result = run_gate(
            engine, _SPEECH_DIR, args.noise_dir, args.snr, args.strength, args.seed
        )
    finally:
        engine.unload()

    passed = print_report(result, model_id, args.snr, args.strength, args.noise_dir)
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
