"""End-to-end smoke test: WAV file -> VAD/segmenter -> STT -> MT -> chatbox.

Runs the real (non-GUI) engine stack over a WAV file on disk and prints what
the app would have sent to VRChat, plus per-stage timings::

    python scripts/smoke_e2e.py path/to.wav [--target Japanese]
        [--stt-model tiny] [--mt-model nllb-600M-int8] [--source English]
        [--device auto|cpu]

Models are downloaded (once) via :class:`DownloadManager` into the same
default models directory the real app uses, so a smoke run primes the cache
for the GUI and vice versa. No OSC/UDP traffic is produced: the
:class:`ChatboxSender` is given a fake client that prints to stdout instead.

The script is import-friendly for tests: :func:`run_smoke` does all the work
and returns a :class:`SmokeResult`; :func:`main` is just argument parsing
around it. Exit code is 0 if at least one phrase survived STT's quality
gates, 1 otherwise.
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Allow running as `python scripts/smoke_e2e.py` from a source checkout
# without an installed package.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from vrcc.audio.segmenter import FRAME, Segmenter, SegFinal  # noqa: E402
from vrcc.audio.vad import StreamingVad  # noqa: E402
from vrcc.core import hardware  # noqa: E402
from vrcc.core.bus import EventBus  # noqa: E402
from vrcc.core.config import (  # noqa: E402
    OscConfig,
    SttConfig,
    TranslateConfig,
    VadConfig,
    default_paths,
)
from vrcc.core.events import (  # noqa: E402
    ChatboxSent,
    DownloadProgress,
    EngineStateChanged,
)
from vrcc.core.languages import LANGUAGES, get  # noqa: E402
from vrcc.download.manager import DownloadManager  # noqa: E402
from vrcc.osc.chatbox import ChatboxSender, format_message  # noqa: E402
from vrcc.stt.engine import SttEngine  # noqa: E402
from vrcc.translate.engine import TranslateEngine  # noqa: E402
from vrcc.translate.registry import MT_MODELS  # noqa: E402

SAMPLE_RATE = 16000
_TRAILING_SILENCE_S = 1.0
_CHATBOX_SEND_TIMEOUT_S = 5.0


@dataclass
class PhraseResult:
    utterance_id: int
    text: str
    language: str
    stt_seconds: float
    translations: list[tuple[str, str]]
    mt_seconds: float
    chatbox_message: str


@dataclass
class SmokeResult:
    audio_seconds: float
    segmentation_wall_s: float
    finals: int
    phrases: list[PhraseResult] = field(default_factory=list)

    @property
    def stt_seconds(self) -> float:
        return sum(p.stt_seconds for p in self.phrases)

    @property
    def mt_seconds(self) -> float:
        return sum(p.mt_seconds for p in self.phrases)


def load_wav(path: Path) -> np.ndarray:
    """Read `path` into mono float32 at 16 kHz, in [-1, 1].

    Only 16-bit PCM WAVs are supported (that is what every TTS/recorder
    produces by default). Multi-channel audio is downmixed by averaging;
    other sample rates are resampled with soxr.
    """
    with wave.open(str(path), "rb") as wav:
        n_channels = wav.getnchannels()
        sampwidth = wav.getsampwidth()
        rate = wav.getframerate()
        raw = wav.readframes(wav.getnframes())

    if sampwidth != 2:
        raise ValueError(
            f"{path}: only 16-bit PCM WAV is supported, got sample width "
            f"{sampwidth * 8} bits"
        )

    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if n_channels > 1:
        samples = samples.reshape(-1, n_channels).mean(axis=1)

    if rate != SAMPLE_RATE:
        import soxr

        samples = soxr.resample(samples, rate, SAMPLE_RATE).astype(np.float32)

    return samples


def segment(samples: np.ndarray, vad_cfg: VadConfig) -> tuple[list[SegFinal], float]:
    """Feed `samples` through StreamingVad+Segmenter in 512-sample frames.

    A second of trailing silence is appended first so the last utterance
    always crosses the finalize-silence threshold. Returns the SegFinal
    events and the wall time spent segmenting.
    """
    silence = np.zeros(int(SAMPLE_RATE * _TRAILING_SILENCE_S), dtype=np.float32)
    stream = np.concatenate([samples.astype(np.float32), silence])

    pad = (-len(stream)) % FRAME
    if pad:
        stream = np.concatenate([stream, np.zeros(pad, dtype=np.float32)])

    vad = StreamingVad(threshold=vad_cfg.threshold)
    segmenter = Segmenter(vad_cfg, vad.prob, sample_rate=SAMPLE_RATE)

    finals: list[SegFinal] = []
    start = time.perf_counter()
    for offset in range(0, len(stream), FRAME):
        for event in segmenter.process(stream[offset : offset + FRAME]):
            if isinstance(event, SegFinal):
                finals.append(event)
    wall = time.perf_counter() - start
    return finals, wall


class _StdoutOscClient:
    """Stand-in for pythonosc's SimpleUDPClient: prints instead of sending."""

    def send_message(self, address: str, value) -> None:
        print(f"  [osc->stdout] {address} {value!r}")


def _send_via_chatbox(bus: EventBus, osc_cfg: OscConfig, messages: list[tuple[str, int]]) -> None:
    """Push each formatted message through a real ChatboxSender (fake client).

    Waits for the ChatboxSent event per message so the sender's worker thread
    has actually processed it before the next submit (and before stop()).
    """
    sent = threading.Event()
    unsubscribe = bus.subscribe(ChatboxSent, lambda _e: sent.set())
    sender = ChatboxSender(
        osc_cfg, bus, client_factory=lambda ip, port: _StdoutOscClient()
    )
    sender.start()
    try:
        for text, utterance_id in messages:
            sent.clear()
            sender.submit(text, utterance_id)
            if not sent.wait(timeout=_CHATBOX_SEND_TIMEOUT_S):
                print("  [warn] chatbox send timed out", file=sys.stderr)
    finally:
        sender.stop()
        unsubscribe()


def _print_download_progress(bus: EventBus) -> None:
    """Log coarse download progress (every ~10%) so big downloads show life."""
    last = {}

    def on_progress(event: DownloadProgress) -> None:
        if event.done:
            print(f"  [download] {event.model_id}: done")
            return
        pct = 100 * event.downloaded // max(event.total, 1)
        if pct >= last.get(event.model_id, -10) + 10:
            last[event.model_id] = pct
            mib = event.downloaded / (1024 * 1024)
            total_mib = event.total / (1024 * 1024)
            print(f"  [download] {event.model_id}: {mib:.0f}/{total_mib:.0f} MiB ({pct}%)")

    bus.subscribe(DownloadProgress, on_progress)


def _resolve_source_language(source_cfg: str, detected_whisper: str):
    """Mirror the pipeline's behavior-5 rule: a configured display name wins;
    'auto' maps the detected Whisper code to the first matching Language,
    falling back to English."""
    if source_cfg != "auto":
        return get(source_cfg)
    for lang in LANGUAGES.values():
        if lang.whisper == detected_whisper:
            return lang
    return get("English")


def run_smoke(
    wav_path: Path,
    *,
    target: str = "Japanese",
    stt_model: str = "tiny",
    mt_model: str = "nllb-600M-int8",
    source: str = "English",
    device: str = "auto",
    use_chatbox: bool = True,
) -> SmokeResult:
    """Run the full WAV -> chatbox-text path and return what happened."""
    spec = MT_MODELS[mt_model]
    target_lang = get(target)
    if source != "auto":
        get(source)  # fail fast on an unknown source language

    bus = EventBus()
    _print_download_progress(bus)
    bus.subscribe(
        EngineStateChanged,
        lambda e: print(f"  [engine] {e.engine}: {e.state}"
                        + (f" ({e.detail})" if e.detail else "")),
    )

    paths = default_paths(portable=False)
    print(f"models dir: {paths.models_dir}")
    dm = DownloadManager(paths.models_dir, bus)

    print(f"ensuring whisper '{stt_model}' "
          f"({'cached' if dm.is_whisper_downloaded(stt_model) else 'downloading'})...")
    whisper_dir = dm.ensure_whisper(stt_model)
    print(f"ensuring mt '{mt_model}' "
          f"({'cached' if dm.is_mt_downloaded(spec) else 'downloading'})...")
    mt_dir = dm.ensure_mt(spec)

    hardware.setup_cuda_dlls()

    samples = load_wav(wav_path)
    audio_seconds = len(samples) / SAMPLE_RATE
    print(f"wav: {wav_path} ({audio_seconds:.2f}s at {SAMPLE_RATE} Hz mono)")

    finals, seg_wall = segment(samples, VadConfig())
    print(f"segmentation: {len(finals)} utterance(s) in {seg_wall:.2f}s wall")
    result = SmokeResult(
        audio_seconds=audio_seconds, segmentation_wall_s=seg_wall, finals=len(finals)
    )
    if not finals:
        return result

    stt_cfg = SttConfig(model=stt_model, device=device, source_language=source)
    stt = SttEngine(stt_cfg, whisper_dir, bus)
    t0 = time.perf_counter()
    stt.load()
    stt.warm_up()
    print(f"stt load+warmup: {time.perf_counter() - t0:.2f}s")

    mt_cfg = TranslateConfig(model=mt_model, device=device, targets=[target])
    mt = TranslateEngine(spec, mt_dir, mt_cfg, bus)
    t0 = time.perf_counter()
    mt.load()
    mt.warm_up()
    print(f"mt load+warmup: {time.perf_counter() - t0:.2f}s")

    osc_cfg = OscConfig()
    messages: list[tuple[str, int]] = []
    try:
        for final in finals:
            t0 = time.perf_counter()
            stt_result = stt.transcribe(final.samples)
            stt_s = time.perf_counter() - t0
            if stt_result is None:
                print(f"[utt {final.utterance_id}] stt {stt_s:.2f}s: "
                      "(gated: no reliable text)")
                continue
            print(f"[utt {final.utterance_id}] stt {stt_s:.2f}s "
                  f"({len(final.samples) / SAMPLE_RATE:.2f}s audio, "
                  f"lang={stt_result.language}, "
                  f"avg_logprob={stt_result.avg_logprob:.2f}): "
                  f"{stt_result.text!r}")

            t0 = time.perf_counter()
            translations = mt.translate(
                stt_result.text,
                _resolve_source_language(source, stt_result.language),
                [target_lang],
            )
            mt_s = time.perf_counter() - t0
            for name, text in translations:
                print(f"[utt {final.utterance_id}] mt {mt_s:.2f}s -> {name}: {text!r}")

            message = format_message(stt_result.text, translations, osc_cfg)
            print(f"[utt {final.utterance_id}] chatbox message:\n{message}")
            messages.append((message, final.utterance_id))
            result.phrases.append(
                PhraseResult(
                    utterance_id=final.utterance_id,
                    text=stt_result.text,
                    language=stt_result.language,
                    stt_seconds=stt_s,
                    translations=translations,
                    mt_seconds=mt_s,
                    chatbox_message=message,
                )
            )
    finally:
        stt.unload()
        mt.unload()

    if use_chatbox and messages:
        print("sending via ChatboxSender (fake stdout client):")
        _send_via_chatbox(bus, osc_cfg, messages)

    print(
        f"totals: segmentation {result.segmentation_wall_s:.2f}s wall, "
        f"stt {result.stt_seconds:.2f}s, mt {result.mt_seconds:.2f}s "
        f"for {result.audio_seconds:.2f}s of audio"
    )
    return result


def main(argv: list[str] | None = None) -> int:
    # Windows consoles/pipes often default to a legacy codepage (cp1252) that
    # cannot represent translated text (e.g. Japanese); force UTF-8 output.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        description="VRCC end-to-end smoke: WAV -> VAD -> STT -> MT -> chatbox text"
    )
    parser.add_argument("wav", type=Path, help="path to a 16-bit PCM WAV file")
    parser.add_argument("--target", default="Japanese",
                        help="target language display name (default: Japanese)")
    parser.add_argument("--stt-model", default="tiny",
                        help="faster-whisper model id (default: tiny)")
    parser.add_argument("--mt-model", default="nllb-600M-int8",
                        choices=sorted(MT_MODELS), help="MT model id")
    parser.add_argument("--source", default="English",
                        help="source language display name, or 'auto' (default: English)")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu"),
                        help="engine device (default: auto)")
    args = parser.parse_args(argv)

    if not args.wav.is_file():
        parser.error(f"WAV file not found: {args.wav}")
    if args.target not in LANGUAGES:
        parser.error(
            f"--target {args.target!r} is not a known language "
            f"(see vrcc.core.languages.LANGUAGES)"
        )
    if args.source != "auto" and args.source not in LANGUAGES:
        parser.error(
            f"--source {args.source!r} is not a known language "
            f"(see vrcc.core.languages.LANGUAGES; 'auto' = detect)"
        )

    result = run_smoke(
        args.wav,
        target=args.target,
        stt_model=args.stt_model,
        mt_model=args.mt_model,
        source=args.source,
        device=args.device,
    )

    if not result.phrases:
        print("FAIL: no phrase produced", file=sys.stderr)
        return 1
    print(f"OK: {len(result.phrases)} phrase(s) produced")
    return 0


if __name__ == "__main__":
    sys.exit(main())
