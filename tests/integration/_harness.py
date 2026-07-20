"""Shared helpers for the real-Whisper integration tests: fixture loading,
word error rate, the mic-gain path and the two ways audio is fed through the
real engine stack (segmenter-only finals, and the threaded pipeline).

Nothing here is a test itself; every ``test_*`` module in this package
imports from it. Keeps each test module focused on one behavior instead of
re-deriving the harness plumbing per file.
"""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Callable

import numpy as np

from vrcc.audio.gain import GainProcessor
from vrcc.core.bus import EventBus
from vrcc.core.config import ConfigStore, VadConfig, default_paths
from vrcc.core.engine_stack import build_engine_stack
from vrcc.core.events import PhraseRecognized
from vrcc.download.manager import DownloadManager

FRAME = 512
SAMPLE_RATE = 16000

_AUDIO_DIR = Path(__file__).resolve().parent / "audio"
_SMOKE_PATH = Path(__file__).resolve().parents[2] / "scripts" / "smoke_e2e.py"

_WHISPER_PREFERENCE = ("small", "base", "tiny")


def _load_smoke_module():
    spec = importlib.util.spec_from_file_location("vrcc_smoke_e2e", _SMOKE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_smoke = _load_smoke_module()


def find_cached_whisper() -> tuple[str, Path] | None:
    """The first model in :data:`_WHISPER_PREFERENCE` already downloaded on
    this machine, as ``(model_id, whisper_dir)``, or ``None`` if none are
    cached. Never downloads."""
    paths = default_paths(portable=False)
    manager = DownloadManager(paths.models_dir, EventBus())
    for model_id in _WHISPER_PREFERENCE:
        if manager.is_whisper_downloaded(model_id):
            return model_id, manager.whisper_model_dir(model_id)
    return None


# -- fixtures ----------------------------------------------------------------


def load_fixture(name: str) -> np.ndarray:
    """Load one of the ``tests/integration/audio`` WAV fixtures as mono
    float32 at 16 kHz."""
    return _smoke.load_wav(_AUDIO_DIR / name)


# -- word error rate -----------------------------------------------------------


def norm_words(text: str) -> list[str]:
    """Lowercase ``text`` and strip punctuation, returning its words. Used on
    both sides of a WER comparison so wording differences (not punctuation or
    case) are what get scored."""
    keep = [ch if (ch.isalnum() or ch.isspace()) else " " for ch in (text or "").lower()]
    return "".join(keep).split()


def wer(ref: str, hyp: str) -> float:
    """Word error rate of ``hyp`` against ``ref`` (Levenshtein distance over
    words, divided by the reference word count). An empty reference scores 0
    only if ``hyp`` is also empty."""
    r, h = norm_words(ref), norm_words(hyp)
    if not r:
        return 0.0 if not h else 1.0
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            cur = d[j]
            d[j] = min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
            prev = cur
    return d[len(h)] / len(r)


# -- gain path -----------------------------------------------------------------


def gain_frames(audio: np.ndarray, auto: bool, scale: float = 1.0) -> np.ndarray:
    """Apply ``GainProcessor`` to ``audio`` (scaled by ``scale`` first) one
    512-sample frame at a time, exactly as ``MicSource._emit`` feeds capture
    frames through gain before they reach the VAD/STT. A trailing partial
    frame (shorter than 512 samples) is dropped, mirroring the rechunker's
    remainder carry."""
    proc = GainProcessor()
    proc.configure(0.0, auto)
    scaled = np.asarray(audio, dtype=np.float32) * scale
    out = [
        proc.process(scaled[i : i + FRAME].copy())
        for i in range(0, len(scaled) - FRAME + 1, FRAME)
    ]
    return np.concatenate(out) if out else scaled[:0]


# -- segmenter-only finals -------------------------------------------------


def finals_text(audio: np.ndarray, stt, **vad_overrides) -> list[str]:
    """Run ``audio`` through ``StreamingVad`` + ``Segmenter`` (via
    ``scripts/smoke_e2e.segment``, one 512-sample frame at a time) and
    transcribe each finalized utterance with ``stt``. Returns the recognized
    texts in order; an utterance that fails STT's quality gates is skipped
    (mirrors the real pipeline's ``forward_final``)."""
    vad_cfg = VadConfig(**vad_overrides)
    finals, _ = _smoke.segment(audio, vad_cfg)
    texts = []
    for final in finals:
        result = stt.transcribe(final.samples)
        if result is not None and result.text.strip():
            texts.append(result.text)
    return texts


# -- threaded pipeline -------------------------------------------------------


class _FakeChatbox:
    def submit_message(self, *a, **k) -> None:
        pass

    def set_typing(self, value) -> None:
        pass

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def reconfigure(self, *a, **k) -> None:
        pass

    def reconfigure_rate(self, *a, **k) -> None:
        pass


class _FakeSource:
    def start(self, on_frame: Callable[[np.ndarray], None]) -> None:
        self._on_frame = on_frame

    def stop(self) -> None:
        pass


def _scratch_config_path() -> Path:
    # A path guaranteed not to exist: ConfigStore.load() then falls back to
    # AppConfig() defaults without touching disk, and nothing here ever calls
    # save(), so no file is ever written.
    return Path(tempfile.gettempdir()) / f"vrcc_integration_{uuid.uuid4().hex}.json"


def pipeline_phrases(
    audio: np.ndarray, stt, *, drain_s: float = 3.0, **vad_overrides
) -> list[str]:
    """Feed ``audio`` through the real threaded ``Pipeline`` (built via
    ``build_engine_stack``, with a fake chatbox/source and translation/mute
    disabled) at real-time pace, and return the ``PhraseRecognized`` texts in
    the order they were published.

    ``vad_overrides`` are applied to the fresh ``AppConfig``'s ``vad`` section
    before the stack is built (e.g. ``sentence_min_words=3``). ``stt`` must
    already be loaded; it is reused, not created from config.
    """
    store = ConfigStore(_scratch_config_path())
    store.load()
    for key, value in vad_overrides.items():
        setattr(store.config.vad, key, value)
    store.config.osc.send_to_vrchat = False

    bus = EventBus()
    stack = build_engine_stack(
        store,
        bus,
        default_paths(portable=False),
        stt_engine=stt,
        mt_engine=None,
        chatbox=_FakeChatbox(),
        source=_FakeSource(),
        mute=None,
    )

    phrases: list[str] = []
    bus.subscribe(PhraseRecognized, lambda e: phrases.append(e.text))

    stack.pipeline.start()
    stack.pipeline.set_captioning(True)
    try:
        audio = np.asarray(audio, dtype=np.float32)
        for i in range(0, len(audio) - FRAME, FRAME):
            stack.pipeline._on_frame(audio[i : i + FRAME].copy())
            time.sleep(FRAME / SAMPLE_RATE)
        time.sleep(drain_s)
    finally:
        stack.pipeline.stop()
    return phrases
