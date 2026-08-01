"""Reference-transcript regression for the real SenseVoice model.

This is what anchors the numpy front-end in :mod:`vrcc.stt.fbank` to *correct*
rather than merely stable: it runs VRCC's own engine over the five demo clips
that ship inside the sherpa-onnx release asset and asserts the transcripts
sherpa-onnx publishes for them. If the front-end drifts (wrong mel scale,
wrong LFR padding, missed int16 scaling) the text degrades here even though
the unit tests still pass, because the model swallows bad features silently
instead of raising.

Needs the model on disk. It is never downloaded from here: run

    python -c "from vrcc.core.bus import EventBus; \\
               from vrcc.core.config import default_paths; \\
               from vrcc.download.manager import DownloadManager; \\
               DownloadManager(default_paths(portable=False).models_dir, \\
                               EventBus()).ensure_whisper('sense-voice-small')"

for the model, and the clips come from the same release asset (see
``_REFERENCE_ARCHIVE``); point ``VRCC_SENSEVOICE_WAVS`` at the extracted
``test_wavs`` directory. Both missing -> skip.
"""

from __future__ import annotations

import os
import wave
from pathlib import Path

import numpy as np
import pytest

from vrcc.core.bus import EventBus
from vrcc.core.config import SttConfig, default_paths
from vrcc.download.manager import DownloadManager
from vrcc.stt.registry import WHISPER_MODELS
from vrcc.stt.sensevoice import SenseVoiceEngine

MODEL_ID = "sense-voice-small"

_REFERENCE_ARCHIVE = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models"
    "/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17.tar.bz2"
)

# sherpa-onnx's own published decodes of the clips in that archive, with the
# <|lang|><|emotion|><|event|><|itn|> prefix stripped (VRCC strips it too).
_REFERENCE = {
    "en": (
        "en",
        "The tribal chieftain called for the boy and presented him with 50 "
        "pieces of code.",
    ),
    "zh": ("zh", "开饭时间早上9点至下午5点。"),
    "ja": (
        "ja",
        "うちの中学は弁当制で持っていきない場合は50円の学校販売のパンを買う。",
    ),
    "ko": ("ko", "조금만 생각을 하면서 살면 훨씬 편할 거야."),
    # Cantonese has no VRCC display language, so it is reported as Chinese.
    "yue": ("zh", "呢几个字都表达唔到我想讲嘅意思。"),
}


def _model_dir() -> Path | None:
    manager = DownloadManager(default_paths(portable=False).models_dir, EventBus())
    if not manager.is_whisper_downloaded(MODEL_ID):
        return None
    return manager.whisper_model_dir(MODEL_ID)


def _wav_dir() -> Path | None:
    override = os.environ.get("VRCC_SENSEVOICE_WAVS")
    if override and Path(override).is_dir():
        return Path(override)
    model_dir = _model_dir()
    if model_dir is not None and (model_dir / "test_wavs").is_dir():
        return model_dir / "test_wavs"
    return None


_MODEL_DIR = _model_dir()
_WAV_DIR = _wav_dir()

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        _MODEL_DIR is None or _WAV_DIR is None,
        reason=(
            f"needs {MODEL_ID} downloaded and its test_wavs (from "
            f"{_REFERENCE_ARCHIVE}); see this module's docstring"
        ),
    ),
]


def _load_wav(path: Path) -> np.ndarray:
    """Mono float32 in [-1, 1], the shape the engine contract expects. The
    demo clips are already 16 kHz mono 16-bit, which is asserted rather than
    converted: a clip that is not would silently measure the wrong thing."""
    with wave.open(str(path)) as handle:
        assert handle.getsampwidth() == 2, f"{path.name} is not 16-bit PCM"
        assert handle.getnchannels() == 1, f"{path.name} is not mono"
        assert handle.getframerate() == 16000, f"{path.name} is not 16 kHz"
        frames = np.frombuffer(handle.readframes(handle.getnframes()), dtype=np.int16)
    return np.ascontiguousarray(frames.astype(np.float32) / 32768.0)


@pytest.fixture(scope="module")
def engine() -> SenseVoiceEngine:
    cfg = SttConfig(model=MODEL_ID, device="cpu", source_language="auto")
    eng = SenseVoiceEngine(cfg, WHISPER_MODELS[MODEL_ID], _MODEL_DIR, EventBus())
    eng.load()
    return eng


@pytest.mark.parametrize("clip", sorted(_REFERENCE))
def test_reference_clip_transcribes_to_the_published_text(engine, clip):
    expected_language, expected_text = _REFERENCE[clip]

    result = engine.transcribe(_load_wav(_WAV_DIR / f"{clip}.wav"))

    assert result is not None
    assert result.text == expected_text
    assert result.language == expected_language


def test_decodes_are_deterministic(engine):
    """int8 ONNX graphs have been seen to decode differently run to run under
    multi-threaded onnxruntime, which shows up as garbled CJK rather than an
    error. Ten decodes of one clip must agree."""
    audio = _load_wav(_WAV_DIR / "ja.wav")
    texts = {engine.transcribe(audio).text for _ in range(10)}

    assert len(texts) == 1, f"non-deterministic decode: {texts}"


def test_tags_never_leak_into_the_caption(engine):
    for clip in sorted(_REFERENCE):
        text = engine.transcribe(_load_wav(_WAV_DIR / f"{clip}.wav")).text
        assert "<|" not in text and "|>" not in text
