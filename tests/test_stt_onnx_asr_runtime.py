"""End-to-end CPU tests of the onnx-asr STT path against the REAL onnx-asr +
onnxruntime stack (no network, no NVIDIA weights): synthetic ONNX exports with
the exact Parakeet-TDT / Canary-AED contracts (tests/onnx_asr_fakes.py) are
laid out exactly as DownloadManager writes them, then driven through
create_stt_engine -> OnnxAsrEngine -> onnx-asr's real mel preprocessing and
decode loops. Guards the integration against onnx-asr contract drift, which
the fake-factory unit tests in test_stt_onnx_asr.py can't see.

Skipped when the ``onnx`` graph-building package (dev extra) is absent.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("onnx")

from vrcc.core.bus import EventBus
from vrcc.core.config import SttConfig
from vrcc.core.events import EngineStateChanged
from vrcc.download.manager import DownloadManager
from vrcc.stt import create_stt_engine
from vrcc.stt.onnx_asr import OnnxAsrEngine

from tests.onnx_asr_fakes import build_fake_aed, build_fake_tdt

PARAKEET_ID = "parakeet-tdt-0.6b-v3"
CANARY_ID = "canary-1b-v2"

# 2s of 440 Hz sine at 16 kHz: content is irrelevant (synthetic weights), but
# it flows through onnx-asr's real nemo128 mel preprocessor.
AUDIO = (0.1 * np.sin(2 * np.pi * 440 * np.arange(32000) / 16000)).astype(np.float32)


@pytest.fixture(scope="module")
def models_dir(tmp_path_factory):
    base = tmp_path_factory.mktemp("models")
    tdt = base / "whisper" / PARAKEET_ID
    aed = base / "whisper" / CANARY_ID
    tdt.mkdir(parents=True)
    aed.mkdir(parents=True)
    build_fake_tdt(tdt)
    build_fake_aed(aed)
    return base


def _engine(models_dir, model_id, source_language):
    bus = EventBus()
    events: list[EngineStateChanged] = []
    bus.subscribe(EngineStateChanged, events.append)
    dm = DownloadManager(models_dir, bus)
    cfg = SttConfig(
        model=model_id, device="cpu", compute_type="int8",
        source_language=source_language,
    )
    engine = create_stt_engine(cfg, dm.whisper_model_dir(model_id), bus)
    return engine, dm, events


def test_fake_exports_satisfy_the_download_manager_presence_check(models_dir):
    dm = DownloadManager(models_dir, EventBus())
    assert dm.is_whisper_downloaded(PARAKEET_ID) is True
    assert dm.is_whisper_downloaded(CANARY_ID) is True


def test_parakeet_full_stack_transcribes_on_cpu(models_dir):
    engine, _, events = _engine(models_dir, PARAKEET_ID, "English")
    assert isinstance(engine, OnnxAsrEngine)

    engine.load()
    assert [e.state for e in events] == ["loading", "ready"]
    assert events[-1].detail == "cpu:int8"

    engine.warm_up()
    result = engine.transcribe(AUDIO)
    assert result is not None
    assert result.text == "hello world"  # the TDT decode loop ran for real
    assert result.language == "en"
    assert (result.avg_logprob, result.no_speech_prob) == (0.0, 0.0)
    engine.unload()


def test_parakeet_auto_source_transcribes_and_falls_back_to_english(models_dir):
    engine, _, _ = _engine(models_dir, PARAKEET_ID, "auto")
    engine.load()
    result = engine.transcribe(AUDIO)
    assert result.text == "hello world"
    assert result.language == "en"


# The fake AED decoder's first emitted word reveals which language token
# onnx-asr wrote into prompt slot 4, so these assert VRCC's Canary language
# forcing through the real decode loop, not through a mock.

def test_canary_prompt_defaults_to_english(models_dir):
    engine, _, _ = _engine(models_dir, CANARY_ID, "English")
    engine.load()
    assert engine.transcribe(AUDIO).text == "english"


def test_canary_forces_the_configured_source_language(models_dir):
    engine, _, _ = _engine(models_dir, CANARY_ID, "French")
    engine.load()
    result = engine.transcribe(AUDIO)
    assert result.text == "bonjour"  # <|fr|> reached the decoder prompt
    assert result.language == "fr"


def test_canary_auto_source_leaves_the_english_default(models_dir):
    engine, _, _ = _engine(models_dir, CANARY_ID, "auto")
    engine.load()
    assert engine.transcribe(AUDIO).text == "english"


def test_canary_unsupported_source_is_not_forced(models_dir):
    engine, _, _ = _engine(models_dir, CANARY_ID, "Japanese")
    engine.load()
    assert engine.transcribe(AUDIO).text == "english"
