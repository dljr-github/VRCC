"""Tests for :meth:`OnnxAsrEngine.transcribe`: split out of test_stt_onnx_asr.py
to stay under the 500-line cap. Shares the fake onnx-asr factory with that
file rather than duplicating it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from vrcc.core.bus import EventBus
from vrcc.core.config import SttConfig
from vrcc.stt.engine import SttResult
from vrcc.stt.onnx_asr import OnnxAsrEngine
from vrcc.stt.registry import WHISPER_MODELS

from .test_stt_onnx_asr import _cfg, _RecordingFactory

PARAKEET_ID = "parakeet-tdt-0.6b-v3"
PARAKEET = WHISPER_MODELS[PARAKEET_ID]


@pytest.fixture()
def model_dir(tmp_path: Path) -> Path:
    d = tmp_path / "models" / "whisper" / PARAKEET_ID
    d.mkdir(parents=True)
    return d


def _loaded_engine(model_dir, spec=PARAKEET, text="hello there", logprobs=None, **cfg_over):
    bus = EventBus()
    factory = _RecordingFactory(text, logprobs=logprobs)
    cfg_over.setdefault("model", spec.id)
    eng = OnnxAsrEngine(_cfg(**cfg_over), spec, model_dir, bus, model_factory=factory)
    eng.load()
    return eng, factory


def test_transcribe_before_load_raises(model_dir):
    eng = OnnxAsrEngine(
        _cfg(), PARAKEET, model_dir, EventBus(), model_factory=_RecordingFactory()
    )
    with pytest.raises(RuntimeError, match="load"):
        eng.transcribe(np.zeros(160, dtype=np.float32))


def test_transcribe_returns_result_with_mean_logprob_and_neutral_no_speech(model_dir):
    eng, factory = _loaded_engine(
        model_dir, text="  Bonjour tout le monde  ", logprobs=[-0.02, -0.04, -0.03],
    )
    result = eng.transcribe(np.zeros(1600, dtype=np.float32))

    assert isinstance(result, SttResult)
    assert result.text == "Bonjour tout le monde"
    assert result.avg_logprob == pytest.approx(-0.03)
    assert result.no_speech_prob == 0.0  # no no-speech signal from this decoder
    call = factory.built[0].calls[0]
    assert call.sample_rate == 16000
    assert call.samples.dtype == np.float32


def test_transcribe_confident_result_passes_the_parakeet_gate(model_dir):
    cfg = SttConfig()
    eng, _ = _loaded_engine(model_dir, logprobs=[-0.02, -0.03])
    result = eng.transcribe(np.zeros(1600, dtype=np.float32))

    assert result is not None
    assert result.avg_logprob >= cfg.parakeet_avg_logprob_gate


def test_transcribe_low_confidence_result_is_gated(model_dir, caplog):
    # Mean well below parakeet_avg_logprob_gate: a babble-degraded decode.
    eng, _ = _loaded_engine(model_dir, logprobs=[-0.9, -1.1, -0.95])

    with caplog.at_level("DEBUG", logger="vrcc.stt.onnx_asr"):
        assert eng.transcribe(np.zeros(1600, dtype=np.float32)) is None

    assert "avg_logprob" in caplog.text


def test_transcribe_missing_logprobs_falls_back_to_neutral_not_gated(model_dir):
    # A future onnx_asr contract change (empty/None logprobs) must not gate
    # blind: neutral avg_logprob, result still returned.
    eng, _ = _loaded_engine(model_dir, logprobs=[])
    result = eng.transcribe(np.zeros(1600, dtype=np.float32))

    assert result is not None
    assert result.avg_logprob == 0.0


def test_transcribe_empty_text_returns_none(model_dir):
    eng, _ = _loaded_engine(model_dir, text="   ")
    assert eng.transcribe(np.zeros(1600, dtype=np.float32)) is None


def test_transcribe_language_echoes_configured_source(model_dir):
    eng, _ = _loaded_engine(model_dir, source_language="French")
    assert eng.transcribe(np.zeros(160, dtype=np.float32)).language == "fr"


def test_transcribe_language_auto_falls_back_to_english(model_dir):
    eng, _ = _loaded_engine(model_dir, source_language="auto")
    assert eng.transcribe(np.zeros(160, dtype=np.float32)).language == "en"


def test_transcribe_detect_language_reports_none(model_dir):
    """Nobody has evidence for what these decoders actually heard; "en" was a
    fabricated tag that fed the translator a source it never detected."""
    eng, _ = _loaded_engine(model_dir, source_language="French")
    result = eng.transcribe(np.zeros(160, dtype=np.float32), detect_language=True)
    assert result.language is None


def test_transcribe_detect_language_overrides_the_auto_fallback(model_dir):
    """The two branches must not collapse into one: detect_language=True is
    the heard stream asking about someone else's speech, and must return None
    even when source_language is also "auto"."""
    eng, _ = _loaded_engine(model_dir, source_language="auto")
    result = eng.transcribe(np.zeros(160, dtype=np.float32), detect_language=True)
    assert result.language is None


def test_transducer_passes_no_language_option(model_dir):
    eng, factory = _loaded_engine(model_dir, source_language="French")
    eng.transcribe(np.zeros(160, dtype=np.float32))
    assert factory.built[0].calls[0].kwargs == {}


def test_warm_up_transcribes_half_second_of_silence(model_dir):
    eng, factory = _loaded_engine(model_dir)
    eng.warm_up()
    call = factory.built[0].calls[0]
    assert len(call.samples) == 8000


def test_unload_drops_model_and_transcribe_raises(model_dir):
    eng, _ = _loaded_engine(model_dir)
    eng.unload()
    with pytest.raises(RuntimeError):
        eng.transcribe(np.zeros(160, dtype=np.float32))
