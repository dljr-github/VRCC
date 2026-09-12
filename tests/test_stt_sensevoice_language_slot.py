"""Tests for SenseVoice's metadata-driven language slot: split out of
test_stt_sensevoice.py to stay under the 500-line cap. Shares the fake
onnxruntime session/factory with that file rather than duplicating them.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tests.test_stt_sensevoice import (
    SENSEVOICE_ID,
    _META,
    _RecordingFactory,
    _VOCAB,
    _engine,
    _speech,
)


@pytest.fixture()
def model_dir(tmp_path: Path) -> Path:
    d = tmp_path / "models" / "whisper" / SENSEVOICE_ID
    d.mkdir(parents=True)
    (d / "model.int8.onnx").write_bytes(b"")
    (d / "tokens.txt").write_text(
        "\n".join(f"{piece} {i}" for i, piece in enumerate(_VOCAB)) + "\n",
        encoding="utf-8",
    )
    return d


@pytest.mark.parametrize(
    ("source", "expected"),
    [("auto", 0), ("English", 4), ("Japanese", 11), ("Korean", 12),
     ("Chinese Simplified", 3)],
)
def test_language_slot_comes_from_the_models_own_metadata(model_dir, source, expected):
    factory = _RecordingFactory()
    eng = _engine(model_dir, factory, source_language=source)
    eng.load()
    eng.transcribe(_speech())

    assert factory.built[0].runs[0]["language"][0] == expected


def test_language_outside_the_models_set_uses_auto(model_dir):
    # French has no lang_* slot in this export; auto beats guessing.
    factory = _RecordingFactory()
    eng = _engine(model_dir, factory, source_language="French")
    eng.load()
    eng.transcribe(_speech())

    assert factory.built[0].runs[0]["language"][0] == 0


def test_language_slot_follows_a_live_source_language_change(model_dir):
    """The spoken-language combo writes straight to the live config without
    rebuilding the engine, so the slot must be read per transcribe."""
    factory = _RecordingFactory()
    eng = _engine(model_dir, factory, source_language="Japanese")
    eng.load()
    eng.transcribe(_speech())

    eng._cfg.source_language = "Korean"
    eng.transcribe(_speech())

    slots = [run["language"][0] for run in factory.built[0].runs]
    assert slots == [11, 12]


def test_cantonese_slot_does_not_displace_mandarin_for_chinese(model_dir):
    """lang_zh and lang_yue both map to the Whisper code "zh"; a VRCC
    "Chinese" source must pin Mandarin, not Cantonese."""
    factory = _RecordingFactory()
    eng = _engine(model_dir, factory, source_language="Chinese Traditional")
    eng.load()
    eng.transcribe(_speech())

    assert factory.built[0].runs[0]["language"][0] == 3  # lang_zh, not lang_yue (7)


def test_normalize_samples_metadata_drives_the_amplitude_scale(model_dir):
    """normalize_samples=1 means the export wants [-1, 1] audio and =0 means
    int16 scale. Getting it wrong does not raise, it silently shifts every
    filterbank energy, so the flag has to actually reach the front-end."""
    features = {}
    for flag in ("0", "1"):
        factory = _RecordingFactory(meta=dict(_META, normalize_samples=flag))
        engine = _engine(model_dir, factory)
        engine.load()
        engine.transcribe(_speech())
        features[flag] = factory.built[0].runs[0]["x"]

    assert not np.allclose(features["0"], features["1"])
