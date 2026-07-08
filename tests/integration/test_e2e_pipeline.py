"""End-to-end pipeline smoke as a pytest: WAV -> VAD -> STT -> MT -> chatbox,
sharing ``scripts/smoke_e2e.py``. Integration-marked and skipped unless
``VRCC_E2E_WAV`` is set (downloads tiny whisper + nllb-600M on first run).
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest

_WAV = os.environ.get("VRCC_E2E_WAV", "")

_SMOKE_PATH = Path(__file__).resolve().parents[2] / "scripts" / "smoke_e2e.py"


def _load_smoke_module():
    spec = importlib.util.spec_from_file_location("vrcc_smoke_e2e", _SMOKE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.integration
@pytest.mark.skipif(
    not _WAV, reason="set VRCC_E2E_WAV to a WAV file of English speech"
)
def test_e2e_wav_produces_translated_chatbox_message():
    wav = Path(_WAV)
    assert wav.is_file(), f"VRCC_E2E_WAV points at a missing file: {wav}"

    smoke = _load_smoke_module()
    result = smoke.run_smoke(
        wav,
        target="Japanese",
        stt_model="tiny",
        mt_model="nllb-600M-int8",
        source="English",
        device="cpu",  # deterministic everywhere, no GPU required
    )

    assert result.finals >= 1, "segmenter produced no finalized utterance"
    assert result.phrases, "no phrase survived STT's quality gates"

    for phrase in result.phrases:
        assert phrase.text.strip(), "recognized text must be non-empty"
        assert phrase.translations, "translation list must be non-empty"
        for name, text in phrase.translations:
            assert name == "Japanese"
            assert text.strip(), "translated text must be non-empty"
        # The chatbox message is original + translation joined by the
        # configured separator; both halves must have made it in.
        assert phrase.text in phrase.chatbox_message
        assert phrase.translations[0][1] in phrase.chatbox_message
