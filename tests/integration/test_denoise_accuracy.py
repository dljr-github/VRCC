"""Phase 0 go/no-go gate: the shipped streaming ``Denoiser`` at the gentle
0.5 dry/wet mix must not regress noisy single-word recognition versus raw
audio, against the real cached Whisper model. Feeds each fixture through
``Denoiser.process`` in 512-sample frames, identical to how ``MicSource``
feeds capture frames through the real pipeline. Needs a whisper model
already downloaded on this machine (see ``_harness.find_cached_whisper``);
skips otherwise.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from vrcc.audio.denoise import Denoiser

from ._harness import find_cached_whisper, load_fixture, norm_words

_DIR = Path(__file__).resolve().parent / "audio" / "noisy"
_FRAME = 512

_WHISPER = find_cached_whisper()

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(_WHISPER is None, reason="no cached whisper model"),
]


def _denoise(sig: np.ndarray, strength: float) -> np.ndarray:
    denoiser = Denoiser()
    denoiser.configure(enabled=True, strength=strength)
    frames = [
        denoiser.process(sig[i : i + _FRAME])
        for i in range(0, len(sig) - _FRAME + 1, _FRAME)
    ]
    return np.concatenate(frames) if frames else sig[:0]


def test_gentle_denoise_beats_raw_on_noisy_words(stt):
    manifest = json.loads((_DIR / "manifest.json").read_text(encoding="utf-8"))

    signals = {m["file"]: load_fixture(f"noisy/{m['file']}") for m in manifest}
    sample = next(iter(signals.values()))
    mixed = _denoise(sample, 0.5)
    assert not np.allclose(mixed, sample[: len(mixed)]), (
        "Denoiser did not change the audio at strength 0.5; check the wiring "
        "before trusting a raw/mix comparison"
    )

    raw_ok = mix_ok = 0
    for m in manifest:
        sig = signals[m["file"]]
        expected = m["text"]

        raw = stt.transcribe(sig)
        raw_ok += int(raw is not None and " ".join(norm_words(raw.text)) == expected)

        mix = stt.transcribe(_denoise(sig, 0.5))
        mix_ok += int(mix is not None and " ".join(norm_words(mix.text)) == expected)

    assert mix_ok >= raw_ok, (
        f"gentle denoise regressed noisy single-word recognition: "
        f"raw {raw_ok}/{len(manifest)} vs mix {mix_ok}/{len(manifest)}"
    )
