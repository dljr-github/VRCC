"""Static registry of the faster-whisper STT models VRCC offers.

Keys are exact faster-whisper model ids (``SttConfig.model`` holds one). ``tier``
is a coarse speed/accuracy class; ``english_only`` marks distil models that must
not be offered for non-English source languages.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WhisperSpec:
    id: str
    label: str
    size_mb: int
    tier: str            # "fast" | "balanced" | "accurate"
    english_only: bool


WHISPER_MODELS: dict[str, WhisperSpec] = {
    spec.id: spec
    for spec in (
        WhisperSpec("tiny", "Tiny", 75, "fast", False),
        WhisperSpec("base", "Base", 145, "fast", False),
        WhisperSpec("small", "Small", 484, "balanced", False),
        WhisperSpec("medium", "Medium", 1530, "balanced", False),
        WhisperSpec("large-v3", "Large v3", 3090, "accurate", False),
        WhisperSpec("large-v3-turbo", "Large v3 Turbo", 1620, "accurate", False),
        WhisperSpec(
            "distil-large-v3.5", "Distil-Large v3.5 (English)", 1510, "accurate", True
        ),
        WhisperSpec(
            "distil-small.en", "Distil-Small (English)", 332, "fast", True
        ),
    )
}
