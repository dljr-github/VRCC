"""Static registry of the speech-to-text models VRCC offers.

Keys are exact model ids (``SttConfig.model`` holds one). The dict keeps its
historical ``WHISPER_MODELS`` name but now covers two backends: faster-whisper
models (``backend="whisper"``, ids are faster-whisper model ids) and NVIDIA
Parakeet ONNX exports run via onnx-asr (``backend="parakeet"``, downloaded
from ``repo``). ``tier`` is a coarse speed/accuracy class; ``english_only``
marks distil models that must not be offered for non-English source languages;
``languages`` (Whisper language codes) restricts a model to a language subset
(``None`` = no restriction) and drives the Settings greying for both the
distil and Parakeet models.
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
    # Supported source languages as Whisper codes; None = unrestricted.
    languages: tuple[str, ...] | None = None
    backend: str = "whisper"       # "whisper" | "parakeet"
    repo: str | None = None        # HF repo for non-whisper backends
    quantization: str | None = None  # onnx-asr quantization suffix ("int8")


# The 25 languages Parakeet TDT 0.6B v3 supports (Whisper codes).
_PARAKEET_V3_LANGUAGES = (
    "bg", "hr", "cs", "da", "nl", "en", "et", "fi", "fr", "de", "el", "hu",
    "it", "lv", "lt", "mt", "pl", "pt", "ro", "ru", "sk", "sl", "es", "sv",
    "uk",
)


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
            "distil-large-v3.5", "Distil-Large v3.5 (English)", 1510, "accurate",
            True, languages=("en",),
        ),
        WhisperSpec(
            "distil-small.en", "Distil-Small (English)", 332, "fast",
            True, languages=("en",),
        ),
        WhisperSpec(
            "parakeet-tdt-0.6b-v3", "Parakeet v3 (European languages)", 690,
            "accurate", False,
            languages=_PARAKEET_V3_LANGUAGES,
            backend="parakeet",
            repo="istupakov/parakeet-tdt-0.6b-v3-onnx",
            quantization="int8",
        ),
    )
}
