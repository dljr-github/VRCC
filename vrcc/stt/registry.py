"""Static registry of the speech-to-text models VRCC offers.

Keys are exact model ids (``SttConfig.model`` holds one). The dict keeps its
historical ``WHISPER_MODELS`` name but now covers two backends: faster-whisper
models (``backend="whisper"``, ids are faster-whisper model ids) and NVIDIA
NeMo ONNX exports run via the onnx-asr package (``backend="onnx_asr"``,
downloaded from ``repo``, model architecture in ``asr_type``). ``tier`` is a
coarse speed/accuracy class; ``english_only`` marks distil models that must
not be offered for non-English source languages; ``languages`` (Whisper
language codes) restricts a model to a language subset (``None`` = no
restriction) and drives the Settings greying; ``auto_language`` is False for
models that cannot detect the spoken language themselves (they transcribe as
English unless told otherwise, so "auto" greys them out).
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
    # Whether the model detects the spoken language by itself ("auto" source).
    auto_language: bool = True
    backend: str = "whisper"       # "whisper" | "onnx_asr"
    repo: str | None = None        # HF repo for the onnx_asr backend
    quantization: str | None = None  # onnx-asr quantization suffix ("int8")
    asr_type: str | None = None    # onnx-asr model type ("nemo-conformer-*")


# The 25 European languages Parakeet TDT 0.6B v3 supports (Whisper codes).
_EUROPEAN_25_LANGUAGES = (
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
            True, languages=("en",), auto_language=False,
        ),
        WhisperSpec(
            "distil-small.en", "Distil-Small (English)", 332, "fast",
            True, languages=("en",), auto_language=False,
        ),
        WhisperSpec(
            "parakeet-tdt-0.6b-v3", "Parakeet v3 (European languages)", 690,
            "accurate", False,
            languages=_EUROPEAN_25_LANGUAGES,
            backend="onnx_asr",
            repo="istupakov/parakeet-tdt-0.6b-v3-onnx",
            quantization="int8",
            asr_type="nemo-conformer-tdt",
        ),
    )
}
