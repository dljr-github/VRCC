"""Static registry of the speech-to-text models VRCC offers.

Keys are exact model ids (``SttConfig.model`` holds one). The dict keeps its
historical ``WHISPER_MODELS`` name but now covers three backends:
faster-whisper models (``backend="whisper"``, ids are faster-whisper model
ids), NVIDIA NeMo ONNX exports run via the onnx-asr package
(``backend="onnx_asr"``, downloaded from ``repo``, model architecture in
``asr_type``), and models driven directly on onnxruntime because onnx-asr has
no class for them (``backend="sensevoice"``). ``tier`` is a coarse
speed/accuracy class; ``english_only`` marks distil models that must not be
offered for non-English source languages; ``languages`` (Whisper language
codes) restricts a model to a language subset (``None`` = no restriction) and
drives the Settings greying; ``auto_language`` is False for models that cannot
detect the spoken language themselves (they transcribe as English unless told
otherwise, so "auto" greys them out); ``reports_language`` is False for models
that detect the language but cannot tell us which one they heard, which is a
separate defect because it mislabels the translator's source.
"""

from __future__ import annotations

from dataclasses import dataclass

# Backends that run on onnxruntime rather than CTranslate2. They share two
# behaviours the GUI and the device resolution key off: greedy decoders (so the
# Speed/Quality beam has nothing to widen) and int8 graphs that are fast enough
# on the CPU to be worth leaving VRChat the VRAM.
ONNX_BACKENDS = ("onnx_asr", "sensevoice")


@dataclass(frozen=True)
class WhisperSpec:
    id: str
    label: str
    size_mb: int
    tier: str            # "fast" | "balanced" | "accurate"
    english_only: bool
    # Supported source languages as Whisper codes; None = unrestricted.
    languages: tuple[str, ...] | None = None
    # Plain-language name for that set, for the Models/Settings blurb. English
    # source text (this module stays i18n-free; vrcc.gui.model_labels marks it
    # for the catalog extractor and translates it).
    language_note: str | None = None
    # Whether the model detects the spoken language by itself ("auto" source).
    auto_language: bool = True
    # Whether it reports the language it detected, so the translator can be
    # told the right source. False = detects but tags every result "en".
    reports_language: bool = True
    backend: str = "whisper"       # "whisper" | "onnx_asr" | "sensevoice"
    repo: str | None = None        # HF repo for the onnx_asr backend
    quantization: str | None = None  # quantization file suffix ("int8")
    asr_type: str | None = None    # onnx-asr model type ("nemo-conformer-*")
    # Archive to fetch and unpack instead of an HF snapshot, for models
    # published only as a release tarball. Members land flat in the model dir.
    archive_url: str | None = None

    @property
    def runs_on_onnxruntime(self) -> bool:
        return self.backend in ONNX_BACKENDS


# The 25 European languages Parakeet TDT 0.6B v3 supports (Whisper codes).
_EUROPEAN_25_LANGUAGES = (
    "bg", "hr", "cs", "da", "nl", "en", "et", "fi", "fr", "de", "el", "hu",
    "it", "lv", "lt", "mt", "pl", "pt", "ro", "ru", "sk", "sl", "es", "sv",
    "uk",
)

# SenseVoice-Small's languages as Whisper codes. It also transcribes Cantonese,
# which has no VRCC display language of its own. Note "zh" here means Simplified
# script: the model has no Traditional mode and no initial_prompt lever like the
# faster-whisper path uses, so a Traditional source gets Simplified glyphs.
_SENSEVOICE_LANGUAGES = ("en", "ja", "ko", "zh")


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
            language_note="European languages only",
            # Detects within its set but tags every result "en".
            reports_language=False,
            backend="onnx_asr",
            repo="istupakov/parakeet-tdt-0.6b-v3-onnx",
            quantization="int8",
            asr_type="nemo-conformer-tdt",
        ),
        WhisperSpec(
            "sense-voice-small", "SenseVoice (Chinese/Japanese/Korean/English)",
            240, "accurate", False,
            languages=_SENSEVOICE_LANGUAGES,
            language_note="Chinese, Japanese, Korean and English only",
            backend="sensevoice",
            repo="FunAudioLLM/SenseVoiceSmall",
            quantization="int8",
            # Published only as a sherpa-onnx release tarball; the upstream HF
            # repo above is the origin the export names, not a fetchable
            # onnxruntime layout.
            archive_url=(
                "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models"
                "/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17.tar.bz2"
            ),
        ),
    )
}
