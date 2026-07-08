"""Friendly display names and blurbs for model ids shown in the GUI.

Config and the registries use terse ids ("nllb-600M-int8"); this maps the known
whisper/MT ids to short plain-language labels plus a one-line blurb (speed lead-in,
size, license/English-only caveat). Unknown ids fall back to the raw id (or empty
blurb) rather than raising, so the dropdown never breaks.
"""

from __future__ import annotations

from vrcc.stt.registry import WHISPER_MODELS
from vrcc.translate.registry import MT_MODELS

_MT_DISPLAY_NAMES: dict[str, str] = {
    "nllb-600M-int8": "NLLB 600M — balanced",
    "nllb-1.3B-int8": "NLLB 1.3B — higher quality",
    "nllb-3.3B-int8": "NLLB 3.3B — best quality (large)",
    "m2m100-418M-int8": "M2M100 418M — small",
    "m2m100-1.2B-int8": "M2M100 1.2B",
    "madlad400-3b": "MADLAD-400 3B",
}


def mt_display_name(model_id: str) -> str:
    """Friendly label for an MT model id; falls back to the id itself if unknown."""
    return _MT_DISPLAY_NAMES.get(model_id, model_id)


def whisper_display_name(model_id: str) -> str:
    """Friendly label for a whisper model id; falls back to the id itself if unknown."""
    spec = WHISPER_MODELS.get(model_id)
    return spec.label if spec is not None else model_id


def fmt_size(size_mb: int) -> str:
    """Render a size in MB as "~X.X GB" (>=1000 MB) or "X MB" otherwise."""
    if size_mb >= 1000:
        return f"~{size_mb / 1000:.1f} GB"
    return f"{size_mb} MB"


_fmt_size = fmt_size  # old name kept as an alias for existing callers


# Short quality/speed lead-ins, distinct per model. Unlisted ids fall back to a
# generic, size-derived lead-in so the blurb never looks broken.
_WHISPER_LEAD_INS: dict[str, str] = {
    "tiny": "Fastest — rough accuracy",
    "base": "Fast — basic accuracy",
    "small": "Good balance for most PCs",
    "medium": "More accurate, needs a decent PC",
    "large-v3": "Most accurate — big download",
    "large-v3-turbo": "Most accurate and fast",
    "distil-large-v3.5": "Near-most accurate, fast",
    "distil-small.en": "Fast, small download",
}

_MT_LEAD_INS: dict[str, str] = {
    "nllb-600M-int8": "Balanced",
    "nllb-1.3B-int8": "Higher quality",
    "nllb-3.3B-int8": "Best quality (large)",
    "m2m100-418M-int8": "Fastest, lower quality",
    "m2m100-1.2B-int8": "Balanced",
    "madlad400-3b": "Best quality (large)",
}


def _generic_lead_in(size_mb: int) -> str:
    """Fallback lead-in for an unmapped model, derived from its size."""
    if size_mb < 300:
        return "Fastest, lower accuracy"
    if size_mb < 1000:
        return "Balanced"
    return "Best accuracy (large)"


def model_blurb(kind: str, model_id: str) -> str:
    """Short one-line descriptor for a model, e.g. "Best accuracy · ~1.6 GB".

    ``kind`` is ``"whisper"`` or ``"mt"``. Includes "· non-commercial use"
    for MT specs whose ``license`` contains "NC"; includes "· English only"
    for whisper specs with ``english_only`` True. Unknown ids return ``""``.
    """
    if kind == "whisper":
        spec = WHISPER_MODELS.get(model_id)
        if spec is None:
            return ""
        lead_in = _WHISPER_LEAD_INS.get(model_id) or _generic_lead_in(spec.size_mb)
        parts = [lead_in, fmt_size(spec.size_mb)]
        blurb = " · ".join(parts)
        if spec.english_only:
            blurb += " · English only"
        return blurb
    if kind == "mt":
        spec = MT_MODELS.get(model_id)
        if spec is None:
            return ""
        lead_in = _MT_LEAD_INS.get(model_id) or _generic_lead_in(spec.size_mb)
        parts = [lead_in, fmt_size(spec.size_mb)]
        blurb = " · ".join(parts)
        if "NC" in spec.license:
            blurb += " · non-commercial use"
        return blurb
    return ""
