"""Friendly display names and blurbs for model ids shown in the GUI.

Config and the registries use terse ids ("nllb-600M-int8"); this maps the known
whisper/MT ids to short plain-language labels plus a one-line blurb (speed lead-in,
size, license/English-only caveat). Unknown ids fall back to the raw id (or empty
blurb) rather than raising, so the dropdown never breaks.
"""

from __future__ import annotations

from vrcc.i18n import tr, tr_noop
from vrcc.stt.registry import WHISPER_MODELS
from vrcc.translate.registry import MT_MODELS

_MT_DISPLAY_NAMES: dict[str, str] = {
    "nllb-600M-int8": tr_noop("NLLB 600M — balanced"),
    "nllb-1.3B-int8": tr_noop("NLLB 1.3B — higher quality"),
    "nllb-3.3B-int8": tr_noop("NLLB 3.3B — best quality (large)"),
    "m2m100-418M-int8": tr_noop("M2M100 418M — small"),
    "m2m100-1.2B-int8": tr_noop("M2M100 1.2B"),
    "madlad400-3b": tr_noop("MADLAD-400 3B"),
}

# whisper_display_name() returns registry labels via tr(); the registry itself is
# Qt- and i18n-free, so these markers exist purely so the catalog extractor sees
# every WHISPER_MODELS label. Keep in sync with vrcc/stt/registry.py.
_WHISPER_LABEL_MARKERS = (
    tr_noop("Tiny"),
    tr_noop("Base"),
    tr_noop("Small"),
    tr_noop("Medium"),
    tr_noop("Large v3"),
    tr_noop("Large v3 Turbo"),
    tr_noop("Distil-Large v3.5 (English)"),
    tr_noop("Distil-Small (English)"),
    tr_noop("Parakeet v3 (European languages)"),
)


def mt_display_name(model_id: str) -> str:
    """Friendly label for an MT model id; falls back to the id itself if unknown."""
    name = _MT_DISPLAY_NAMES.get(model_id)
    return tr(name) if name is not None else model_id


def whisper_display_name(model_id: str) -> str:
    """Friendly label for a whisper model id; falls back to the id itself if unknown."""
    spec = WHISPER_MODELS.get(model_id)
    return tr(spec.label) if spec is not None else model_id


def fmt_size(size_mb: int) -> str:
    """Render a size in MB as "~X.X GB" (>=1000 MB) or "X MB" otherwise."""
    if size_mb >= 1000:
        return tr("~{gb:.1f} GB", gb=size_mb / 1000)
    return tr("{mb} MB", mb=size_mb)


_fmt_size = fmt_size  # old name kept as an alias for existing callers


# Short quality/speed lead-ins, distinct per model. Unlisted ids fall back to a
# generic, size-derived lead-in so the blurb never looks broken.
_WHISPER_LEAD_INS: dict[str, str] = {
    "tiny": tr_noop("Fastest — rough accuracy"),
    "base": tr_noop("Fast — basic accuracy"),
    "small": tr_noop("Good balance for most PCs"),
    "medium": tr_noop("More accurate, needs a decent PC"),
    "large-v3": tr_noop("Most accurate — big download"),
    "large-v3-turbo": tr_noop("Most accurate and fast"),
    "distil-large-v3.5": tr_noop("Near-most accurate, fast"),
    "distil-small.en": tr_noop("Fast, small download"),
    "parakeet-tdt-0.6b-v3": tr_noop("Very accurate and fast"),
}

_MT_LEAD_INS: dict[str, str] = {
    "nllb-600M-int8": tr_noop("Balanced"),
    "nllb-1.3B-int8": tr_noop("Higher quality"),
    "nllb-3.3B-int8": tr_noop("Best quality (large)"),
    "m2m100-418M-int8": tr_noop("Fastest, lower quality"),
    "m2m100-1.2B-int8": tr_noop("Balanced"),
    "madlad400-3b": tr_noop("Best quality (large)"),
}


def _generic_lead_in(size_mb: int) -> str:
    """Fallback lead-in for an unmapped model, derived from its size."""
    if size_mb < 300:
        return tr("Fastest, lower accuracy")
    if size_mb < 1000:
        return tr("Balanced")
    return tr("Best accuracy (large)")


def model_blurb(kind: str, model_id: str) -> str:
    """Short one-line descriptor for a model, e.g. "Best accuracy · ~1.6 GB".

    ``kind`` is ``"whisper"`` or ``"mt"``. Includes "· non-commercial use"
    for MT specs whose ``license`` contains "NC"; includes "· English only"
    for voice specs with ``english_only`` True and "· European languages only"
    for other language-restricted voice specs. Unknown ids return ``""``.
    """
    if kind == "whisper":
        spec = WHISPER_MODELS.get(model_id)
        if spec is None:
            return ""
        raw_lead_in = _WHISPER_LEAD_INS.get(model_id)
        lead_in = tr(raw_lead_in) if raw_lead_in else _generic_lead_in(spec.size_mb)
        parts = [lead_in, fmt_size(spec.size_mb)]
        blurb = " · ".join(parts)
        if spec.english_only:
            blurb += " · " + tr("English only")
        elif spec.languages is not None:
            # Today the only language-restricted non-English model is Parakeet
            # (25 European languages); revisit the wording if that changes.
            blurb += " · " + tr("European languages only")
        return blurb
    if kind == "mt":
        spec = MT_MODELS.get(model_id)
        if spec is None:
            return ""
        raw_lead_in = _MT_LEAD_INS.get(model_id)
        lead_in = tr(raw_lead_in) if raw_lead_in else _generic_lead_in(spec.size_mb)
        parts = [lead_in, fmt_size(spec.size_mb)]
        blurb = " · ".join(parts)
        if "NC" in spec.license:
            blurb += " · " + tr("non-commercial use")
        return blurb
    return ""
