"""Friendly display names and blurbs for model ids shown in the GUI.

Config and the registries use terse ids ("nllb-600M-int8"); this maps the known
whisper/MT ids to short plain-language labels plus a one-line blurb (speed lead-in,
size, license/English-only caveat). Unknown ids fall back to the raw id (or empty
blurb) rather than raising, so the dropdown never breaks.
"""

from __future__ import annotations

from vrcc.core.languages import LANGUAGES
from vrcc.i18n import tr, tr_noop
from vrcc.stt.registry import WHISPER_MODELS
from vrcc.translate.registry import MT_MODELS, collapses_onto

# Identity only. The quality ladder lives in _MT_LEAD_INS; carrying it here
# too meant two hand-synced copies, and they drifted the moment the
# preference order moved: the badged row read "small" while the only
# ladder in the name column pointed at the family that had been demoted.
_MT_DISPLAY_NAMES: dict[str, str] = {
    "nllb-600M-int8": tr_noop("NLLB 600M"),
    "nllb-1.3B-int8": tr_noop("NLLB 1.3B"),
    "nllb-3.3B-int8": tr_noop("NLLB 3.3B"),
    "m2m100-418M-int8": tr_noop("M2M100 418M"),
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
    tr_noop("SenseVoice (Chinese/Japanese/Korean/English)"),
)

# Same deal for WhisperSpec.language_note: the registry holds the English
# source text, these markers put it in the catalog. Keep in sync with
# vrcc/stt/registry.py.
_LANGUAGE_NOTE_MARKERS = (
    tr_noop("European languages only"),
    tr_noop("Chinese, Japanese, Korean and English only"),
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
    "tiny": tr_noop("Fastest - rough accuracy"),
    "base": tr_noop("Fast - basic accuracy"),
    "small": tr_noop("Good balance for most PCs"),
    "medium": tr_noop("More accurate, needs a decent PC"),
    "large-v3": tr_noop("Most accurate - big download"),
    "large-v3-turbo": tr_noop("Most accurate and fast"),
    "distil-large-v3.5": tr_noop("Near-most accurate, fast"),
    "distil-small.en": tr_noop("Fast, small download"),
    "parakeet-tdt-0.6b-v3": tr_noop("Very accurate and fast"),
    "sense-voice-small": tr_noop("Fast and accurate, small download"),
}

# The M2M entries read as they do because they are measured, not because they
# are small: at beam 1 both M2M sizes render a bare "Okay" correctly where both
# NLLB sizes fabricate a sentence. Calling the recommended default "lower
# quality", as this table did while NLLB led, contradicted both the badge
# beside it and the measurement behind the recommendation.
_MT_LEAD_INS: dict[str, str] = {
    "nllb-600M-int8": tr_noop("Balanced"),
    "nllb-1.3B-int8": tr_noop("Higher quality"),
    "nllb-3.3B-int8": tr_noop("Best quality (large)"),
    "m2m100-418M-int8": tr_noop("Balanced"),
    "m2m100-1.2B-int8": tr_noop("Higher quality"),
    "madlad400-3b": tr_noop("Best quality (large)"),
}


def collapsed_pair(family: str) -> tuple[str, str] | None:
    """The first language ``family`` cannot write distinctly and the language
    it writes instead, or ``None`` when it keeps every language apart.

    Derived from the registry's own collapse map rather than declared here, so
    the blurb cannot claim a limitation the decoder does not have. Today each
    affected family collapses exactly one pair (Chinese Traditional onto
    Simplified); a family that grew a second would need this to list more.
    """
    for display in LANGUAGES:
        other = collapses_onto(family, display)
        if other is not None:
            return display, other
    return None


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
    for voice specs with ``english_only`` True and the spec's own
    ``language_note`` for other language-restricted voice specs. Unknown ids
    return ``""``.

    An MT family that cannot keep two languages apart says so here too. Two
    families read "Balanced" at different sizes, and without this the only
    difference the user could see between them was the number of megabytes,
    while one of them silently answers a Chinese Traditional request in
    Simplified.
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
        elif spec.language_note is not None:
            blurb += " · " + tr(spec.language_note)
        return blurb
    if kind == "mt":
        spec = MT_MODELS.get(model_id)
        if spec is None:
            return ""
        raw_lead_in = _MT_LEAD_INS.get(model_id)
        lead_in = tr(raw_lead_in) if raw_lead_in else _generic_lead_in(spec.size_mb)
        parts = [lead_in, fmt_size(spec.size_mb)]
        blurb = " · ".join(parts)
        # Both halves, because the licence is the ONLY reason to prefer an
        # m2m100 and the accuracy gap is the only reason not to. Naming just
        # one of them would steer the user without telling them what it costs.
        if "NC" in spec.license:
            blurb += " · " + tr("personal use only")
        elif spec.family == "m2m100":
            blurb += " · " + tr(
                "fine for paid streaming, less accurate than NLLB"
            )
        collapsed = collapsed_pair(spec.family)
        if collapsed is not None:
            blurb += " · " + tr(
                "writes {language} as {other}",
                language=collapsed[0], other=collapsed[1],
            )
        return blurb
    return ""


def mt_license_note(model_id: str) -> str:
    """One sentence naming an MT model's license, for the first-run wizard.

    Only a CC-BY-NC license restricts use to personal and non-commercial; MIT
    and Apache-2.0 grant commercial use, so saying otherwise about them would
    be false. Same "NC" test :func:`model_blurb` uses, kept beside it so the
    two cannot say different things about one model. Unknown ids get "".
    """
    spec = MT_MODELS.get(model_id)
    if spec is None:
        return ""
    if "NC" in spec.license:
        return tr(
            "Note: the translation model is licensed {license} "
            "(free for personal, non-commercial use).",
            license=spec.license,
        )
    return tr("Note: the translation model is licensed {license}.",
              license=spec.license)
