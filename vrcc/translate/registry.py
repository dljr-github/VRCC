"""Static registry of the CTranslate2 MT models VRCC offers.

Each entry names a HuggingFace CT2 repo. ``spm_file`` is the tokenizer file
required for a complete download; ``prefix_side`` records where the target token
is injected (target: nllb/m2m100, source: madlad). ``lang_token`` renders it.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from types import MappingProxyType

from vrcc.core.languages import LANGUAGES, Language


@dataclass(frozen=True)
class MtModelSpec:
    id: str
    repo: str
    family: str              # "nllb" | "m2m100" | "madlad"
    size_mb: int
    license: str
    spm_file: str            # tokenizer file that must exist inside the repo
    prefix_side: str         # "target" (nllb/m2m100) | "source" (madlad)


MT_MODELS: dict[str, MtModelSpec] = {
    spec.id: spec
    for spec in (
        MtModelSpec(
            "nllb-600M-int8",
            "JustFrederik/nllb-200-distilled-600M-ct2-int8",
            "nllb",
            647,
            "CC-BY-NC-4.0",
            "sentencepiece.bpe.model",
            "target",
        ),
        MtModelSpec(
            "nllb-1.3B-int8",
            "OpenNMT/nllb-200-distilled-1.3B-ct2-int8",
            "nllb",
            1400,
            "CC-BY-NC-4.0",
            "tokenizer.json",
            "target",
        ),
        MtModelSpec(
            "nllb-3.3B-int8",
            "OpenNMT/nllb-200-3.3B-ct2-int8",
            "nllb",
            3300,
            "CC-BY-NC-4.0",
            "tokenizer.json",
            "target",
        ),
        MtModelSpec(
            "m2m100-418M-int8",
            "jncraton/m2m100_418M-ct2-int8",
            "m2m100",
            483,
            "MIT",
            "sentencepiece.bpe.model",
            "target",
        ),
        MtModelSpec(
            "m2m100-1.2B-int8",
            "jncraton/m2m100_1.2B-ct2-int8",
            "m2m100",
            1200,
            "MIT",
            "sentencepiece.bpe.model",
            "target",
        ),
        MtModelSpec(
            "madlad400-3b",
            "santhosh/madlad400-3b-ct2",
            "madlad",
            3500,
            "Apache-2.0",
            "sentencepiece.model",
            "source",
        ),
    )
}

_KNOWN_FAMILIES = ("nllb", "m2m100", "madlad")


def lang_token(family: str, lang: Language) -> str:
    """Render ``lang`` into the control token that ``family`` expects.

    nllb -> FLORES-200 code (``"jpn_Jpan"``); m2m100 -> ``"__ja__"``; madlad ->
    ``"<2ja>"``. Raises ``ValueError`` for an unknown family.
    """
    if family == "nllb":
        return lang.nllb
    if family == "m2m100":
        return f"__{lang.m2m100}__"
    if family == "madlad":
        return f"<2{lang.m2m100}>"
    raise ValueError(
        f"Unknown MT family: {family!r}. Known families: {list(_KNOWN_FAMILIES)}"
    )


@lru_cache(maxsize=None)
def _collapse_map(family: str) -> Mapping[str, str]:
    """Display name -> the earlier language whose control token it shares.

    Derived from :func:`lang_token` rather than declared per model, so it
    cannot drift from what the engine actually puts in front of the decoder,
    and a Language that later gains a script-specific code stops collapsing
    with nothing else to update. Cached because LANGUAGES never changes at
    runtime; a test that patches it must patch ``registry.LANGUAGES`` (the name
    this module binds) and call ``_collapse_map.cache_clear()``. Read-only
    because the cache hands every caller the same object, so one mutation would
    change what every later target check sees.
    """
    first: dict[str, str] = {}
    collapsed: dict[str, str] = {}
    for display, lang in LANGUAGES.items():
        token = lang_token(family, lang)
        if token in first:
            collapsed[display] = first[token]
        else:
            first[token] = display
    return MappingProxyType(collapsed)


def collapses_onto(family: str, display: str) -> str | None:
    """The language ``display`` is indistinguishable from under ``family``, or
    ``None`` when the family renders it distinctly.

    m2m100 and madlad carry one Chinese token where nllb carries two, so a
    Chinese Traditional request reaches their decoder as the Simplified one and
    comes back in Simplified characters.
    """
    return _collapse_map(family).get(display)


def distinct_targets(
    family: str, targets: list[Language], source: Language | None = None
) -> list[Language]:
    """``targets`` resolved onto the language that owns each control token,
    deduplicated in first-seen order, minus anything that lands on ``source``.

    Two identical decoder prefixes cost two beam searches and return the same
    text, which would then occupy two of the three chatbox lines.

    Resolved rather than merely deduplicated, so the name on the caption
    describes the text that came back: keeping the entry as asked would label
    Simplified output "Chinese Traditional". Same rule
    :func:`vrcc.gui.mt_prompts.usable_targets` applies to the stored list, so
    the caption log and the combo agree.

    The source check happens AFTER resolution, which is the whole point of
    doing it here: a Chinese Traditional target under a Chinese Simplified
    source is distinct by name, but this family renders both with one token, so
    it decodes the source language into itself. Measured on the real m2m100,
    that returns a same-language paraphrase rather than a translation
    ("我这个学期要去台湾" -> "这个学期我要去台湾。"), which is a chatbox line
    spent saying nothing.

    Direction matters and only this one is dropped. The reverse, a Chinese
    Simplified target under a Chinese Traditional source, resolves to a
    language that is NOT the source and delivers real script conversion, so it
    survives (see vrcc.gui.main_targets.rebuild_targets, which measures it).
    """
    seen: set[str] = set()
    out: list[Language] = []
    for lang in targets:
        canonical = collapses_onto(family, lang.display)
        resolved = LANGUAGES[canonical] if canonical else lang
        if source is not None and resolved == source:
            continue
        token = lang_token(family, resolved)
        if token not in seen:
            seen.add(token)
            out.append(resolved)
    return out
