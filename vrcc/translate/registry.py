"""Static registry of the CTranslate2 MT models VRCC offers.

Each entry names a HuggingFace CT2 repo. ``spm_file`` is the tokenizer file
required for a complete download; ``prefix_side`` records where the target token
is injected (target: nllb/m2m100, source: madlad). ``lang_token`` renders it.
"""

from __future__ import annotations

from dataclasses import dataclass

from vrcc.core.languages import Language


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
