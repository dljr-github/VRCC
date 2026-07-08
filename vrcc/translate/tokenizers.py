"""SentencePiece / HuggingFace tokenization for the CTranslate2 MT models.

CT2 consumes/emits token *strings*. Backend chosen by extension (``*.model`` ->
SentencePiece, ``tokenizer.json`` -> HuggingFace); both yield raw pieces with NO
special tokens (layout adds them); decode maps ``▁`` to a space (never HF decode).
"""

from __future__ import annotations

import re
from pathlib import Path

from vrcc.core.languages import Language
from vrcc.translate.registry import lang_token

# SentencePiece metaspace marker (stands in for a leading space).
_METASPACE = "▁"  # ▁

_KNOWN_FAMILIES = ("nllb", "m2m100", "madlad")

# -- decode: drop control/special tokens --

# Exact structural tokens shared across the model families.
_SPECIAL_LITERALS = frozenset({"</s>", "<pad>", "<s>", "<unk>"})

# m2m100 language tags, e.g. "__ja__", "__yue__".
_M2M100_TAG = re.compile(r"^__[a-z]{2,3}__$")

# NLLB / FLORES-200 language codes, e.g. "jpn_Jpan", "eng_Latn".
_FLORES_TAG = re.compile(r"^[a-z]{2,3}_[A-Z][a-z]{3}$")

# MADLAD source-side target tags, e.g. "<2ja>", "<2fr>".
_MADLAD_TAG = re.compile(r"^<2[a-z]{2,3}>$")


def _is_special(token: str) -> bool:
    """True if ``token`` is a control token that decode must drop."""
    return (
        token in _SPECIAL_LITERALS
        or bool(_M2M100_TAG.match(token))
        or bool(_FLORES_TAG.match(token))
        or bool(_MADLAD_TAG.match(token))
    )


class MtTokenizer:
    """Tokenizer + per-family token layout for one CTranslate2 MT model.

    ``tokenizer_path`` selects the backend by extension; ``family`` is one of
    ``"nllb" | "m2m100" | "madlad"`` and drives the control-token layout.
    """

    def __init__(self, tokenizer_path: Path, family: str) -> None:
        if family not in _KNOWN_FAMILIES:
            raise ValueError(
                f"Unknown MT family: {family!r}. "
                f"Known families: {list(_KNOWN_FAMILIES)}"
            )
        self.family = family

        path = Path(tokenizer_path)
        suffix = path.suffix.lower()
        if suffix == ".model":
            import sentencepiece as spm

            sp = spm.SentencePieceProcessor()
            sp.load(str(path))
            self._backend = "spm"
            self._sp = sp
        elif suffix == ".json":
            from tokenizers import Tokenizer

            self._backend = "hf"
            self._tok = Tokenizer.from_file(str(path))
        else:
            raise ValueError(
                f"Unsupported tokenizer file {path.name!r}: expected a "
                f"'*.model' SentencePiece model or a 'tokenizer.json'."
            )

    # -- internals ---------------------------------------------------------

    def _pieces(self, text: str) -> list[str]:
        """Raw sub-word pieces for ``text`` with no special tokens attached."""
        if self._backend == "spm":
            return self._sp.encode(text, out_type=str)
        # HuggingFace backend: control tokens are added by the layout code,
        # so keep them off here to match the SentencePiece path exactly.
        return self._tok.encode(text, add_special_tokens=False).tokens

    # -- public API --------------------------------------------------------

    def encode_source(
        self,
        text: str,
        src: Language,
        tgt: Language | None = None,
    ) -> list[str]:
        """Lay the source-side control tokens around the pieces of ``text``.

        nllb/m2m100: ``[<src tag>] + pieces + ["</s>"]``. madlad: ``[<tgt tag>]
        + pieces + ["</s>"]`` -- the *target* rides on the source side, so
        ``tgt`` is required (``ValueError`` if omitted).
        """
        pieces = self._pieces(text)
        if self.family == "madlad":
            if tgt is None:
                raise ValueError(
                    "madlad encodes the target language on the source side; "
                    "encode_source requires a tgt Language."
                )
            head = lang_token("madlad", tgt)
        else:
            head = lang_token(self.family, src)
        return [head, *pieces, "</s>"]

    def target_prefix(self, tgt: Language) -> list[str]:
        """Decoder start tokens forcing the target language.

        nllb: ``[<tgt FLORES>]``; m2m100: ``[<__tgt__>]``; madlad: ``[]``
        (target already rode in on the source side).
        """
        if self.family == "madlad":
            return []
        return [lang_token(self.family, tgt)]

    def decode(self, tokens: list[str]) -> str:
        """Turn model output tokens back into plain text: drop every
        control/special token (``</s>``, ``<pad>``, ``<s>``, ``<unk>``, echoed
        language tags), join the survivors, map the meta-space ``▁`` to a space,
        collapse whitespace.
        """
        kept = [t for t in tokens if not _is_special(t)]
        text = "".join(kept).replace(_METASPACE, " ")
        return " ".join(text.split())
