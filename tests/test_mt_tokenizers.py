"""Tests for :mod:`vrcc.translate.tokenizers`, pinning the per-family token
layout and special-token stripping via toy SentencePiece and HuggingFace
``tokenizers`` models (never the real NLLB/M2M100/MADLAD tokenizers).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from vrcc.core.languages import get
from vrcc.translate.tokenizers import MtTokenizer

# Toy corpus: lowercase words only, enough distinct substrings to train a
# small unigram model. "hello world" appears so round-trip tests have a
# stable target.
_SENTENCES = [
    "hello world",
    "foo bar baz",
    "the quick brown fox",
    "lorem ipsum dolor sit",
    "hello foo world bar",
    "quick brown lorem",
    "dolor sit amet",
    "the fox jumps over",
    "baz world hello there",
    "amet ipsum dolor now",
]


@pytest.fixture(scope="session")
def toy_spm_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Train a tiny SentencePiece unigram model once and return its .model path."""
    import sentencepiece as spm

    out = tmp_path_factory.mktemp("spm")
    prefix = out / "toy"
    # vocab_size=64 is more than 10 short sentences can fill, so relax the
    # hard limit -- the trainer emits as many pieces as it can.
    spm.SentencePieceTrainer.train(
        sentence_iterator=iter(_SENTENCES),
        model_prefix=str(prefix),
        vocab_size=64,
        hard_vocab_limit=False,
    )
    model = Path(str(prefix) + ".model")
    assert model.exists()
    return model


@pytest.fixture(scope="session")
def toy_json_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Build a minimal HF ``tokenizer.json`` (WordLevel + Metaspace).

    Metaspace prefixes the first sub-token of each word with ``▁`` (U+2581),
    mirroring how real SentencePiece-derived ``tokenizer.json`` files behave,
    so the tokenizer exercises the same ``▁``-normalisation decode path as the
    SentencePiece backend. Round-trip is verified in-fixture.
    """
    from tokenizers import Tokenizer, models, pre_tokenizers

    words: set[str] = set()
    for sentence in _SENTENCES:
        words.update(sentence.split())

    vocab: dict[str, int] = {"<unk>": 0}
    for word in sorted(words):
        vocab["▁" + word] = len(vocab)
        vocab[word] = len(vocab)

    tok = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Metaspace()

    # in-fixture round-trip sanity check
    tokens = tok.encode("hello world", add_special_tokens=False).tokens
    assert "".join(tokens).replace("▁", " ").strip() == "hello world"

    out = tmp_path_factory.mktemp("hf")
    path = out / "tokenizer.json"
    tok.save(str(path))
    assert path.exists()
    return path


# --------------------------------------------------------------------------
# Backend selection
# --------------------------------------------------------------------------

def test_spm_backend_selected_for_dot_model(toy_spm_path: Path):
    tok = MtTokenizer(toy_spm_path, "nllb")
    # if the wrong backend loaded, encoding would fail
    assert tok.encode_source("hello world", get("Japanese"))


def test_json_backend_selected_for_tokenizer_json(toy_json_path: Path):
    tok = MtTokenizer(toy_json_path, "nllb")
    assert tok.encode_source("hello world", get("Japanese"))


def test_bpe_model_extension_uses_spm_backend(
    toy_spm_path: Path, tmp_path: Path
):
    # a "sentencepiece.bpe.model" name must route to the SentencePiece backend
    bpe = tmp_path / "sentencepiece.bpe.model"
    bpe.write_bytes(toy_spm_path.read_bytes())
    tok = MtTokenizer(bpe, "m2m100")
    assert tok.encode_source("hello world", get("Japanese"))


def test_unsupported_extension_raises_value_error(tmp_path: Path):
    bogus = tmp_path / "vocab.txt"
    bogus.write_text("x")
    with pytest.raises(ValueError):
        MtTokenizer(bogus, "nllb")


def test_unknown_family_raises_value_error(toy_spm_path: Path):
    with pytest.raises(ValueError):
        MtTokenizer(toy_spm_path, "bogus")


# --------------------------------------------------------------------------
# nllb layout
# --------------------------------------------------------------------------

def test_nllb_source_first_is_src_flores_last_is_eos(toy_spm_path: Path):
    tok = MtTokenizer(toy_spm_path, "nllb")
    out = tok.encode_source("hello world", get("Japanese"))
    assert out[0] == "jpn_Jpan"
    assert out[-1] == "</s>"
    assert len(out) > 2  # some pieces between the control tokens


def test_nllb_target_prefix_is_tgt_flores(toy_spm_path: Path):
    tok = MtTokenizer(toy_spm_path, "nllb")
    assert tok.target_prefix(get("Japanese")) == ["jpn_Jpan"]
    assert tok.target_prefix(get("English")) == ["eng_Latn"]


# --------------------------------------------------------------------------
# m2m100 layout
# --------------------------------------------------------------------------

def test_m2m100_source_first_is_src_token_last_is_eos(toy_spm_path: Path):
    tok = MtTokenizer(toy_spm_path, "m2m100")
    out = tok.encode_source("hello world", get("Japanese"))
    assert out[0] == "__ja__"
    assert out[-1] == "</s>"


def test_m2m100_target_prefix_wraps_code(toy_spm_path: Path):
    tok = MtTokenizer(toy_spm_path, "m2m100")
    assert tok.target_prefix(get("Japanese")) == ["__ja__"]
    assert tok.target_prefix(get("English")) == ["__en__"]


# --------------------------------------------------------------------------
# madlad layout (source-prefix family)
# --------------------------------------------------------------------------

def test_madlad_source_first_is_target_tag_last_is_eos(toy_spm_path: Path):
    tok = MtTokenizer(toy_spm_path, "madlad")
    # madlad injects the *target* language on the source side
    out = tok.encode_source("hello world", get("English"), get("Japanese"))
    assert out[0] == "<2ja>"
    assert out[-1] == "</s>"


def test_madlad_target_prefix_is_empty(toy_spm_path: Path):
    tok = MtTokenizer(toy_spm_path, "madlad")
    assert tok.target_prefix(get("Japanese")) == []


def test_madlad_requires_tgt_in_encode_source(toy_spm_path: Path):
    tok = MtTokenizer(toy_spm_path, "madlad")
    with pytest.raises(ValueError):
        tok.encode_source("hello world", get("English"))


# --------------------------------------------------------------------------
# decode: special-token stripping
# --------------------------------------------------------------------------

def test_decode_strips_every_special_class(toy_spm_path: Path):
    tok = MtTokenizer(toy_spm_path, "nllb")
    tokens = [
        "<s>",
        "jpn_Jpan",      # FLORES shape
        "__ja__",        # m2m100 shape
        "<2ja>",         # madlad tag
        "▁hello",
        "▁world",
        "</s>",
        "<pad>",
        "<unk>",
    ]
    assert tok.decode(tokens) == "hello world"


def test_decode_strips_three_letter_lang_codes(toy_spm_path: Path):
    tok = MtTokenizer(toy_spm_path, "m2m100")
    # __yue__ (3-letter) and eng_Latn (3-letter FLORES) must also be dropped
    tokens = ["__yue__", "eng_Latn", "▁hello", "▁world"]
    assert tok.decode(tokens) == "hello world"


def test_decode_collapses_and_strips_whitespace(toy_spm_path: Path):
    tok = MtTokenizer(toy_spm_path, "nllb")
    tokens = ["▁", "▁hello", "▁▁world", "</s>"]
    assert tok.decode(tokens) == "hello world"


def test_decode_empty_after_stripping_is_empty_string(toy_spm_path: Path):
    tok = MtTokenizer(toy_spm_path, "nllb")
    assert tok.decode(["</s>", "<pad>", "jpn_Jpan"]) == ""


# --------------------------------------------------------------------------
# round-trip on both backends
# --------------------------------------------------------------------------

@pytest.mark.parametrize("family", ["nllb", "m2m100"])
def test_round_trip_hello_world_spm_backend(toy_spm_path: Path, family: str):
    tok = MtTokenizer(toy_spm_path, family)
    encoded = tok.encode_source("hello world", get("English"))
    assert tok.decode(encoded) == "hello world"


@pytest.mark.parametrize("family", ["nllb", "m2m100"])
def test_round_trip_hello_world_json_backend(toy_json_path: Path, family: str):
    tok = MtTokenizer(toy_json_path, family)
    encoded = tok.encode_source("hello world", get("English"))
    assert tok.decode(encoded) == "hello world"


def test_round_trip_madlad_spm_backend(toy_spm_path: Path):
    tok = MtTokenizer(toy_spm_path, "madlad")
    encoded = tok.encode_source("hello world", get("English"), get("Japanese"))
    assert tok.decode(encoded) == "hello world"
