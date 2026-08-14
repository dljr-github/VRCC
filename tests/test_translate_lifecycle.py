"""TranslateEngine lifecycle: warm-up, unload, and target resolution.

Split from test_translate_engine (which covers token layout) for the source
cap. What these have in common is that they are about which decodes RUN rather
than how one is shaped: a warm-up exists to touch the decoder before the first
real utterance, so one that decodes nothing has verified nothing, and a family
that renders two languages alike must decode them once rather than twice.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from vrcc.core.bus import EventBus
from vrcc.core.languages import get
from vrcc.translate.engine import TranslateEngine

from tests.test_translate_engine import (  # noqa: F401
    _RecordingFactory,
    _cfg,
    _spec,
    model_dir,
    toy_spm_path,
)


# warm_up / unload
# --------------------------------------------------------------------------

def test_warm_up_uses_first_configured_target(model_dir: Path):
    factory = _RecordingFactory()
    cfg = _cfg(targets=["Korean", "Japanese"])
    eng = TranslateEngine(_spec("nllb"), model_dir, cfg, EventBus(), translator_factory=factory)
    eng.load()

    eng.warm_up()

    call = factory.built[0].batch_calls[0]
    assert len(call.source) == 1
    assert call.target_prefix == [["kor_Hang"]]  # first configured target


def test_warm_up_defaults_to_japanese_without_targets(model_dir: Path):
    factory = _RecordingFactory()
    eng = TranslateEngine(_spec("nllb"), model_dir, _cfg(targets=[]), EventBus(), translator_factory=factory)
    eng.load()

    eng.warm_up()

    assert factory.built[0].batch_calls[0].target_prefix == [["jpn_Jpan"]]


def test_unload_is_safe_before_and_after_load(model_dir: Path):
    factory = _RecordingFactory()
    eng = TranslateEngine(_spec("nllb"), model_dir, _cfg(), EventBus(), translator_factory=factory)

    eng.unload()  # safe with nothing loaded

    eng.load()
    eng.unload()
    with pytest.raises(RuntimeError):
        eng.translate("hi", get("English"), [get("Japanese")])


def test_module_does_not_import_ctranslate2_eagerly():
    import vrcc.translate.engine as engine_mod

    # ctranslate2 is imported lazily inside load(), never at module top level.
    assert not hasattr(engine_mod, "ctranslate2")


# --------------------------------------------------------------------------
# Targets the family cannot tell apart
# --------------------------------------------------------------------------

def test_collapsing_family_decodes_once_for_both_chinese_scripts(model_dir: Path):
    factory = _RecordingFactory()
    eng = TranslateEngine(_spec("m2m100"), model_dir, _cfg(), EventBus(), translator_factory=factory)
    eng.load()

    out = eng.translate(
        "hello world",
        get("English"),
        [get("Chinese Simplified"), get("Chinese Traditional"), get("Japanese")],
    )

    call = factory.built[0].batch_calls[0]
    # Two prefixes, not three: the second Chinese entry would have carried the
    # same __zh__ token and returned the same text into a second chatbox line.
    assert call.target_prefix == [["__zh__"], ["__ja__"]]
    assert [name for name, _ in out] == ["Chinese Simplified", "Japanese"]


def test_nllb_keeps_both_chinese_scripts(model_dir: Path):
    factory = _RecordingFactory()
    eng = TranslateEngine(_spec("nllb"), model_dir, _cfg(), EventBus(), translator_factory=factory)
    eng.load()

    out = eng.translate(
        "hello world",
        get("English"),
        [get("Chinese Simplified"), get("Chinese Traditional")],
    )

    assert factory.built[0].batch_calls[0].target_prefix == [["zho_Hans"], ["zho_Hant"]]
    assert [name for name, _ in out] == ["Chinese Simplified", "Chinese Traditional"]


def test_warm_up_skips_a_target_equal_to_the_source(model_dir: Path):
    """warm_up translates from English, and translate() drops a target that
    resolves onto the source, so a first target of English decoded NOTHING and
    still reported a healthy engine. The point of warming up is touching the
    decoder before the first real utterance."""
    factory = _RecordingFactory()
    cfg = _cfg(targets=["English", "Korean"])
    eng = TranslateEngine(_spec("nllb"), model_dir, cfg, EventBus(), translator_factory=factory)
    eng.load()

    eng.warm_up()

    call = factory.built[0].batch_calls[0]
    assert call.target_prefix == [["kor_Hang"]], "must warm on a real target"


def test_warm_up_falls_back_to_japanese_when_every_target_is_the_source(model_dir: Path):
    factory = _RecordingFactory()
    cfg = _cfg(targets=["English"])
    eng = TranslateEngine(_spec("nllb"), model_dir, cfg, EventBus(), translator_factory=factory)
    eng.load()

    eng.warm_up()

    assert factory.built[0].batch_calls[0].target_prefix == [["jpn_Jpan"]]


def test_warm_up_raises_when_it_decoded_nothing(model_dir: Path):
    """A warm-up that produced no output has verified nothing, so it must not
    pass as a healthy engine."""
    factory = _RecordingFactory()
    eng = TranslateEngine(_spec("nllb"), model_dir, _cfg(), EventBus(), translator_factory=factory)
    eng.load()
    eng.translate = lambda *a, **k: []

    with pytest.raises(RuntimeError, match="warm-up"):
        eng.warm_up()


def test_warm_up_ignores_a_target_the_registry_does_not_know(model_dir: Path):
    # A hand-edited config must not break startup.
    factory = _RecordingFactory()
    cfg = _cfg(targets=["Klingon", "Korean"])
    eng = TranslateEngine(_spec("nllb"), model_dir, cfg, EventBus(), translator_factory=factory)
    eng.load()

    eng.warm_up()

    assert factory.built[0].batch_calls[0].target_prefix == [["kor_Hang"]]
