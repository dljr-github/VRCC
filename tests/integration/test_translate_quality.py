"""Real-model translation quality gates, run against every model VRCC ships.

Integration-marked: each parameter loads a real CTranslate2 model (~0.5 to
1.4 GB) from the shared models dir, so this is opt-in.

What it does NOT do is pin exact output strings. Those move with the
ctranslate2 version and with any weight revision, so a snapshot test here would
fail for reasons that are not regressions and would be re-blessed until it
meant nothing. It asserts properties instead, chosen because they are exactly
what separated working decodes from broken ones when this was measured:

- digits survive. Greedy decoding lost them outright ("Okay 1,2,3,4,5,6,7" came
  back as "现在,我们要做什么?"), and no correct translation of a number drops it.
- the output is in the target's script, not the source's.
- nothing is echoed back untranslated, and nothing comes back empty.
- no control-token or unknown-token fragments reach the caption.
- no runaway repetition, the failure the anti-loop guards exist to prevent.

Every one of those is objective. Quality judgements that need a human (or a
panel) live in the blind A/B that chose the default, not in the suite.
"""

from __future__ import annotations

import re

import pytest

from vrcc.core.bus import EventBus
from vrcc.core.config import TranslateConfig, default_paths
from vrcc.core.languages import get
from vrcc.download.manager import DownloadManager
from vrcc.translate.engine import TranslateEngine
from vrcc.translate.registry import MT_MODELS

pytestmark = pytest.mark.integration

# Short, casual, VRChat-shaped. Deliberately the register the app runs on
# rather than newswire text, since that is where a general MT benchmark stops
# predicting anything useful.
_CORPUS = [
    "Hey, can you hear me okay?",
    "My microphone was muted, sorry about that.",
    "I will be back in 5 minutes.",
    "Good morning, how are you doing today?",
    "That world was really cool, thanks for showing me.",
]

# Kept out of _CORPUS: both models render this as "1, 2, 3, 4, 5, 6, 7.",
# dropping "Okay" entirely, so there is no target script to find. Whether that
# drop is acceptable is a judgement call and not what this file decides; the
# digits are the objective part, and greedy decoding lost those too.
_NUMERIC = ["Okay 1, 2, 3, 4, 5, 6, 7."]

_TARGETS = ["Japanese", "Chinese Simplified", "Korean"]

# The models a user is actually steered onto. The 3B+ entries are excluded on
# download cost, not because they are exempt.
_MODELS = ["nllb-600M-int8", "m2m100-418M-int8"]

_CJK = re.compile(r"[぀-ヿ㐀-鿿가-힯]")
_TOKEN_ARTIFACTS = ("<unk>", "</s>", "<s>", "_Latn", "_Hans", "_Hant", "__")


def _engine(model_id: str):
    spec = MT_MODELS[model_id]
    bus = EventBus()
    model_dir = DownloadManager(
        default_paths(portable=False).models_dir, bus
    ).ensure_mt(spec)
    # Default decoding on purpose: this gate is meaningless against settings no
    # user runs.
    cfg = TranslateConfig(model=spec.id, device="cpu", compute_type="int8")
    engine = TranslateEngine(spec, model_dir, cfg, bus)
    engine.load()
    return engine


def _longest_run(text: str, unit: int = 4) -> int:
    """How many times the most-repeated `unit`-character slice repeats back to
    back. A translator stuck in a loop scores high; ordinary text does not."""
    best = 1
    for start in range(max(0, len(text) - unit)):
        piece = text[start : start + unit]
        if not piece.strip():
            continue
        run = 1
        while text.startswith(piece * (run + 1), start):
            run += 1
        best = max(best, run)
    return best


@pytest.fixture(scope="module", params=_MODELS)
def engine(request):
    eng = _engine(request.param)
    yield request.param, eng
    eng.unload()


def test_translations_are_present_and_in_the_right_script(engine):
    model_id, eng = engine
    source = get("English")

    for text in _CORPUS:
        results = eng.translate(text, source, [get(t) for t in _TARGETS])
        assert results, (model_id, text)
        for name, out in results:
            assert out.strip(), f"{model_id} returned nothing for {text!r} -> {name}"
            assert _CJK.search(out), (
                f"{model_id} produced no {name} script for {text!r}: {out!r}"
            )


def test_nothing_is_echoed_back_untranslated(engine):
    model_id, eng = engine
    source = get("English")

    for text in _CORPUS + _NUMERIC:
        for name, out in eng.translate(text, source, [get(t) for t in _TARGETS]):
            assert out.strip().lower() != text.strip().lower(), (
                f"{model_id} echoed the source for {name}: {out!r}"
            )


def test_digits_survive_translation(engine):
    """The single most diagnostic property here. Greedy decoding did not
    mistranslate the counting utterance, it replaced it with an unrelated
    sentence and dropped every digit."""
    model_id, eng = engine
    source = get("English")

    # The counting utterance is almost nothing BUT digits, so every one has to
    # come through. Leniency here is what let the original regression pass: at
    # beam 1 the output contained no digits at all, and a rule that only
    # compared digits "when the model emitted some" simply skipped itself.
    counting = "Okay 1, 2, 3, 4, 5, 6, 7."
    expected = set("1234567")
    for name, out in eng.translate(counting, source, [get(t) for t in _TARGETS]):
        assert set(re.findall(r"\d", out)) == expected, (
            f"{model_id} lost digits for {name}: {counting!r} -> {out!r}"
        )

    # Embedded in a sentence, a target may legitimately spell the number out,
    # so only require that what it does emit is right.
    embedded = "I will be back in 5 minutes."
    for name, out in eng.translate(embedded, source, [get(t) for t in _TARGETS]):
        got = set(re.findall(r"\d", out))
        assert got in (set(), {"5"}), (
            f"{model_id} invented digits for {name}: {embedded!r} -> {out!r}"
        )


def test_no_token_artifacts_reach_the_caption(engine):
    model_id, eng = engine
    source = get("English")

    for text in _CORPUS + _NUMERIC:
        for name, out in eng.translate(text, source, [get(t) for t in _TARGETS]):
            for artifact in _TOKEN_ARTIFACTS:
                assert artifact not in out, (
                    f"{model_id} leaked {artifact!r} into {name}: {out!r}"
                )


def test_no_runaway_repetition(engine):
    """What repetition_penalty and no_repeat_ngram_size are set for. A loop
    fills every chatbox line with one phrase and is the most visible failure a
    translator has."""
    model_id, eng = engine
    source = get("English")

    for text in _CORPUS + _NUMERIC:
        for name, out in eng.translate(text, source, [get(t) for t in _TARGETS]):
            assert _longest_run(out) < 4, (
                f"{model_id} looped for {name}: {out!r}"
            )
