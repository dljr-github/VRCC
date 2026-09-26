"""Wizard language preferences must steer decoding, not just its language label."""

from types import SimpleNamespace

import numpy as np
import pytest

from vrcc.core.bus import EventBus
from vrcc.stt.engine import SttEngine
from .stt_fakes import MODEL_DIR, _cfg, _collect, _OOM_TEXT, _seg


class DetectionModel:
    """Whisper returns language info eagerly and decodes its segments lazily."""

    def __init__(self, probabilities, fail_for=None):
        self.probabilities = probabilities
        self.fail_for = fail_for
        self.calls = []
        self.decoded = []

    def transcribe(self, samples, **kwargs):
        self.calls.append(kwargs)
        language = kwargs.get("language") or self.probabilities[0][0]
        if language == self.fail_for:
            raise RuntimeError(_OOM_TEXT)

        def segments():
            self.decoded.append(language)
            yield _seg(f"caption in {language}", 0, 1, -0.1, 0.05)

        return segments(), SimpleNamespace(
            language=language,
            all_language_probs=(
                self.probabilities if kwargs.get("language") is None else None
            ),
        )


def engine_for(probabilities, **overrides):
    cfg = _cfg(source_language="auto", spoken_languages=["English", "Japanese"])
    for key, value in overrides.items():
        setattr(cfg, key, value)
    model = DetectionModel(probabilities)
    engine = SttEngine(cfg, MODEL_DIR, EventBus(), model_factory=lambda *a, **k: model)
    engine.load()
    return engine, model


@pytest.mark.parametrize(
    "probabilities, expected",
    [
        ([("fr", 0.45), ("en", 0.35), ("ja", 0.20)], "en"),
        ([("fr", 0.90), ("en", 0.06), ("ja", 0.04)], "fr"),
        ([("ja", 0.50), ("en", 0.40), ("fr", 0.10)], "ja"),
    ],
)
def test_auto_prefers_selected_languages_but_allows_clear_outside_match(probabilities, expected):
    engine, model = engine_for(probabilities)
    result = engine.transcribe(np.zeros(1600, dtype=np.float32))
    assert result.language == expected
    assert result.text == f"caption in {expected}"
    # The rejected auto-detected language must never be decoded.
    assert model.decoded == [expected]


@pytest.mark.parametrize("spoken", [[], ["unknown", "auto"], ["Korean"]])
def test_absent_or_irrelevant_preferences_keep_auto_detection(spoken):
    engine, model = engine_for([("fr", 0.55), ("en", 0.45)], spoken_languages=spoken)
    assert engine.transcribe(np.zeros(1600)).language == "fr"
    assert len(model.calls) == 1


@pytest.mark.parametrize(
    "overrides, detect, expected",
    [
        ({"source_language": "French"}, False, "fr"),
        ({}, True, "fr"),
        ({"extra_transcribe_kwargs": {"language": "ko"}}, False, "ko"),
        ({"source_language": "Japanese"}, True, "fr"),
    ],
)
def test_manual_language_and_heard_stream_bypass_preferences(overrides, detect, expected):
    engine, model = engine_for([("fr", 0.55), ("en", 0.45)], **overrides)
    assert engine.transcribe(np.zeros(1600), detect_language=detect).language == expected
    assert len(model.calls) == 1


def test_preferences_are_read_again_for_each_utterance():
    engine, _ = engine_for([("fr", 0.45), ("en", 0.35), ("ja", 0.20)])
    assert engine.transcribe(np.zeros(1600)).language == "en"
    engine._cfg.spoken_languages = ["Japanese"]
    assert engine.transcribe(np.zeros(1600)).language == "fr"


def test_chinese_scripts_do_not_double_the_bias():
    engine, _ = engine_for(
        [("fr", 0.70), ("zh", 0.30)],
        spoken_languages=["Chinese Simplified", "Chinese Traditional"],
    )
    assert engine.transcribe(np.zeros(1600)).language == "fr"


def test_english_only_model_with_no_language_scores_keeps_its_result():
    engine, model = engine_for([("en", 1.0)], spoken_languages=["Japanese"])
    original = model.transcribe

    def transcribe(*args, **kwargs):
        segments, info = original(*args, **kwargs)
        info.all_language_probs = None
        return segments, info

    model.transcribe = transcribe
    assert engine.transcribe(np.zeros(1600)).language == "en"
    assert len(model.calls) == 1


def test_biased_decode_retains_user_prompt_and_decoding_settings():
    engine, model = engine_for(
        [("fr", 0.55), ("en", 0.45)], initial_prompt="VRChat, avatar",
        extra_transcribe_kwargs={"beam_size": 5, "patience": 2.0},
    )
    assert engine.transcribe(np.zeros(1600)).language == "en"
    assert model.calls[-1] == dict(model.calls[0], language="en")
    assert model.calls[-1]["initial_prompt"] == "VRChat, avatar"
    assert model.calls[-1]["beam_size"] == 5


def test_biased_decode_cuda_failure_repeats_detection_on_cpu():
    probabilities = [("fr", 0.55), ("en", 0.45)]
    gpu = DetectionModel(probabilities, fail_for="en")
    cpu = DetectionModel(probabilities)
    models = iter([gpu, cpu])
    bus = EventBus()
    events = _collect(bus)
    engine = SttEngine(
        _cfg(source_language="auto", spoken_languages=["English"]),
        MODEL_DIR, bus, model_factory=lambda *a, **k: next(models),
    )
    engine.load()
    assert engine.transcribe(np.zeros(1600)).language == "en"
    assert cpu.decoded == ["en"]
    assert [e.state for e in events] == ["loading", "ready", "fallback_cpu", "ready"]


def test_detection_segment_majority_is_not_replaced_by_last_segment_scores():
    engine, model = engine_for(
        [("de", 0.45), ("fr", 0.40), ("ko", 0.15)],
        spoken_languages=["Korean"],
        extra_transcribe_kwargs={"language_detection_segments": 3},
    )
    original = model.transcribe

    def transcribe(samples, **kwargs):
        # Faster-whisper can vote French across segments but return the
        # final segment's German-leading probability list in the same info.
        segments, info = original(samples, **dict(kwargs, language=kwargs["language"] or "fr"))
        info.all_language_probs = model.probabilities
        return segments, info

    model.transcribe = transcribe
    result = engine.transcribe(np.zeros(1600))
    assert result.language == "fr"
    assert result.text == "caption in fr"
    assert len(model.calls) == 1


@pytest.mark.parametrize("mode, custom, expected", [
    ("off", ["English"], "fr"),
    ("custom", [], "fr"),
    ("custom", ["French"], "fr"),
    ("custom", ["English"], "en"),
    ("wizard", ["French"], "en"),
])
def test_bias_override_modes(mode, custom, expected):
    engine, _ = engine_for(
        [("fr", 0.55), ("en", 0.45)],
        language_bias_mode=mode, language_bias_languages=custom,
    )
    assert engine.transcribe(np.zeros(1600)).language == expected
    assert engine.transcribe(np.zeros(1600), detect_language=True).language == "fr"
    engine._cfg.source_language = "Japanese"
    assert engine.transcribe(np.zeros(1600)).language == "ja"
