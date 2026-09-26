"""SenseVoice must re-decode with the preferred language input when uncertain."""

import numpy as np
import pytest

from .test_stt_sensevoice import _FakeSession, _engine, _speech
from .test_stt_sensevoice_language_slot import model_dir  # noqa: F401


class LanguageSession(_FakeSession):
    def __init__(self, gap=0.3, first_token=1, bad_retry=False):
        super().__init__([1, 3, 4, 5, 6, 7])
        self.gap = gap
        self.first_token = first_token
        self.bad_retry = bad_retry

    def run(self, outputs, feeds):
        logits = super().run(outputs, feeds)[0]
        if feeds["language"][0] == 0:
            logits[0, 0] = 0
            logits[0, 0, self.first_token] = 12
            logits[0, 0, 2] = 12 - self.gap
        elif feeds["language"][0] == 4:
            # English conditioning changes both the tag and the actual caption.
            logits[0, 0] = 0
            logits[0, 0, 2] = 12
            logits[0, 4] = 0
            logits[0, 4, 7] = 12
            if self.bad_retry:
                logits *= 0.05  # Same nonempty caption, below the confidence gate.
        return [logits]


def make_engine(model_dir, session, **overrides):
    cfg = dict(source_language="auto", spoken_languages=["English"])
    cfg.update(overrides)
    engine = _engine(model_dir, lambda *a, **k: session, **cfg)
    engine.load()
    return engine


def test_ambiguous_detection_reruns_with_preferred_language_input(model_dir):
    session = LanguageSession()
    engine = make_engine(model_dir, session)
    result = engine.transcribe(_speech())
    assert result.language == "en"
    assert result.text == "world"
    assert [run["language"][0] for run in session.runs] == [0, 4]
    np.testing.assert_array_equal(session.runs[0]["x"], session.runs[1]["x"])


def test_clear_outside_language_does_not_rerun(model_dir):
    session = LanguageSession(gap=4)
    engine = make_engine(model_dir, session)
    assert engine.transcribe(_speech()).language == "ja"
    assert len(session.runs) == 1


@pytest.mark.parametrize(
    "overrides, detect",
    [
        ({}, True),
        ({"source_language": "Japanese"}, False),
        ({"spoken_languages": []}, False),
        ({"spoken_languages": ["unknown", "French"]}, False),
        ({"spoken_languages": ["Japanese", "English"]}, False),
    ],
)
def test_manual_heard_and_unusable_preferences_do_not_rerun(model_dir, overrides, detect):
    session = LanguageSession()
    engine = make_engine(model_dir, session, **overrides)
    assert engine.transcribe(_speech(), detect_language=detect).language == "ja"
    assert len(session.runs) == 1


def test_blank_or_unknown_detection_is_not_overridden(model_dir):
    for token in (0, 8):
        session = LanguageSession(first_token=token)
        engine = make_engine(model_dir, session)
        engine.transcribe(_speech())
        assert len(session.runs) == 1


def test_preferences_follow_live_config(model_dir):
    session = LanguageSession()
    engine = make_engine(model_dir, session)
    assert engine.transcribe(_speech()).language == "en"
    engine._cfg.spoken_languages = ["Japanese"]
    assert engine.transcribe(_speech()).language == "ja"


def test_retry_still_passes_through_quality_gates(model_dir):
    engine = make_engine(model_dir, LanguageSession(bad_retry=True))
    assert engine.transcribe(_speech()) is None


def test_ordinary_vocabulary_piece_is_not_a_language_candidate(model_dir):
    # The real vocabulary includes words whose [2:-2] slice is 'en'.
    with (model_dir / "tokens.txt").open("a", encoding="utf-8") as handle:
        handle.write("french 9\n")

    class OrdinaryTokenSession(LanguageSession):
        def run(self, outputs, feeds):
            logits = super().run(outputs, feeds)[0]
            logits = np.pad(logits, ((0, 0), (0, 0), (0, 1)))
            logits[0, 0, 2] = 0  # English tag has no supporting evidence.
            logits[0, 0, 9] = 11.9  # Ordinary word is close to Japanese's 12.
            return [logits]

    session = OrdinaryTokenSession()
    result = make_engine(model_dir, session).transcribe(_speech())
    assert result.language == "ja"
    assert len(session.runs) == 1


@pytest.mark.parametrize("mode, custom, expected", [
    ("off", ["English"], "ja"),
    ("custom", [], "ja"),
    ("custom", ["Japanese"], "ja"),
    ("custom", ["English"], "en"),
    ("wizard", ["Japanese"], "en"),
])
def test_bias_override_modes(model_dir, mode, custom, expected):
    engine = make_engine(
        model_dir, LanguageSession(),
        language_bias_mode=mode, language_bias_languages=custom,
    )
    assert engine.transcribe(_speech()).language == expected
    assert engine.transcribe(_speech(), detect_language=True).language == "ja"
