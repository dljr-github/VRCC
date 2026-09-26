"""Soft language priors for microphone auto detection."""

import math

import numpy as np

from vrcc.core.config import SttConfig
from vrcc.core.languages import LANGUAGES


# Initial policy, not an accuracy-calibrated threshold: a preferred language
# can beat a rival with up to twice its probability, but cannot exclude it.
_PREFERRED_WEIGHT = 2.0


def preferred_codes(cfg: SttConfig, detect_language: bool) -> set[str]:
    """Resolve the saved preference for the user's microphone in Auto mode."""
    if detect_language or cfg.source_language != "auto" or cfg.language_bias_mode == "off":
        return set()
    names = (
        cfg.language_bias_languages if cfg.language_bias_mode == "custom"
        else cfg.spoken_languages
    )
    return {
        LANGUAGES[name].whisper for name in names if name in LANGUAGES
    }


def choose_language(probabilities, preferred: set[str]) -> str | None:
    """Reweight all candidates without restricting the model's language set."""
    if not probabilities or not preferred:
        return None
    return max(
        probabilities,
        key=lambda item: item[1] * (_PREFERRED_WEIGHT if item[0] in preferred else 1),
    )[0]


def sensevoice_override(
    logits: np.ndarray, language_tokens: dict[int, tuple[str, int]],
    preferred: set[str],
) -> int | None:
    """Return a conditioning slot only when the prior changes the LID winner.

    SenseVoice trains the first output frame as the language tag. Reweight
    that frame's logits, not the caption tokens; the caller must rerun with
    the chosen input slot so the text agrees with the language. Unknown or
    blank winners carry no usable language evidence and are left alone.
    """
    if not preferred or not language_tokens or logits.shape[0] == 0:
        return None
    frame = logits[0]
    original = int(frame.argmax())
    if original not in language_tokens:
        return None

    def score(token):
        code, _slot = language_tokens[token]
        return float(frame[token]) + (math.log(_PREFERRED_WEIGHT) if code in preferred else 0)

    winner = max(language_tokens, key=score)
    if winner == original or score(winner) <= score(original):
        return None
    return language_tokens[winner][1]
