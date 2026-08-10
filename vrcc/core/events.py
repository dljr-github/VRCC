"""Event dataclasses passed over the :class:`vrcc.core.bus.EventBus`.

Plain frozen dataclasses, no behavior; kept import-cheap. Utterance audio never
rides the bus (the pipeline consumes the segmenter's ``Seg*`` types directly).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MicLevel:
    rms: float
    vad_prob: float


@dataclass(frozen=True)
class SpeechStarted:
    utterance_id: int


@dataclass(frozen=True)
class PhraseRecognized:
    utterance_id: int
    text: str
    language: str
    avg_logprob: float
    no_speech_prob: float


@dataclass(frozen=True)
class PhraseTranslated:
    utterance_id: int
    original: str
    source_lang: str
    translations: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class ChatboxSent:
    text: str
    utterance_id: int
    # True when the message was clipped to VRChat's 144-char chatbox limit.
    truncated: bool = False


@dataclass(frozen=True)
class VrchatDetected:
    """Whether a VRChat client with OSC is discoverable (OSCQuery/mDNS) -- a
    proxy for "is the chatbox reachable?", since OSC is fire-and-forget UDP with
    no ack."""

    detected: bool


@dataclass(frozen=True)
class TypingStateChanged:
    typing: bool


@dataclass(frozen=True)
class MuteChanged:
    muted: "bool | None"


@dataclass(frozen=True)
class DownloadProgress:
    model_id: str
    downloaded: int
    total: int
    done: bool = False


@dataclass(frozen=True)
class EngineStateChanged:
    # engine: "stt" | "mt"
    # state: "loading" | "ready" | "failed" | "fallback_cpu"
    engine: str
    state: str
    detail: str = ""


@dataclass(frozen=True)
class AppError:
    code: str
    message: str
    detail: str = ""


@dataclass(frozen=True)
class UpdateCheckResult:
    available: bool
    latest: str = ""
    url: str = ""
    error: str = ""


@dataclass(frozen=True)
class HeardPhrase:
    """Something the SPEAKERS played, transcribed for the user to read.

    Deliberately not a PhraseRecognized: that one carries an utterance_id the
    chatbox path uses to tie a caption to a send, and this one is never sent.
    Keeping it a separate type means a handler cannot treat heard speech as the
    user's own by accident.

    ``translations`` is ``[(language_display, text), ...]``, rendered INTO the
    languages the user reads rather than the outbound targets.
    """

    text: str
    language: str
    translations: list[tuple[str, str]]
