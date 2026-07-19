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
class PhrasePartial:
    """A tentative, in-progress transcription of the active utterance. Original
    text only; no language/logprob/no-speech fields, since it never firms up
    into a stored result on its own."""

    utterance_id: int
    text: str


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
