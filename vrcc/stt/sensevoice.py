"""SenseVoice STT engine: the FunAudioLLM SenseVoice-Small CTC export.

Same duck-typed contract as :class:`vrcc.stt.engine.SttEngine` (load /
warm_up / unload / transcribe), turning mono float32 16 kHz audio into an
:class:`SttResult`. Unlike the onnx-asr backed models this one is driven
directly on onnxruntime, because onnx-asr has no class for it: the graph takes
pre-computed features plus language/inverse-text-normalization selector inputs
rather than raw audio. :mod:`vrcc.stt.fbank` supplies the front-end and every
constant it needs is read from the ONNX file's own metadata, so a re-exported
model cannot silently drift away from the code.

Device ``auto`` runs on CPU even when CUDA is available, for the same reason
:mod:`vrcc.stt.onnx_asr` does: the int8 graph is fast enough on CPU and a CUDA
session takes VRAM from VRChat. An explicit ``cuda`` config still builds CUDA
sessions.

The decoder is greedy CTC, so there is no beam to widen, but the logits it
runs are still a real distribution: softmax gives each frame's chosen-token
logprob, and the mean over non-blank frames (this model is CTC, so most
frames are blank even on clean speech) is checked against
``cfg.sensevoice_avg_logprob_gate`` -- see
:attr:`vrcc.core.config.SttConfig.sensevoice_avg_logprob_gate` for the
measurement backing that threshold. The blank-frame ratio was also
measured as a no-speech signal and rejected: on LibriSpeech test-clean with
6-speaker babble (50 utterances/condition, this machine, 2026-09-12) it was
0.72 at clean and only 0.77 at 0 dB babble, too close to gate on without also
catching clean speech, so ``no_speech_prob`` stays a neutral 0.0. What the
model does report, and Parakeet does not, is the language it heard: the
decode is prefixed with tags like
``<|ja|><|NEUTRAL|><|Speech|><|withitn|>``, so ``SttResult.language`` carries a
real detected language and a translating "auto" source stays correct. Zero Qt.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np

from vrcc.core.bus import EventBus
from vrcc.core.config import SttConfig
from vrcc.core.events import EngineStateChanged
from vrcc.core.hardware import resolve
from vrcc.core.languages import get
from vrcc.stt.engine import SttResult
from vrcc.stt.fbank import apply_cmvn, apply_lfr, fbank
from vrcc.stt.registry import WhisperSpec

logger = logging.getLogger("vrcc.stt.sensevoice")

_CPU_PROVIDERS = ("CPUExecutionProvider",)

_SAMPLE_RATE = 16000

# warm_up() transcribes this much silence (0.5s at 16kHz).
_WARM_UP_SAMPLES = 8000

# CTC blank. The export's vocab puts it at 0 ("<unk>"), matching what
# sherpa-onnx's own decoder assumes for this model family.
_BLANK_ID = 0

# The decode is prefixed with <|language|><|emotion|><|event|><|itn|> markers.
_TAG_RE = re.compile(r"<\|([^|]*)\|>")

# SenseVoice language tags -> the Whisper codes vrcc.core.languages uses.
# Cantonese has no VRCC display language of its own; "zh" is the closest the
# translator can be told, and its script is Chinese either way.
_TAG_TO_WHISPER = {
    "zh": "zh",
    "en": "en",
    "ja": "ja",
    "ko": "ko",
    "yue": "zh",
}


def scale_samples(samples: np.ndarray, normalize: bool) -> np.ndarray:
    """Put ``samples`` (mono float32 in [-1, 1]) into the amplitude scale the
    export's ``normalize_samples`` metadata asks for.

    ``normalize_samples=0`` -- what the published int8 export carries -- means
    the features were computed over int16-scaled audio, so the float input has
    to be scaled up by 32768 first. Getting this wrong does not raise; it just
    shifts every filterbank energy by a constant and quietly degrades the
    transcript, which is why it is read from metadata rather than assumed.
    """
    return samples if normalize else samples * 32768.0


class SenseVoiceEngine:
    """One loaded SenseVoice model.

    Single-caller contract (same as ``SttEngine``): driven by exactly one
    worker thread; ``transcribe()`` and ``unload()`` aren't thread-safe
    against each other.
    """

    def __init__(
        self,
        cfg: SttConfig,
        spec: WhisperSpec,
        model_dir: Path,
        bus: EventBus,
        session_factory=None,
    ) -> None:
        self._cfg = cfg
        self._spec = spec
        self._model_dir = Path(model_dir)
        self._bus = bus
        # Defaults to onnxruntime.InferenceSession, resolved lazily in load()
        # so the native import stays out of module import time. Tests inject
        # fakes.
        self._session_factory = session_factory

        self._session = None
        self._device: str | None = None
        self._tokens: list[str] = []
        self._neg_mean: np.ndarray | None = None
        self._inv_stddev: np.ndarray | None = None
        self._lfr_window = 7
        self._lfr_shift = 6
        self._normalize_samples = False
        self._text_norm_id = 14
        self._auto_language_id = 0
        # Whisper code -> the model's own language slot, from its metadata.
        self._language_ids: dict[str, int] = {}

    # -- lifecycle -----------------------------------------------------------

    def load(self) -> None:
        """Build the onnxruntime session, read the model's own front-end
        constants, and announce readiness.

        Publishes ``loading`` then ``ready`` (``detail="<device>:<quant>"``).
        A resolved ``auto`` device builds on CPU even when it resolves to CUDA
        (deliberate, so no ``fallback_cpu`` event); only an explicit ``cuda``
        config builds a CUDA session. A failed CUDA session falls back to CPU
        once, and a CUDA provider that fails to *initialize* is caught by
        inspecting the built session, because onnxruntime does not raise for
        that -- it quietly builds a CPU session instead. Any other error
        publishes ``failed`` and re-raises.
        """
        self._bus.publish(EngineStateChanged("stt", "loading"))
        try:
            model_path, vocab_path = self._model_files()
            device, index, _compute = resolve(
                self._cfg.device, self._cfg.device_index, self._cfg.compute_type
            )
            if device == "cuda" and self._cfg.device == "auto":
                # Deliberate, not a fallback: same reasoning as the onnx-asr
                # engine -- the int8 graph is fast enough on CPU and a CUDA
                # session takes VRAM from VRChat.
                device = "cpu"
            providers = self._providers(device, index)
            try:
                self._session = self._build_session(model_path, providers)
                self._device = device if providers != _CPU_PROVIDERS else "cpu"
            except Exception as exc:  # noqa: BLE001 -- CUDA-session failures vary by runtime
                if providers == _CPU_PROVIDERS:
                    raise
                logger.warning(
                    "%s CUDA session failed; falling back to CPU: %s",
                    self._spec.id, exc,
                )
                self._bus.publish(EngineStateChanged("stt", "fallback_cpu", str(exc)))
                self._session = self._build_session(model_path, _CPU_PROVIDERS)
                self._device = "cpu"
            else:
                if self._device == "cuda":
                    self._check_cuda_session()

            self._read_metadata()
            self._tokens = _read_vocab(vocab_path)

            self._bus.publish(
                EngineStateChanged(
                    "stt", "ready", f"{self._device}:{self._spec.quantization or 'fp32'}"
                )
            )
        except Exception as exc:
            self._session = None
            self._bus.publish(EngineStateChanged("stt", "failed", str(exc)))
            raise

    def warm_up(self) -> None:
        """Transcribe 0.5s of silence to prime the session/allocations (result
        discarded). Errors are not swallowed -- a failed warm-up means the
        engine is unhealthy. Re-checks the provider state afterward, because
        onnxruntime's CUDA->CPU fallback is a run-time event a build-time
        check cannot see.
        """
        self.transcribe(np.zeros(_WARM_UP_SAMPLES, dtype=np.float32))
        self._recheck_device_after_run()

    def unload(self) -> None:
        """Drop the session reference. Safe to call when not loaded."""
        self._session = None

    # -- transcription ---------------------------------------------------------

    def transcribe(
        self, samples: np.ndarray, detect_language: bool = False
    ) -> SttResult | None:
        """Transcribe ``samples`` (mono float32, 16 kHz) into an :class:`SttResult`.

        Returns ``None`` for audio too short to make one filterbank frame, for
        empty text, or when the mean non-blank-frame logprob falls below
        ``cfg.sensevoice_avg_logprob_gate`` (see
        :attr:`vrcc.core.config.SttConfig.sensevoice_avg_logprob_gate` for the
        measurement backing that threshold). ``language`` is the language the
        model reports, falling back to the configured source when the decode
        carries no usable tag. Raises ``RuntimeError`` if called before
        :meth:`load`.

        ``detect_language`` feeds the model its own ``lang_auto`` slot instead
        of the configured spoken language, for speech captured from the
        speakers: that is someone else's, and constraining it to the user's
        language turns a foreign sentence into confident nonsense. Every STT
        backend takes this argument, so one caller can serve any of them.
        """
        if self._session is None:
            raise RuntimeError(
                "SenseVoiceEngine.transcribe() called before a successful load(); "
                "call load() first."
            )

        features = self._features(np.ascontiguousarray(samples, dtype=np.float32))
        if features.shape[0] == 0:
            return None

        logits = self._session.run(
            ["logits"],
            {
                "x": features[None, :, :],
                "x_length": np.array([features.shape[0]], dtype=np.int32),
                "language": np.array(
                    [self._language_id(detect_language)], dtype=np.int32
                ),
                "text_norm": np.array([self._text_norm_id], dtype=np.int32),
            },
        )[0]

        logits0 = np.asarray(logits[0])
        ids = logits0.argmax(axis=-1)
        raw = self._decode(ids)
        text, language = self._split_tags(raw)
        if not text:
            return None

        # Non-empty text (checked above) decoded at least one non-blank token,
        # so None is unreachable; handled as a drop rather than an assert,
        # which -O strips and which would otherwise leave avg_logprob=None on
        # an SttResult field typed float.
        avg_logprob = _mean_nonblank_logprob(logits0, ids)
        if avg_logprob is None or avg_logprob < self._cfg.sensevoice_avg_logprob_gate:
            logger.debug(
                "%s gated by avg_logprob: %s < %.3f",
                self._spec.id,
                "None" if avg_logprob is None else f"{avg_logprob:.3f}",
                self._cfg.sensevoice_avg_logprob_gate,
            )
            return None

        # No no-speech signal (blank ratio was measured and rejected, see
        # module docstring): neutral value, always passes no_speech_gate.
        return SttResult(
            text=text, language=language, avg_logprob=avg_logprob, no_speech_prob=0.0,
        )

    # -- internals -------------------------------------------------------------

    def _model_files(self) -> tuple[Path, Path]:
        """The model and vocab paths, or a clear error naming the Models window."""
        if not self._model_dir.is_dir():
            raise RuntimeError(
                f"model files for {self._spec.id!r} not found in "
                f"{self._model_dir}; download the model in the Models window"
            )
        suffix = f".{self._spec.quantization}" if self._spec.quantization else ""
        model_path = self._model_dir / f"model{suffix}.onnx"
        vocab_path = self._model_dir / "tokens.txt"
        for path in (model_path, vocab_path):
            if not path.is_file():
                raise RuntimeError(
                    f"model files for {self._spec.id!r} are incomplete ({path.name} "
                    f"missing from {self._model_dir}); re-download the model in the "
                    f"Models window"
                )
        return model_path, vocab_path

    def _read_metadata(self) -> None:
        """Read the front-end constants the export carries.

        Every one of these is a silent-failure knob if it drifts (wrong mel
        scaling, wrong stacking, wrong prompt slot), so they come from the file
        rather than from constants here.
        """
        meta = self._session.get_modelmeta().custom_metadata_map
        self._neg_mean = np.fromstring(meta["neg_mean"], sep=",", dtype=np.float32)
        self._inv_stddev = np.fromstring(meta["inv_stddev"], sep=",", dtype=np.float32)
        self._lfr_window = int(meta["lfr_window_size"])
        self._lfr_shift = int(meta["lfr_window_shift"])
        self._normalize_samples = bool(int(meta["normalize_samples"]))
        self._text_norm_id = int(meta["with_itn"])
        self._auto_language_id = int(meta["lang_auto"])
        # First tag wins per Whisper code: "zh" maps from both lang_zh and
        # lang_yue, and Mandarin is the right default for a VRCC "Chinese".
        self._language_ids = {}
        for tag, whisper in _TAG_TO_WHISPER.items():
            key = f"lang_{tag}"
            if key in meta and whisper not in self._language_ids:
                self._language_ids[whisper] = int(meta[key])

    def _language_id(self, detect: bool = False) -> int:
        """The model's language slot for the *currently configured* source
        language, or its ``lang_auto`` slot when ``detect`` is set, the source
        is "auto", or the source is outside the model's set.

        Read per transcribe, not cached at load: the spoken-language combo
        writes straight to the live config without rebuilding the engine, so a
        cached slot would keep transcribing the previous language.
        """
        source = self._cfg.source_language
        if detect or source == "auto":
            return self._auto_language_id
        return self._language_ids.get(get(source).whisper, self._auto_language_id)

    def _features(self, samples: np.ndarray) -> np.ndarray:
        scaled = scale_samples(samples, self._normalize_samples)
        stacked = apply_lfr(fbank(scaled, _SAMPLE_RATE), self._lfr_window, self._lfr_shift)
        if stacked.shape[0] == 0:
            return stacked
        return apply_cmvn(stacked, self._neg_mean, self._inv_stddev)

    def _decode(self, ids: np.ndarray) -> str:
        """Greedy CTC collapse: drop repeats, drop blanks, join the pieces and
        turn SentencePiece's word marker back into a space."""
        pieces, previous = [], -1
        for token_id in ids.tolist():
            if token_id != previous and token_id != _BLANK_ID:
                pieces.append(self._tokens[token_id])
            previous = token_id
        return "".join(pieces).replace("▁", " ").strip()

    def _split_tags(self, raw: str) -> tuple[str, str]:
        """Split the decode into caption text and the detected language.

        The tags are metadata, never caption content, so all of them are
        stripped. The language falls back to the configured source (and "en"
        for an "auto" source) when no tag maps to a language VRCC knows, which
        keeps the translator's source honest rather than guessing.
        """
        language = None
        for tag in _TAG_RE.findall(raw):
            if tag in _TAG_TO_WHISPER:
                language = _TAG_TO_WHISPER[tag]
                break
        text = _TAG_RE.sub("", raw).strip()

        if language is None:
            source = self._cfg.source_language
            language = "en" if source == "auto" else get(source).whisper
        return text, language

    def _check_cuda_session(self) -> None:
        """Downgrade ``_device`` to cpu when the built session did not actually
        get the CUDA provider. onnxruntime doesn't raise when a requested
        provider fails to initialize (e.g. the onnxruntime-gpu build wants CUDA
        runtime DLLs the install doesn't ship) -- it logs and builds a CPU
        session -- so the requested device can't be trusted."""
        try:
            providers = set(self._session.get_providers())
        except Exception:  # noqa: BLE001 -- session-like fakes may misbehave
            return
        if not providers or "CUDAExecutionProvider" in providers:
            return
        detail = "onnxruntime built a CPU-only session (CUDA provider failed to initialize)"
        logger.warning("%s: %s", self._spec.id, detail)
        self._bus.publish(EngineStateChanged("stt", "fallback_cpu", detail))
        self._device = "cpu"

    def _recheck_device_after_run(self) -> None:
        """Downgrade ``_device`` to cpu when the session dropped CUDA during
        its first run. onnxruntime performs the CUDA->CPU fallback (a missing
        cuDNN kernel, for instance) at run time, not at build, and only then
        removes CUDAExecutionProvider from get_providers() -- the build-time
        ``_check_cuda_session`` runs before any op has executed, so it cannot
        see this. Publishing the corrected ``ready`` after ``fallback_cpu``
        restores the event invariant (fallback_cpu always immediately
        precedes a ready) and stops ``_device``/the UI from claiming cuda for
        a session that actually ran on CPU."""
        if self._device != "cuda" or self._session is None:
            return
        try:
            providers = set(self._session.get_providers())
        except Exception:  # noqa: BLE001 -- session-like fakes may misbehave
            return
        if not providers or "CUDAExecutionProvider" in providers:
            return
        detail = "onnxruntime fell back to CPU at first run (CUDA kernel unavailable)"
        logger.warning("%s: %s", self._spec.id, detail)
        self._device = "cpu"
        self._bus.publish(EngineStateChanged("stt", "fallback_cpu", detail))
        self._bus.publish(
            EngineStateChanged("stt", "ready", f"{self._device}:{self._spec.quantization or 'fp32'}")
        )

    def _providers(self, device: str, index: int) -> tuple:
        """onnxruntime providers for the resolved device: CUDA (pinned to the
        configured card) when requested *and* available in this onnxruntime
        build, else CPU."""
        if device == "cuda":
            import onnxruntime

            if "CUDAExecutionProvider" in onnxruntime.get_available_providers():
                return (
                    ("CUDAExecutionProvider", {"device_id": index}),
                    "CPUExecutionProvider",
                )
            logger.info(
                "CUDA requested but this onnxruntime build has no "
                "CUDAExecutionProvider; %s runs on CPU", self._spec.id,
            )
        return _CPU_PROVIDERS

    def _build_session(self, model_path: Path, providers: tuple):
        """Construct the onnxruntime session from the downloaded model.

        ``cpu_threads`` is honoured here rather than left to onnxruntime's
        default because int8 graphs can decode differently at different thread
        counts, so the setting has to be reachable when a machine needs it
        pinned.
        """
        import onnxruntime

        options = onnxruntime.SessionOptions()
        if self._cfg.cpu_threads > 0:
            options.intra_op_num_threads = self._cfg.cpu_threads
        factory = self._session_factory or onnxruntime.InferenceSession
        return factory(str(model_path), sess_options=options, providers=list(providers))


def _mean_nonblank_logprob(logits: np.ndarray, ids: np.ndarray) -> float | None:
    """Mean softmax logprob of each frame's chosen token, over frames whose
    argmax is not the CTC blank. Averaging over every frame instead would be
    dominated by blank frames (measured at 72-77% of frames regardless of
    noise, see module docstring), which barely moves between clean and
    degraded audio. ``None`` when every frame decoded blank."""
    max_logit = logits.max(axis=-1)
    logsumexp = np.log(np.exp(logits - max_logit[:, None]).sum(axis=-1)) + max_logit
    token_logprob = max_logit - logsumexp
    non_blank = ids != _BLANK_ID
    if not non_blank.any():
        return None
    return float(token_logprob[non_blank].mean())


def _read_vocab(path: Path) -> list[str]:
    """Read a sherpa-style ``tokens.txt`` ("<piece> <id>" per line) into a list
    indexed by id. ``rsplit`` on the last space so a piece that contains one
    still parses."""
    tokens: dict[int, str] = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if not line:
                continue
            piece, _, index = line.rpartition(" ")
            tokens[int(index)] = piece
    if not tokens:
        raise RuntimeError(f"{path} contains no tokens")
    return [tokens.get(i, "") for i in range(max(tokens) + 1)]
