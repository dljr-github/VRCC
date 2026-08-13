"""Tests for :mod:`vrcc.stt.sensevoice` with a fake onnxruntime session (no
240 MB model load): load events, provider selection + CPU fallback, the
metadata-driven front-end wiring, tag stripping, detected-language reporting
and the create_stt_engine dispatch.

The real model is exercised in tests/integration/test_sensevoice_reference.py.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from vrcc.core.bus import EventBus
from vrcc.core.config import SttConfig
from vrcc.core.events import EngineStateChanged
from vrcc.stt import create_stt_engine
from vrcc.stt.registry import WHISPER_MODELS
from vrcc.stt.sensevoice import SenseVoiceEngine

SENSEVOICE_ID = "sense-voice-small"
SENSEVOICE = WHISPER_MODELS[SENSEVOICE_ID]

# id -> piece. 0 is the CTC blank; the four tag slots mirror the real export's
# <|language|><|emotion|><|event|><|itn|> prefix.
_VOCAB = [
    "<unk>", "<|ja|>", "<|en|>", "<|NEUTRAL|>", "<|Speech|>", "<|withitn|>",
    "▁hello", "▁world", "<|zz|>",
]

_META = {
    "neg_mean": ",".join(["0.0"] * 560),
    "inv_stddev": ",".join(["1.0"] * 560),
    "lfr_window_size": "7",
    "lfr_window_shift": "6",
    "normalize_samples": "0",
    "with_itn": "14",
    "lang_auto": "0",
    "lang_zh": "3",
    "lang_en": "4",
    "lang_yue": "7",
    "lang_ja": "11",
    "lang_ko": "12",
}


class _FakeSession:
    """Quacks like onnxruntime.InferenceSession over a scripted token path."""

    def __init__(
        self, token_ids, providers=("CPUExecutionProvider",), meta=None,
        providers_after_run=None,
    ):
        self._token_ids = list(token_ids)
        self._providers = list(providers)
        self._meta = dict(_META if meta is None else meta)
        self._providers_after_run = providers_after_run
        self.runs: list[dict] = []

    def run(self, outputs, feeds):
        self.runs.append(feeds)
        if self._providers_after_run is not None:
            # Mirrors onnxruntime: the CUDA->CPU fallback is a run-time event,
            # so get_providers() only reflects it after the first run.
            self._providers = list(self._providers_after_run)
        vocab = len(_VOCAB)
        logits = np.zeros((1, len(self._token_ids), vocab), dtype=np.float32)
        for step, token_id in enumerate(self._token_ids):
            logits[0, step, token_id] = 1.0
        return [logits]

    def get_modelmeta(self):
        return SimpleNamespace(custom_metadata_map=self._meta)

    def get_providers(self):
        return list(self._providers)


class _RecordingFactory:
    """Fake ``onnxruntime.InferenceSession`` recording every construction."""

    def __init__(
        self, token_ids=(1, 3, 4, 5, 6, 7), fail_at=(), session_providers=None, meta=None,
        providers_after_run=None,
    ):
        self.calls: list[SimpleNamespace] = []
        self.built: list[_FakeSession] = []
        self._token_ids = token_ids
        self._fail_at = set(fail_at)
        self._session_providers = session_providers
        self._meta = meta
        self._providers_after_run = providers_after_run

    def __call__(self, path, sess_options=None, providers=None):
        idx = len(self.calls)
        self.calls.append(
            SimpleNamespace(path=path, options=sess_options, providers=providers)
        )
        if idx in self._fail_at:
            raise RuntimeError("CUDA provider unavailable")
        reported = self._session_providers or providers or ["CPUExecutionProvider"]
        session = _FakeSession(
            self._token_ids, providers=_names(reported), meta=self._meta,
            providers_after_run=self._providers_after_run,
        )
        self.built.append(session)
        return session


def _names(providers) -> list[str]:
    """onnxruntime reports plain provider names even when built from
    (name, options) tuples."""
    return [p[0] if isinstance(p, tuple) else p for p in providers]


def _cfg(**over) -> SttConfig:
    base = dict(model=SENSEVOICE_ID, device="cpu", device_index=0, compute_type="int8")
    base.update(over)
    return SttConfig(**base)


def _collect(bus: EventBus) -> list[EngineStateChanged]:
    events: list[EngineStateChanged] = []
    bus.subscribe(EngineStateChanged, events.append)
    return events


def _speech(seconds: float = 1.0) -> np.ndarray:
    t = np.arange(int(16000 * seconds)) / 16000.0
    return (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)


@pytest.fixture()
def model_dir(tmp_path: Path) -> Path:
    d = tmp_path / "models" / "whisper" / SENSEVOICE_ID
    d.mkdir(parents=True)
    (d / "model.int8.onnx").write_bytes(b"")
    (d / "tokens.txt").write_text(
        "\n".join(f"{piece} {i}" for i, piece in enumerate(_VOCAB)) + "\n",
        encoding="utf-8",
    )
    return d


def _engine(model_dir, factory, **over) -> SenseVoiceEngine:
    bus = over.pop("bus", None) or EventBus()
    return SenseVoiceEngine(
        _cfg(**over), SENSEVOICE, model_dir, bus, session_factory=factory
    )


# --------------------------------------------------------------------------
# load(): session args + event sequence
# --------------------------------------------------------------------------

def test_load_builds_the_quantized_model_on_cpu(model_dir):
    bus = EventBus()
    events = _collect(bus)
    factory = _RecordingFactory()
    eng = _engine(model_dir, factory, bus=bus)

    eng.load()

    assert len(factory.calls) == 1
    assert Path(factory.calls[0].path) == model_dir / "model.int8.onnx"
    assert factory.calls[0].providers == ["CPUExecutionProvider"]
    assert [(e.engine, e.state) for e in events] == [("stt", "loading"), ("stt", "ready")]
    assert events[-1].detail == "cpu:int8"


def test_load_missing_model_dir_publishes_failed_and_raises(tmp_path):
    bus = EventBus()
    events = _collect(bus)
    factory = _RecordingFactory()
    eng = SenseVoiceEngine(
        _cfg(), SENSEVOICE, tmp_path / "nope", bus, session_factory=factory
    )

    with pytest.raises(RuntimeError, match="Models window"):
        eng.load()

    assert factory.calls == []
    assert [(e.engine, e.state) for e in events] == [("stt", "loading"), ("stt", "failed")]


def test_load_incomplete_download_names_the_missing_file(model_dir):
    (model_dir / "tokens.txt").unlink()
    eng = _engine(model_dir, _RecordingFactory())

    with pytest.raises(RuntimeError, match="tokens.txt"):
        eng.load()


def test_auto_device_builds_on_cpu_even_when_cuda_resolves(model_dir, monkeypatch):
    # Deliberate, and it must not look like a failure: no fallback_cpu event.
    monkeypatch.setattr(
        "vrcc.stt.sensevoice.resolve", lambda *a, **k: ("cuda", 0, "int8")
    )
    bus = EventBus()
    events = _collect(bus)
    factory = _RecordingFactory()
    eng = _engine(model_dir, factory, bus=bus, device="auto")

    eng.load()

    assert factory.calls[0].providers == ["CPUExecutionProvider"]
    assert [e.state for e in events] == ["loading", "ready"]


def test_explicit_cuda_requests_the_cuda_provider(model_dir, monkeypatch):
    monkeypatch.setattr(
        "vrcc.stt.sensevoice.resolve", lambda *a, **k: ("cuda", 1, "int8")
    )
    monkeypatch.setattr(
        "onnxruntime.get_available_providers",
        lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    factory = _RecordingFactory(session_providers=["CUDAExecutionProvider"])
    eng = _engine(model_dir, factory, device="cuda", device_index=1)

    eng.load()

    assert factory.calls[0].providers == [
        ("CUDAExecutionProvider", {"device_id": 1}),
        "CPUExecutionProvider",
    ]


def test_failed_cuda_session_falls_back_to_cpu_once(model_dir, monkeypatch):
    monkeypatch.setattr(
        "vrcc.stt.sensevoice.resolve", lambda *a, **k: ("cuda", 0, "int8")
    )
    monkeypatch.setattr(
        "onnxruntime.get_available_providers",
        lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    bus = EventBus()
    events = _collect(bus)
    factory = _RecordingFactory(fail_at=(0,))
    eng = _engine(model_dir, factory, bus=bus, device="cuda")

    eng.load()

    assert len(factory.calls) == 2
    assert factory.calls[1].providers == ["CPUExecutionProvider"]
    assert [e.state for e in events] == ["loading", "fallback_cpu", "ready"]
    assert events[-1].detail == "cpu:int8"


def test_cuda_provider_that_silently_builds_cpu_is_reported(model_dir, monkeypatch):
    # onnxruntime does not raise when a requested provider fails to
    # initialize; it logs and builds a CPU session, so the built session has
    # to be inspected rather than the request trusted.
    monkeypatch.setattr(
        "vrcc.stt.sensevoice.resolve", lambda *a, **k: ("cuda", 0, "int8")
    )
    monkeypatch.setattr(
        "onnxruntime.get_available_providers",
        lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    bus = EventBus()
    events = _collect(bus)
    factory = _RecordingFactory(session_providers=["CPUExecutionProvider"])
    eng = _engine(model_dir, factory, bus=bus, device="cuda")

    eng.load()

    assert [e.state for e in events] == ["loading", "fallback_cpu", "ready"]
    assert events[-1].detail == "cpu:int8"


def test_cpu_threads_config_pins_intra_op_threads(model_dir):
    factory = _RecordingFactory()
    _engine(model_dir, factory, cpu_threads=2).load()

    assert factory.calls[0].options.intra_op_num_threads == 2


def test_cpu_threads_zero_leaves_onnxruntimes_own_default(model_dir):
    factory = _RecordingFactory()
    _engine(model_dir, factory, cpu_threads=0).load()

    assert factory.calls[0].options.intra_op_num_threads == 0


# --------------------------------------------------------------------------
# transcribe()
# --------------------------------------------------------------------------

def test_transcribe_before_load_raises(model_dir):
    eng = _engine(model_dir, _RecordingFactory())
    with pytest.raises(RuntimeError, match="load\\(\\) first"):
        eng.transcribe(_speech())


def test_transcribe_strips_tags_and_reports_the_detected_language(model_dir):
    # <|ja|><|NEUTRAL|><|Speech|><|withitn|>▁hello▁world
    factory = _RecordingFactory(token_ids=(1, 3, 4, 5, 6, 7))
    eng = _engine(model_dir, factory, source_language="auto")
    eng.load()

    result = eng.transcribe(_speech())

    assert result.text == "hello world"
    # The one thing parakeet cannot do: tell the translator what it heard.
    assert result.language == "ja"
    assert result.avg_logprob == 0.0
    assert result.no_speech_prob == 0.0


def test_transcribe_collapses_ctc_repeats_and_blanks(model_dir):
    # hello, hello, blank, hello -> the repeat collapses, the blank separates.
    factory = _RecordingFactory(token_ids=(6, 6, 0, 6, 7))
    eng = _engine(model_dir, factory)
    eng.load()

    assert eng.transcribe(_speech()).text == "hello hello world"


def test_transcribe_unknown_language_tag_falls_back_to_the_configured_source(model_dir):
    factory = _RecordingFactory(token_ids=(8, 6))  # <|zz|>▁hello
    eng = _engine(model_dir, factory, source_language="Korean")
    eng.load()

    result = eng.transcribe(_speech())

    assert result.text == "hello"
    assert result.language == "ko"


def test_transcribe_untagged_auto_source_falls_back_to_english(model_dir):
    factory = _RecordingFactory(token_ids=(6,))
    eng = _engine(model_dir, factory, source_language="auto")
    eng.load()

    assert eng.transcribe(_speech()).language == "en"


def test_transcribe_returns_none_for_audio_shorter_than_one_frame(model_dir):
    factory = _RecordingFactory()
    eng = _engine(model_dir, factory)
    eng.load()

    assert eng.transcribe(np.zeros(200, dtype=np.float32)) is None
    assert factory.built[0].runs == []  # never reached the session


def test_transcribe_returns_none_for_empty_text(model_dir):
    factory = _RecordingFactory(token_ids=(0, 0, 0))  # all blanks
    eng = _engine(model_dir, factory)
    eng.load()

    assert eng.transcribe(_speech()) is None


def test_transcribe_returns_none_when_only_tags_decode(model_dir):
    factory = _RecordingFactory(token_ids=(1, 3, 4, 5))
    eng = _engine(model_dir, factory)
    eng.load()

    assert eng.transcribe(_speech()) is None


def test_transcribe_feeds_the_expected_input_contract(model_dir):
    factory = _RecordingFactory()
    eng = _engine(model_dir, factory, source_language="auto")
    eng.load()
    eng.transcribe(_speech())

    feeds = factory.built[0].runs[0]
    assert set(feeds) == {"x", "x_length", "language", "text_norm"}
    assert feeds["x"].ndim == 3 and feeds["x"].shape[2] == 560
    assert feeds["x"].dtype == np.float32
    assert feeds["x_length"].dtype == np.int32
    assert feeds["x_length"][0] == feeds["x"].shape[1]
    assert feeds["text_norm"][0] == 14  # with_itn


# --------------------------------------------------------------------------
# metadata-driven language slot
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("source", "expected"),
    [("auto", 0), ("English", 4), ("Japanese", 11), ("Korean", 12),
     ("Chinese Simplified", 3)],
)
def test_language_slot_comes_from_the_models_own_metadata(model_dir, source, expected):
    factory = _RecordingFactory()
    eng = _engine(model_dir, factory, source_language=source)
    eng.load()
    eng.transcribe(_speech())

    assert factory.built[0].runs[0]["language"][0] == expected


def test_language_outside_the_models_set_uses_auto(model_dir):
    # French has no lang_* slot in this export; auto beats guessing.
    factory = _RecordingFactory()
    eng = _engine(model_dir, factory, source_language="French")
    eng.load()
    eng.transcribe(_speech())

    assert factory.built[0].runs[0]["language"][0] == 0


def test_language_slot_follows_a_live_source_language_change(model_dir):
    """The spoken-language combo writes straight to the live config without
    rebuilding the engine, so the slot must be read per transcribe."""
    factory = _RecordingFactory()
    eng = _engine(model_dir, factory, source_language="Japanese")
    eng.load()
    eng.transcribe(_speech())

    eng._cfg.source_language = "Korean"
    eng.transcribe(_speech())

    slots = [run["language"][0] for run in factory.built[0].runs]
    assert slots == [11, 12]


def test_cantonese_slot_does_not_displace_mandarin_for_chinese(model_dir):
    """lang_zh and lang_yue both map to the Whisper code "zh"; a VRCC
    "Chinese" source must pin Mandarin, not Cantonese."""
    factory = _RecordingFactory()
    eng = _engine(model_dir, factory, source_language="Chinese Traditional")
    eng.load()
    eng.transcribe(_speech())

    assert factory.built[0].runs[0]["language"][0] == 3  # lang_zh, not lang_yue (7)


def test_normalize_samples_metadata_drives_the_amplitude_scale(model_dir):
    """normalize_samples=1 means the export wants [-1, 1] audio and =0 means
    int16 scale. Getting it wrong does not raise, it silently shifts every
    filterbank energy, so the flag has to actually reach the front-end."""
    features = {}
    for flag in ("0", "1"):
        factory = _RecordingFactory(meta=dict(_META, normalize_samples=flag))
        engine = _engine(model_dir, factory)
        engine.load()
        engine.transcribe(_speech())
        features[flag] = factory.built[0].runs[0]["x"]

    assert not np.allclose(features["0"], features["1"])


# --------------------------------------------------------------------------
# lifecycle + dispatch
# --------------------------------------------------------------------------

def test_warm_up_runs_a_transcribe_and_unload_drops_the_session(model_dir):
    factory = _RecordingFactory()
    eng = _engine(model_dir, factory)
    eng.load()

    eng.warm_up()
    assert len(factory.built[0].runs) == 1

    eng.unload()
    with pytest.raises(RuntimeError, match="load\\(\\) first"):
        eng.transcribe(_speech())


def test_unload_without_load_is_a_no_op(model_dir):
    _engine(model_dir, _RecordingFactory()).unload()


def test_create_stt_engine_dispatches_to_sensevoice(tmp_path):
    engine = create_stt_engine(_cfg(), tmp_path, EventBus())
    assert isinstance(engine, SenseVoiceEngine)


def test_create_stt_engine_honours_an_explicit_model_id(tmp_path):
    cfg = _cfg(model="small")
    engine = create_stt_engine(cfg, tmp_path, EventBus(), model_id=SENSEVOICE_ID)
    assert isinstance(engine, SenseVoiceEngine)


def test_every_voice_backend_accepts_detect_language():
    """HeardStream calls transcribe(detect_language=True) on whichever engine
    the user picked, so the argument is part of the shared contract, not one
    backend's extra. SenseVoice shipped without it: every heard utterance
    raised TypeError into a handler that logged and moved on, and the feature
    produced nothing at all for anyone on that model.
    """
    import inspect

    from vrcc.stt.engine import SttEngine
    from vrcc.stt.onnx_asr import OnnxAsrEngine
    from vrcc.stt.sensevoice import SenseVoiceEngine

    for cls in (SttEngine, OnnxAsrEngine, SenseVoiceEngine):
        params = inspect.signature(cls.transcribe).parameters
        assert "detect_language" in params, cls.__name__
        assert params["detect_language"].default is False, cls.__name__
