"""Tests for :mod:`vrcc.translate.engine` with a fake translator factory
(no CTranslate2): ctor device/compute/threads, one translate_batch per N
targets with per-family layout, kwargs merge, and the CUDA-unusable CPU
fallback.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from vrcc.core.bus import EventBus
from vrcc.core.config import TranslateConfig
from vrcc.core.events import EngineStateChanged
from vrcc.core.languages import get
from vrcc.translate.engine import TranslateEngine
from vrcc.translate.registry import MtModelSpec

# Same toy corpus used by test_mt_tokenizers: lowercase words, enough distinct
# substrings to train a tiny unigram model with a stable "hello world" piece.
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
    spm.SentencePieceTrainer.train(
        sentence_iterator=iter(_SENTENCES),
        model_prefix=str(prefix),
        vocab_size=64,
        hard_vocab_limit=False,
    )
    model = Path(str(prefix) + ".model")
    assert model.exists()
    return model


@pytest.fixture
def model_dir(tmp_path: Path, toy_spm_path: Path) -> Path:
    """A model directory holding the toy tokenizer under ``spec.spm_file``."""
    d = tmp_path / "mt" / "test-model"
    d.mkdir(parents=True)
    (d / "toy.model").write_bytes(toy_spm_path.read_bytes())
    return d


def _spec(family: str) -> MtModelSpec:
    return MtModelSpec(
        id=f"test-{family}",
        repo="test/repo",
        family=family,
        size_mb=1,
        license="TEST",
        spm_file="toy.model",
        prefix_side="source" if family == "madlad" else "target",
    )


def _cfg(**over) -> TranslateConfig:
    base = dict(device="cpu", device_index=0, compute_type="int8")
    base.update(over)
    return TranslateConfig(**base)


def _collect(bus: EventBus) -> list[EngineStateChanged]:
    events: list[EngineStateChanged] = []
    bus.subscribe(EngineStateChanged, events.append)
    return events


_OOM_TEXT = "CUDA failed with error out of memory (CUBLAS_STATUS_ALLOC_FAILED)"
# Verbatim CTranslate2 dynamic-loader error from a CPU-only install driving a
# visible GPU: cudart is statically linked, so the device enumerates, and the
# load dies at model build or the first CUDA op instead.
_MISSING_LIBRARY_TEXT = "Library cublas64_12.dll is not found or cannot be loaded"

# Both shapes of CUDA-unusable RuntimeError must take the one-shot CPU fallback.
_CUDA_UNUSABLE_TEXTS = [_OOM_TEXT, _MISSING_LIBRARY_TEXT]


class _FakeTranslator:
    """Records translate_batch calls; echoes each source list as its hypothesis."""

    def __init__(self, fail_on_translate: bool = False, error_text: str = _OOM_TEXT) -> None:
        self.fail_on_translate = fail_on_translate
        self.error_text = error_text
        self.batch_calls: list[SimpleNamespace] = []

    def translate_batch(self, source, target_prefix=None, **kwargs):
        self.batch_calls.append(
            SimpleNamespace(source=source, target_prefix=target_prefix, kwargs=kwargs)
        )
        if self.fail_on_translate:
            raise RuntimeError(self.error_text)
        return [SimpleNamespace(hypotheses=[list(s)], scores=[0.0]) for s in source]


class _RecordingFactory:
    """Fake ``ctranslate2.Translator`` factory recording every ctor call."""

    def __init__(self, ctor_fail_at=(), translate_fail_at=(), error_text=_OOM_TEXT) -> None:
        self.calls: list[SimpleNamespace] = []
        self.built: list[_FakeTranslator] = []
        self._ctor_fail_at = set(ctor_fail_at)
        self._translate_fail_at = set(translate_fail_at)
        self._error_text = error_text

    def __call__(
        self,
        model_path,
        *,
        device,
        device_index,
        compute_type,
        inter_threads,
        intra_threads,
        max_queued_batches,
    ):
        idx = len(self.calls)
        self.calls.append(
            SimpleNamespace(
                model_path=model_path,
                device=device,
                device_index=device_index,
                compute_type=compute_type,
                inter_threads=inter_threads,
                intra_threads=intra_threads,
                max_queued_batches=max_queued_batches,
            )
        )
        if idx in self._ctor_fail_at:
            raise RuntimeError(self._error_text)
        t = _FakeTranslator(
            fail_on_translate=idx in self._translate_fail_at,
            error_text=self._error_text,
        )
        self.built.append(t)
        return t


# --------------------------------------------------------------------------
# load(): ctor kwargs + event sequence
# --------------------------------------------------------------------------

def test_load_records_ctor_kwargs_and_publishes_loading_then_ready(model_dir: Path):
    bus = EventBus()
    events = _collect(bus)
    factory = _RecordingFactory()
    cfg = _cfg(inter_threads=3, intra_threads=5)
    eng = TranslateEngine(_spec("nllb"), model_dir, cfg, bus, translator_factory=factory)

    eng.load()

    assert len(factory.calls) == 1
    call = factory.calls[0]
    assert call.model_path == str(model_dir)
    assert call.device == "cpu"
    assert call.device_index == 0
    assert call.compute_type == "int8"
    assert call.inter_threads == 3
    assert call.intra_threads == 5
    assert call.max_queued_batches == 0  # config default: CT2 automatic
    assert [(e.engine, e.state) for e in events] == [("mt", "loading"), ("mt", "ready")]
    assert events[-1].detail == "cpu:int8"


def test_load_passes_configured_max_queued_batches(model_dir: Path):
    factory = _RecordingFactory()
    cfg = _cfg(max_queued_batches=8)
    eng = TranslateEngine(_spec("nllb"), model_dir, cfg, EventBus(), translator_factory=factory)

    eng.load()

    assert factory.calls[0].max_queued_batches == 8


def test_load_passes_config_device_settings_to_resolve(model_dir: Path, monkeypatch):
    seen = []

    def fake_resolve(device_cfg, device_index, compute_cfg):
        seen.append((device_cfg, device_index, compute_cfg))
        return ("cpu", 0, "int8")

    monkeypatch.setattr("vrcc.translate.engine.resolve", fake_resolve)
    factory = _RecordingFactory()
    cfg = _cfg(device="cuda", device_index=2, compute_type="auto")
    eng = TranslateEngine(_spec("nllb"), model_dir, cfg, bus=EventBus(), translator_factory=factory)

    eng.load()

    assert seen == [("cuda", 2, "auto")]


# --------------------------------------------------------------------------
# translate(): one batch, per-family layout
# --------------------------------------------------------------------------

def test_nllb_single_batch_two_targets_with_per_entry_prefixes(model_dir: Path):
    factory = _RecordingFactory()
    cfg = _cfg(targets=["Japanese", "Korean"])
    eng = TranslateEngine(_spec("nllb"), model_dir, cfg, EventBus(), translator_factory=factory)
    eng.load()

    out = eng.translate("hello world", get("English"), [get("Japanese"), get("Korean")])

    translator = factory.built[0]
    assert len(translator.batch_calls) == 1  # ONE call for two targets
    call = translator.batch_calls[0]
    assert len(call.source) == 2
    assert call.source[0] == call.source[1]  # identical source tokens
    assert call.source[0][0] == "eng_Latn"  # source FLORES head
    assert call.source[0][-1] == "</s>"
    assert call.target_prefix == [["jpn_Jpan"], ["kor_Hang"]]
    assert [name for name, _ in out] == ["Japanese", "Korean"]
    assert all(text == "hello world" for _, text in out)


def test_madlad_per_target_source_encodings_and_no_target_prefix(model_dir: Path):
    factory = _RecordingFactory()
    cfg = _cfg(targets=["Japanese", "Korean"])
    eng = TranslateEngine(_spec("madlad"), model_dir, cfg, EventBus(), translator_factory=factory)
    eng.load()

    out = eng.translate("hello world", get("English"), [get("Japanese"), get("Korean")])

    translator = factory.built[0]
    assert len(translator.batch_calls) == 1
    call = translator.batch_calls[0]
    assert len(call.source) == 2
    # madlad injects the target on the SOURCE side, so the two entries differ.
    assert call.source[0][0] == "<2ja>"
    assert call.source[1][0] == "<2ko>"
    assert call.source[0] != call.source[1]
    assert call.target_prefix is None
    assert [name for name, _ in out] == ["Japanese", "Korean"]


def test_translate_batch_kwargs_merge_extra_overrides_beam_size(model_dir: Path):
    factory = _RecordingFactory()
    cfg = _cfg(beam_size=4, extra_translate_kwargs={"beam_size": 9, "no_repeat_ngram_size": 3})
    eng = TranslateEngine(_spec("nllb"), model_dir, cfg, EventBus(), translator_factory=factory)
    eng.load()

    eng.translate("hello world", get("English"), [get("Japanese")])

    kwargs = factory.built[0].batch_calls[0].kwargs
    assert kwargs["return_scores"] is False
    assert kwargs["beam_size"] == 9  # user override wins
    assert kwargs["no_repeat_ngram_size"] == 3


# --------------------------------------------------------------------------
# CUDA-unusable fallback (VRAM OOM or missing runtime library)
# --------------------------------------------------------------------------

@pytest.mark.parametrize("error_text", _CUDA_UNUSABLE_TEXTS)
def test_cuda_unusable_in_ctor_rebuilds_on_cpu_int8_with_fallback_event(
    model_dir: Path, monkeypatch, error_text
):
    monkeypatch.setattr(
        "vrcc.translate.engine.resolve",
        lambda *a: ("cuda", 0, "int8_float16"),
    )
    bus = EventBus()
    events = _collect(bus)
    factory = _RecordingFactory(ctor_fail_at=[0], error_text=error_text)
    cfg = _cfg(device="cuda", compute_type="int8_float16", inter_threads=3, intra_threads=5)
    eng = TranslateEngine(_spec("nllb"), model_dir, cfg, bus, translator_factory=factory)

    eng.load()

    assert len(factory.calls) == 2
    assert factory.calls[0].device == "cuda"
    second = factory.calls[1]
    assert (second.device, second.device_index, second.compute_type) == ("cpu", 0, "int8")
    assert second.inter_threads == 3 and second.intra_threads == 5  # threads preserved
    assert [(e.engine, e.state) for e in events] == [
        ("mt", "loading"),
        ("mt", "fallback_cpu"),
        ("mt", "ready"),
    ]
    assert events[-1].detail == "cpu:int8"
    # engine is usable afterwards on the cpu translator
    out = eng.translate("hello world", get("English"), [get("Japanese")])
    assert out == [("Japanese", "hello world")]


@pytest.mark.parametrize("error_text", _CUDA_UNUSABLE_TEXTS)
def test_cuda_unusable_in_translate_rebuilds_cpu_and_retries_once(
    model_dir: Path, monkeypatch, error_text
):
    monkeypatch.setattr(
        "vrcc.translate.engine.resolve",
        lambda *a: ("cuda", 0, "int8_float16"),
    )
    bus = EventBus()
    events = _collect(bus)
    factory = _RecordingFactory(translate_fail_at=[0], error_text=error_text)
    cfg = _cfg(device="cuda", compute_type="int8_float16")
    eng = TranslateEngine(_spec("nllb"), model_dir, cfg, bus, translator_factory=factory)
    eng.load()
    assert factory.calls[0].device == "cuda"
    events.clear()  # drop the load events; focus on the translate path

    out = eng.translate("hello world", get("English"), [get("Japanese")])

    assert len(factory.calls) == 2
    assert (factory.calls[1].device, factory.calls[1].compute_type) == ("cpu", "int8")
    # fallback_cpu is transient: a successful recovery re-announces readiness
    # (same detail format as load) so a GUI indicator returns to healthy.
    assert [(e.engine, e.state) for e in events] == [
        ("mt", "fallback_cpu"),
        ("mt", "ready"),
    ]
    assert events[-1].detail == "cpu:int8"
    assert out == [("Japanese", "hello world")]
    assert len(factory.built[0].batch_calls) == 1  # failed attempt
    assert len(factory.built[1].batch_calls) == 1  # successful retry


def test_second_oom_after_fallback_propagates(model_dir: Path, monkeypatch):
    monkeypatch.setattr(
        "vrcc.translate.engine.resolve",
        lambda *a: ("cuda", 0, "int8_float16"),
    )
    # both the cuda translator AND the rebuilt cpu translator OOM on translate.
    factory = _RecordingFactory(translate_fail_at=[0, 1])
    cfg = _cfg(device="cuda", compute_type="int8_float16")
    eng = TranslateEngine(_spec("nllb"), model_dir, cfg, EventBus(), translator_factory=factory)
    eng.load()

    with pytest.raises(RuntimeError, match="out of memory"):
        eng.translate("hello world", get("English"), [get("Japanese")])
    assert len(factory.calls) == 2  # rebuilt exactly once, no infinite retry


def test_unrecognized_error_in_ctor_publishes_failed_and_reraises(model_dir: Path):
    bus = EventBus()
    events = _collect(bus)

    class _BoomFactory:
        def __call__(self, *a, **k):
            raise RuntimeError("model.bin is corrupt")

    eng = TranslateEngine(_spec("nllb"), model_dir, _cfg(), bus, translator_factory=_BoomFactory())

    with pytest.raises(RuntimeError, match="corrupt"):
        eng.load()
    assert [(e.engine, e.state) for e in events] == [("mt", "loading"), ("mt", "failed")]
    assert events[-1].detail == "model.bin is corrupt"


# --------------------------------------------------------------------------
# guards: translate before / after a failed load
# --------------------------------------------------------------------------

def test_translate_before_load_raises(model_dir: Path):
    eng = TranslateEngine(_spec("nllb"), model_dir, _cfg(), EventBus(), translator_factory=_RecordingFactory())
    with pytest.raises(RuntimeError):
        eng.translate("hi", get("English"), [get("Japanese")])


def test_translate_after_failed_load_raises(model_dir: Path):
    class _BoomFactory:
        def __call__(self, *a, **k):
            raise RuntimeError("boom")

    eng = TranslateEngine(_spec("nllb"), model_dir, _cfg(), EventBus(), translator_factory=_BoomFactory())
    with pytest.raises(RuntimeError):
        eng.load()
    with pytest.raises(RuntimeError):
        eng.translate("hi", get("English"), [get("Japanese")])


# --------------------------------------------------------------------------
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
