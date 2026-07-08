"""End-to-end translation against the real nllb-600M-int8 CTranslate2 model:
the layout-correctness gate proving per-family token layout and special-token
stripping yield clean target text. Integration-marked (~650MB download).
"""

from __future__ import annotations

import pytest

from vrcc.core.bus import EventBus
from vrcc.core.config import TranslateConfig, default_paths
from vrcc.core.languages import get
from vrcc.download.manager import DownloadManager
from vrcc.translate.engine import TranslateEngine
from vrcc.translate.registry import MT_MODELS


@pytest.mark.integration
def test_nllb_600m_en_to_ja_produces_clean_japanese():
    spec = MT_MODELS["nllb-600M-int8"]
    bus = EventBus()

    # Download (or reuse) into the real default models dir so later tasks share it.
    models_dir = default_paths(portable=False).models_dir
    manager = DownloadManager(models_dir, bus)
    model_dir = manager.ensure_mt(spec)

    cfg = TranslateConfig(
        model=spec.id,
        device="cpu",
        compute_type="int8",
        targets=["Japanese"],
        beam_size=1,
    )
    engine = TranslateEngine(spec, model_dir, cfg, bus)
    engine.load()
    try:
        out = engine.translate("Hello, how are you?", get("English"), [get("Japanese")])
    finally:
        engine.unload()

    assert len(out) == 1
    name, text = out[0]
    assert name == "Japanese"
    assert text.strip(), "translation must be non-empty"

    # Contains CJK ideographs / kana (U+3040-U+9FFF).
    assert any("぀" <= c <= "鿿" for c in text), f"no CJK in {text!r}"

    # No layout artifacts leaking through: FLORES tag fragments, m2m100 tags,
    # or the EOS token.
    for artifact in ("_Latn", "__", "</s>"):
        assert artifact not in text, f"artifact {artifact!r} leaked into {text!r}"
