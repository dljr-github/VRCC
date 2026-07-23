"""Session-scoped fixtures for the real-Whisper integration tests: one loaded
``SttEngine``, shared across every ``test_*`` module in this package so the
(multi-second) model load happens once per pytest run instead of once per
test. Each test module still computes its own module-level cached-model
check for its ``skipif`` guard (see ``_harness.find_cached_whisper``); this
fixture only builds the engine once that guard has already passed.
"""

from __future__ import annotations

import pytest

from vrcc.core.bus import EventBus
from vrcc.core.config import SttConfig

from ._harness import find_cached_whisper


@pytest.fixture(scope="session")
def stt():
    from vrcc.stt.engine import SttEngine

    found = find_cached_whisper()
    if found is None:
        pytest.skip("no cached whisper model")
    model_id, whisper_dir = found

    cfg = SttConfig(model=model_id, device="cpu", source_language="English")
    engine = SttEngine(cfg, whisper_dir, EventBus())
    engine.load()
    try:
        yield engine
    finally:
        engine.unload()
