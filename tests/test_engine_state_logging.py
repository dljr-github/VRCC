"""One bus subscriber writes every EngineStateChanged to the log, so the
device an engine came up on, and any fallback or failure before it, is in
the file a bug report attaches. Subscribed where the bus is built
(vrcc.app.run), so no engine has to remember a log line beside its publish.
"""

from __future__ import annotations

import logging

from vrcc.core.bus import EventBus
from vrcc.core.events import EngineStateChanged
from vrcc.core.logs import log_engine_states

_LOGGER = "vrcc.core.logs"


def _lines(caplog) -> list[str]:
    return [r.message for r in caplog.records if r.name == _LOGGER]


def test_ready_logs_the_engine_and_its_device(caplog):
    bus = EventBus()
    log_engine_states(bus)

    with caplog.at_level(logging.INFO, logger=_LOGGER):
        bus.publish(EngineStateChanged("stt", "ready", "cuda:float16"))
        bus.publish(EngineStateChanged("mt", "ready", "cpu:int8"))

    assert _lines(caplog) == [
        "stt engine ready: cuda:float16",
        "mt engine ready: cpu:int8",
    ]


def test_fallback_and_failure_carry_their_detail(caplog):
    bus = EventBus()
    log_engine_states(bus)

    with caplog.at_level(logging.INFO, logger=_LOGGER):
        bus.publish(EngineStateChanged("stt", "fallback_cpu", "CUDA out of memory"))
        bus.publish(EngineStateChanged("stt", "failed", "no model files"))

    assert _lines(caplog) == [
        "stt engine fallback_cpu: CUDA out of memory",
        "stt engine failed: no model files",
    ]


def test_a_state_without_detail_logs_bare(caplog):
    bus = EventBus()
    log_engine_states(bus)

    with caplog.at_level(logging.INFO, logger=_LOGGER):
        bus.publish(EngineStateChanged("stt", "loading"))

    assert _lines(caplog) == ["stt engine loading"]
