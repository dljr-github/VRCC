"""Existing installs adopt short-reply handling without resetting tuning."""

import pytest

from vrcc.core.config import AppConfig
from vrcc.core.config_migrate import apply_migrations


@pytest.mark.parametrize("version, minimum, expected", [
    (1, 500, 96), (3, 500, 96), (3, 250, 250), (4, 500, 500),
])
def test_short_reply_migration(version, minimum, expected):
    config = AppConfig()
    config.vad.min_utterance_ms = minimum
    config.vad.threshold = 0.6
    config.audio.energy_gate_enabled = True
    apply_migrations(config, version)
    assert config.vad.min_utterance_ms == expected
    assert config.vad.threshold == 0.6
    assert config.audio.energy_gate_enabled
