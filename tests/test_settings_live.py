"""Qt-free tests for :mod:`vrcc.gui.settings_live` -- the diff/dispatch behind
the Settings dialog's live apply. Each field group must fire its own hook
exactly once when it moves, skip an unchanged group, and route the engine /
audio / OSC / mute / VAD groups to the apply handle while ``gui`` (theme +
text size) uses the separate retint hook. UI language is deliberately no group.
"""

from __future__ import annotations

import pytest

from vrcc.core.config import AppConfig
from vrcc.gui import settings_live


class _RecordingApply:
    def __init__(self) -> None:
        self.calls: list = []

    def apply_audio_device(self, device):
        self.calls.append(("audio", device))
        return True

    def apply_audio_gain(self, cfg):
        self.calls.append(("audio_gain", cfg.gain_db, cfg.auto_gain))

    def reload_engine(self, kind):
        self.calls.append(("reload", kind))

    def apply_osc(self, cfg):
        self.calls.append(("osc", cfg.port))

    def apply_mute_sync(self, enabled):
        self.calls.append(("mute", enabled))

    def apply_vad(self, cfg):
        self.calls.append(("vad", cfg.finalize_silence_ms))


class _Counter:
    def __init__(self) -> None:
        self.count = 0

    def __call__(self) -> None:
        self.count += 1


def _env():
    cfg = AppConfig()
    apply = _RecordingApply()
    theme = _Counter()
    applied = settings_live.snapshot(settings_live.live_specs(cfg, apply, theme))
    return cfg, apply, theme, applied


def _flush(cfg, apply, theme, applied):
    settings_live.flush(applied, settings_live.live_specs(cfg, apply, theme))


def test_no_change_fires_nothing():
    cfg, apply, theme, applied = _env()
    _flush(cfg, apply, theme, applied)
    assert apply.calls == []
    assert theme.count == 0


@pytest.mark.parametrize(
    "mutate, expected",
    [
        (lambda c: setattr(c.audio, "device", "Mic X"), ("audio", "Mic X")),
        (lambda c: setattr(c.stt, "device", "cpu"), ("reload", "stt")),
        (lambda c: setattr(c.stt, "cpu_threads", 4), ("reload", "stt")),
        (lambda c: setattr(c.translate, "compute_type", "int8"), ("reload", "mt")),
        (lambda c: setattr(c.translate, "intra_threads", 3), ("reload", "mt")),
        (lambda c: setattr(c.osc, "port", 9002), ("osc", 9002)),
        (lambda c: setattr(c.osc, "ip", "10.0.0.2"), ("osc", 9000)),
        (lambda c: setattr(c.mute_sync, "enabled", False), ("mute", False)),
        (lambda c: setattr(c.vad, "finalize_silence_ms", 999), ("vad", 999)),
        (lambda c: setattr(c.vad, "threshold", 0.42), ("vad", 600)),
    ],
)
def test_each_field_routes_to_its_hook_once(mutate, expected):
    cfg, apply, theme, applied = _env()
    mutate(cfg)
    _flush(cfg, apply, theme, applied)
    assert apply.calls == [expected]
    assert theme.count == 0
    # A second flush with no further change must not re-fire the hook.
    _flush(cfg, apply, theme, applied)
    assert apply.calls == [expected]


def test_change_then_revert_before_flush_is_a_noop():
    cfg, apply, theme, applied = _env()
    original = cfg.stt.cpu_threads
    cfg.stt.cpu_threads = original + 2
    cfg.stt.cpu_threads = original  # back to the applied value before any flush
    _flush(cfg, apply, theme, applied)
    assert apply.calls == []


def test_two_groups_move_together_fire_both_once():
    cfg, apply, theme, applied = _env()
    cfg.stt.device = "cpu"
    cfg.osc.port = 9100
    _flush(cfg, apply, theme, applied)
    assert apply.calls == [("reload", "stt"), ("osc", 9100)]


def test_gui_group_uses_theme_hook_not_the_apply_handle():
    # The gui group (theme + text size) retints via the theme/font hook, never
    # through the apply handle. Theme is dark-only, so text size drives the
    # move; a second move retints again, without a rebuild.
    cfg, apply, theme, applied = _env()
    cfg.gui.font_scale = 1.2
    _flush(cfg, apply, theme, applied)
    assert theme.count == 1
    assert apply.calls == []
    cfg.gui.font_scale = 1.4
    _flush(cfg, apply, theme, applied)
    assert theme.count == 2


def test_gain_edit_runs_audio_gain_hook_once_without_source_restart():
    # The gain fields must apply in place, never through apply_audio_device
    # (which restarts the source).
    cfg, apply, theme, applied = _env()
    cfg.audio.gain_db = 6.0
    _flush(cfg, apply, theme, applied)
    assert apply.calls == [("audio_gain", 6.0, True)]
    # A second flush with no further change must not re-fire the hook.
    _flush(cfg, apply, theme, applied)
    assert apply.calls == [("audio_gain", 6.0, True)]


def test_ui_language_is_not_a_live_group():
    # tr() is read at construction, so a language change can't apply live; it is
    # handled by a window rebuild on dialog close, never by the live flush.
    cfg, apply, theme, applied = _env()
    cfg.gui.ui_language = "ja"
    _flush(cfg, apply, theme, applied)
    assert apply.calls == []
    assert theme.count == 0
