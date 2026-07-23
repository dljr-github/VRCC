"""Live-apply dispatch for the Settings dialog (Qt-free, unit-testable).

Diffs each edit against the values already live in the running stack and runs
only the hooks whose group moved, so every setting takes effect with no restart.
Qt stays in the caller's hooks (theme retint / :class:`~vrcc.core.live_apply.
LiveApply` delegation); this module is plain data + dispatch.
"""

from __future__ import annotations

from typing import Callable

# Engine (stt/mt) build inputs that keep the model id: a change rebuilds the
# engine via reload_engine (the model itself hot-swaps through on_model_change).
_STT_ENGINE_FIELDS = ("device", "device_index", "compute_type", "cpu_threads", "num_workers")
_MT_ENGINE_FIELDS = (
    "device", "device_index", "compute_type",
    "inter_threads", "intra_threads", "max_queued_batches",
)
# Chatbox target + send-rate (apply_osc retargets the client and retunes it).
_OSC_FIELDS = ("ip", "port", "min_interval_s", "burst")
# VAD threshold + timings (apply_vad recomputes the segmenter's frame counts).
_VAD_FIELDS = (
    "threshold", "speculative_silence_ms", "finalize_silence_ms",
    "min_utterance_ms", "pre_roll_ms", "max_utterance_s",
)

Spec = tuple[str, tuple, Callable[[], None]]


def live_specs(cfg, apply, text_size_hook: Callable[[], None]) -> tuple[Spec, ...]:
    """``(group, current-values, hook)`` for each field group that applies live.
    The values feed the diff; the hook runs when that group moved. ``gui`` (the
    text-size preset) retints the app (``text_size_hook``); the rest delegate to
    ``apply`` (LiveApply). The palette is fixed, so theme is not a live group."""
    return (
        ("audio", (cfg.audio.device,), lambda: apply.apply_audio_device(cfg.audio.device)),
        ("stt", tuple(getattr(cfg.stt, f) for f in _STT_ENGINE_FIELDS),
         lambda: apply.reload_engine("stt")),
        ("mt", tuple(getattr(cfg.translate, f) for f in _MT_ENGINE_FIELDS),
         lambda: apply.reload_engine("mt")),
        ("osc", tuple(getattr(cfg.osc, f) for f in _OSC_FIELDS),
         lambda: apply.apply_osc(cfg.osc)),
        ("mute", (cfg.mute_sync.enabled,),
         lambda: apply.apply_mute_sync(cfg.mute_sync.enabled)),
        ("vad", tuple(getattr(cfg.vad, f) for f in _VAD_FIELDS),
         lambda: apply.apply_vad(cfg.vad)),
        ("audio_denoise", (cfg.audio.denoise_enabled, cfg.audio.denoise_strength),
         lambda: apply.apply_audio_denoise(cfg.audio)),
        ("gui", (cfg.gui.font_scale,), text_size_hook),
    )


def snapshot(specs) -> dict:
    """Group -> current values, seeding the last-applied baseline."""
    return {group: values for group, values, _ in specs}


def rebaseline(applied: dict, specs, groups: tuple[str, ...]) -> None:
    """Mark ``groups`` as already live at their current values: the caller
    pushed the change through another path (e.g. a model hot-swap, which
    rebuilds with every current engine field), so the next flush must not
    re-run those hooks against a stale baseline."""
    for group, values, _ in specs:
        if group in groups:
            applied[group] = values


def flush(applied: dict, specs) -> None:
    """Run each group's hook whose values moved since ``applied`` was taken,
    updating ``applied`` in place so a later no-op re-selection does nothing."""
    for group, values, hook in specs:
        if applied.get(group) != values:
            applied[group] = values
            hook()
