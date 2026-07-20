"""Typed application configuration with debounced JSON persistence.

`AppConfig` (one pydantic model per subsystem) is the single source of truth.
`ConfigStore` owns config.json and reads it tolerantly (unknown keys dropped, bad
sections/fields default with a warning in `load_warnings`, never raises); writes
are debounced.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import sys
import threading
from pathlib import Path
from typing import Any, Literal, NamedTuple

import platformdirs
from pydantic import BaseModel, Field, ValidationError

logger = logging.getLogger("vrcc.core.config")


class AudioConfig(BaseModel):
    device: str = "auto"
    energy_gate_enabled: bool = False
    energy_threshold: int = 300
    # Capture gain applied before VAD/STT (dB). auto_gain overrides the fixed
    # value with a smoothed auto-level toward a target loudness.
    gain_db: float = 0.0
    auto_gain: bool = True


class VadConfig(BaseModel):
    threshold: float = 0.5
    # Silence bar, decoupled from the speech threshold so raising sensitivity
    # (lowering the speech threshold) never raises the silence bar and chops
    # words mid-utterance. Clamped below the speech threshold at use.
    silence_threshold: float = 0.35
    speculative_silence_ms: int = 250
    finalize_silence_ms: int = 600
    min_utterance_ms: int = 500
    pre_roll_ms: int = 150
    max_utterance_s: float = 28.0
    # Emit a sentence to the chatbox as soon as a speculative transcription
    # ends in terminal punctuation, instead of waiting for the full stop.
    sentence_inject: bool = True
    # Words required before sentence_inject fires early. A gate of 2 lets a
    # mid-sentence pause (a comma, a breath) leave behind a short punctuated
    # fragment that reads as a complete sentence and gets injected too soon.
    sentence_min_words: int = 3
    # Stream the in-progress transcription to the log and chatbox while the
    # utterance is still active, ahead of the speculative/final result.
    live_partials: bool = True
    # How often the segmenter snapshots the active buffer for a live partial.
    partial_interval_ms: int = 300


class SttConfig(BaseModel):
    model: str = "small"
    device: str = "auto"  # "auto"|"cpu"|"cuda"
    device_index: int = 0
    compute_type: str = "auto"
    cpu_threads: int = 0
    num_workers: int = 1
    source_language: str = "English"  # display name; "auto" = detect
    beam_size: int = 1
    temperature: float = 0.0
    condition_on_previous_text: bool = False
    without_timestamps: bool = True
    avg_logprob_gate: float = -0.8
    no_speech_gate: float = 0.6
    initial_prompt: str = ""
    no_repeat_ngram_size: int = 3
    # Drop a transcription whose worst segment compresses past this. A runaway
    # whisper repetition loop lands far above real speech; whisper's own
    # degeneracy line is 2.4, so 2.5 only catches clear loops.
    compression_ratio_gate: float = 2.5
    extra_transcribe_kwargs: dict = Field(default_factory=dict)


class TranslateConfig(BaseModel):
    enabled: bool = True
    model: str = "nllb-600M-int8"
    device: str = "auto"
    device_index: int = 0
    compute_type: str = "auto"
    inter_threads: int = 1
    intra_threads: int = 0
    # ctranslate2.Translator(max_queued_batches=...): 0 = automatic (CT2's
    # own default), -1 = unlimited. Applied at Translator build time.
    max_queued_batches: int = 0
    targets: list[str] = Field(default_factory=lambda: ["Japanese"])  # display names, max 3
    beam_size: int = 1
    repetition_penalty: float = 1.1
    no_repeat_ngram_size: int = 3
    extra_translate_kwargs: dict = Field(default_factory=dict)


class OscConfig(BaseModel):
    ip: str = "127.0.0.1"
    port: int = 9000
    send_to_vrchat: bool = True
    notification_sfx: bool = False
    min_interval_s: float = Field(default=1.3, gt=0)
    burst: int = Field(default=5, ge=1)
    overflow: Literal["truncate", "split", "send"] = "split"
    split_delay_s: float = 2.0
    include_original: bool = True
    translation_separator: str = "\n"
    coalesce_latest_wins: bool = True


class MuteSyncConfig(BaseModel):
    enabled: bool = True
    mode: Literal["pause", "ignore", "invert"] = "pause"


class GuiConfig(BaseModel):
    profile: Literal["latency", "quality"] = "latency"
    # Field kept so stored configs and callers keep loading; only one palette
    # exists, so a stored "light"/"system" is dropped back to this default.
    theme: Literal["dark"] = "dark"
    font_scale: float = 1.0
    # UI language code from vrcc.i18n.UI_LANGUAGES; "auto" follows the OS
    # locale. Free-form (not Literal) so a catalog added later needs no
    # schema change; unknown values resolve to English at startup.
    ui_language: str = "auto"
    window_geometry: str = ""
    # Check GitHub releases on launch and offer a notice. Opt out here.
    update_check_enabled: bool = True


class AppConfig(BaseModel):
    schema_version: int = 1
    audio: AudioConfig = Field(default_factory=AudioConfig)
    vad: VadConfig = Field(default_factory=VadConfig)
    stt: SttConfig = Field(default_factory=SttConfig)
    translate: TranslateConfig = Field(default_factory=TranslateConfig)
    osc: OscConfig = Field(default_factory=OscConfig)
    mute_sync: MuteSyncConfig = Field(default_factory=MuteSyncConfig)
    gui: GuiConfig = Field(default_factory=GuiConfig)


# Latency/Quality kwargs bundles: section -> {field: value}. Single source of
# truth for the Speed/Quality profile controls.
PROFILES: dict[str, dict[str, dict[str, Any]]] = {
    "latency": {
        "vad": {
            "speculative_silence_ms": 250,
            "finalize_silence_ms": 600,
            "min_utterance_ms": 500,
            "pre_roll_ms": 150,
            "max_utterance_s": 28.0,
        },
        "stt": {"beam_size": 1, "temperature": 0.0},
        "translate": {"beam_size": 1},
    },
    "quality": {
        "vad": {
            "speculative_silence_ms": 350,
            "finalize_silence_ms": 800,
            "min_utterance_ms": 500,
            "pre_roll_ms": 200,
            "max_utterance_s": 28.0,
        },
        "stt": {"beam_size": 5, "temperature": 0.0},
        "translate": {"beam_size": 3},
    },
}


def apply_profile(config: AppConfig, profile: str) -> None:
    """Apply the ``profile`` bundle from :data:`PROFILES` to ``config`` in place
    and record it in ``config.gui.profile``. beam_size fields take effect next
    utterance; VAD timings apply on next launch. Raises ``KeyError`` if unknown.
    """
    preset = PROFILES[profile]
    for section_name, fields in preset.items():
        section = getattr(config, section_name)
        for field_name, value in fields.items():
            setattr(section, field_name, value)
    config.gui.profile = profile  # type: ignore[assignment]


# Section name -> model class, in the same order as AppConfig's fields.
_SECTION_MODELS: dict[str, type[BaseModel]] = {
    "audio": AudioConfig,
    "vad": VadConfig,
    "stt": SttConfig,
    "translate": TranslateConfig,
    "osc": OscConfig,
    "mute_sync": MuteSyncConfig,
    "gui": GuiConfig,
}


class Paths(NamedTuple):
    config_file: Path
    models_dir: Path
    logs_dir: Path


def _running_app_dir() -> Path:
    """Best-effort directory of the running app, for portable mode."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(sys.argv[0]).resolve().parent


def _migrate_legacy_dir(new_dir: Path, legacy_dir: Path) -> Path:
    """One-shot rename of a pre-VRCC app dir to its VRCC location.

    Returns the dir paths should use: the new dir normally; the legacy dir
    if the move failed, so gigabytes of downloaded models are never left
    stranded or inaccessible.
    """
    if new_dir.exists() or not legacy_dir.exists():
        return new_dir
    try:
        new_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(legacy_dir), str(new_dir))
        logger.info("migrated app dir %s -> %s", legacy_dir, new_dir)
        return new_dir
    except OSError:
        logger.warning(
            "could not migrate legacy app dir %s; using it in place",
            legacy_dir,
            exc_info=True,
        )
        return legacy_dir


def default_paths(portable: bool, app_dir: Path | None = None) -> Paths:
    """Where config/models/logs live for this run.

    Non-portable: OS-standard per-user dirs (`platformdirs`). Portable:
    everything under `app_dir` (defaults to the running app's dir; tests can
    pass `tmp_path`).
    """
    if portable:
        base = app_dir if app_dir is not None else _running_app_dir()
        return Paths(
            config_file=base / "config.json",
            models_dir=base / "models",
            logs_dir=base / "logs",
        )
    config_dir = _migrate_legacy_dir(
        Path(platformdirs.user_config_dir("VRCC")),
        Path(platformdirs.user_config_dir("VRCT2")),
    )
    data_dir = _migrate_legacy_dir(
        Path(platformdirs.user_data_dir("VRCC")),
        Path(platformdirs.user_data_dir("VRCT2")),
    )
    return Paths(
        config_file=config_dir / "config.json",
        models_dir=data_dir / "models",
        logs_dir=data_dir / "logs",
    )


class ConfigStore:
    """Owns one `AppConfig` bound to a `config.json` file on disk."""

    def __init__(self, path: Path, debounce_s: float = 2.0) -> None:
        self.path = path
        self.debounce_s = debounce_s
        self.config = AppConfig()
        self.load_warnings: list[str] = []
        # True only when the last load() found no file at all (a fresh
        # install). A malformed file is an EXISTING config: first-launch
        # defaulting (OS caption language) must not rewrite the user's choices.
        self.missing_on_load = False
        self._lock = threading.Lock()
        self._timer: threading.Timer | None = None

    def load(self) -> None:
        """Populate `self.config` from `self.path`.

        A missing or malformed file (bad JSON, non-object root) falls back to
        defaults (warning for malformed). Otherwise each section is validated
        independently, defaulting any invalid field and warning per fallback.
        """
        self.load_warnings = []
        self.missing_on_load = not self.path.exists()

        if self.missing_on_load:
            self.config = AppConfig()
            return

        try:
            raw_text = self.path.read_text(encoding="utf-8")
            raw = json.loads(raw_text)
        except (OSError, json.JSONDecodeError) as exc:
            self.load_warnings.append(f"could not read config file, using defaults: {exc}")
            self.config = AppConfig()
            return

        if not isinstance(raw, dict):
            self.load_warnings.append("config file root is not a JSON object, using defaults")
            self.config = AppConfig()
            return

        self.config = self._build_config(raw)

    def _build_config(self, raw: dict[str, Any]) -> AppConfig:
        data: dict[str, Any] = {}

        if "schema_version" in raw:
            try:
                data["schema_version"] = int(raw["schema_version"])
            except (TypeError, ValueError):
                self.load_warnings.append(
                    f"schema_version: invalid value {raw['schema_version']!r}, using default"
                )

        for name, model_cls in _SECTION_MODELS.items():
            if name not in raw:
                continue
            section_raw = raw[name]
            if not isinstance(section_raw, dict):
                self.load_warnings.append(f"{name}: invalid section, using defaults")
                continue
            data[name] = self._load_section(name, model_cls, section_raw)

        return AppConfig(**data)

    def _load_section(
        self, name: str, model_cls: type[BaseModel], section_raw: dict[str, Any]
    ) -> BaseModel:
        # Drop unknown keys silently before validating, so an otherwise-valid
        # section with a stray key isn't treated as a validation failure.
        known = {k: v for k, v in section_raw.items() if k in model_cls.model_fields}

        try:
            return model_cls.model_validate(known)
        except ValidationError:
            pass

        # Whole-section validation failed: fall back field-by-field, keeping
        # every field that validates on its own and defaulting the rest.
        defaults_dump = model_cls().model_dump()
        accepted: dict[str, Any] = {}
        for field_name, value in known.items():
            trial = dict(defaults_dump)
            trial[field_name] = value
            try:
                validated = model_cls.model_validate(trial)
            except ValidationError:
                self.load_warnings.append(
                    f"{name}.{field_name}: invalid value {value!r}, using default"
                )
                continue
            accepted[field_name] = getattr(validated, field_name)
        return model_cls(**accepted)

    def save_soon(self) -> None:
        """Schedule a debounced write. Safe to call from any thread; repeat
        calls within the debounce window restart the timer so only the last
        one actually fires."""
        with self._lock:
            if self._timer is not None:
                self._timer.cancel()
            timer = threading.Timer(self.debounce_s, self._on_timer_fire)
            timer.daemon = True
            self._timer = timer
            timer.start()

    def save_now(self) -> None:
        """Cancel any pending debounced write and write immediately."""
        with self._lock:
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None
            self._write()

    def _on_timer_fire(self) -> None:
        with self._lock:
            self._timer = None
            self._write()

    def _write(self) -> None:
        # Debounced save runs on a daemon Timer thread; an unhandled raise there
        # would vanish silently (stderr is os.devnull in the windowed exe). Log
        # instead (disk full, AV/OneDrive lock, or permissions are usual causes).
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = self.path.with_suffix(".json.tmp")
            tmp_path.write_text(
                json.dumps(self.config.model_dump(), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            os.replace(tmp_path, self.path)
        except OSError:
            logger.warning("failed to write config to %s", self.path, exc_info=True)
