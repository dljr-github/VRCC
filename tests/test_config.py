import json
import threading
import time

import pytest

from vrcc.core.config import (
    AppConfig,
    AudioConfig,
    ConfigStore,
    GuiConfig,
    MuteSyncConfig,
    OscConfig,
    Paths,
    SttConfig,
    TranslateConfig,
    VadConfig,
    default_paths,
)


def _wait_until(predicate, timeout=2.0, interval=0.01):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()


def test_app_config_defaults_match_spec():
    cfg = AppConfig()
    assert cfg.schema_version == 1
    assert cfg.audio == AudioConfig()
    assert cfg.vad == VadConfig()
    assert cfg.stt == SttConfig()
    assert cfg.translate == TranslateConfig()
    assert cfg.osc == OscConfig()
    assert cfg.mute_sync == MuteSyncConfig()
    assert cfg.gui == GuiConfig()

    assert cfg.translate.targets == ["Japanese"]
    assert cfg.stt.extra_transcribe_kwargs == {}
    assert cfg.translate.extra_translate_kwargs == {}


def test_translate_repetition_guards_default_on():
    cfg = TranslateConfig()
    assert cfg.repetition_penalty == 1.1
    assert cfg.no_repeat_ngram_size == 3


def test_defaults_round_trip_through_save_and_load(tmp_path):
    path = tmp_path / "config.json"
    store = ConfigStore(path)
    store.save_now()

    assert path.exists()

    loaded = ConfigStore(path)
    loaded.load()

    assert loaded.config == AppConfig()
    assert loaded.load_warnings == []


def test_missing_on_load_flags_only_a_fresh_install(tmp_path):
    path = tmp_path / "config.json"
    store = ConfigStore(path)
    store.load()
    assert store.missing_on_load is True

    store.save_now()
    reloaded = ConfigStore(path)
    reloaded.load()
    assert reloaded.missing_on_load is False


def test_missing_on_load_false_for_malformed_file(tmp_path):
    # A corrupt file is an EXISTING config: it loads as defaults, but the
    # first-launch defaulting gated on this flag must never treat it as fresh.
    path = tmp_path / "config.json"
    path.write_text("{not json", encoding="utf-8")
    store = ConfigStore(path)
    store.load()
    assert store.missing_on_load is False
    assert store.load_warnings


def test_write_failure_is_swallowed_not_raised(tmp_path, monkeypatch):
    """Regression: a debounced save runs on a daemon Timer thread; a raise
    there dies to stderr (os.devnull in the windowed exe), i.e. silently, and
    crashes the thread. A failed write must log and return, not raise."""
    import vrcc.core.config as config_mod

    store = ConfigStore(tmp_path / "config.json")
    monkeypatch.setattr(
        config_mod.os, "replace", lambda *a, **k: (_ for _ in ()).throw(OSError("disk full"))
    )
    store.save_now()  # must not raise


def test_edited_values_round_trip(tmp_path):
    path = tmp_path / "config.json"
    store = ConfigStore(path)
    store.config.osc.ip = "10.0.0.5"
    store.config.osc.port = 9001
    store.config.gui.theme = "dark"
    store.save_now()

    loaded = ConfigStore(path)
    loaded.load()

    assert loaded.config.osc.ip == "10.0.0.5"
    assert loaded.config.osc.port == 9001
    assert loaded.config.gui.theme == "dark"
    assert loaded.load_warnings == []


def test_load_missing_file_keeps_defaults_without_error(tmp_path):
    path = tmp_path / "does-not-exist" / "config.json"
    store = ConfigStore(path)

    store.load()  # must not raise

    assert store.config == AppConfig()
    assert store.load_warnings == []


def test_unknown_top_level_keys_are_dropped(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps({"schema_version": 1, "totally_unknown_key": {"x": 1}}),
        encoding="utf-8",
    )

    store = ConfigStore(path)
    store.load()  # must not raise

    assert store.config == AppConfig()
    assert store.load_warnings == []


def test_unknown_keys_within_a_section_are_dropped(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps({"osc": {"ip": "192.168.1.1", "mystery_field": 42}}),
        encoding="utf-8",
    )

    store = ConfigStore(path)
    store.load()

    assert store.config.osc.ip == "192.168.1.1"
    assert store.config.osc.port == OscConfig().port
    assert store.load_warnings == []


def test_invalid_field_falls_back_to_default_with_warning(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps({"osc": {"ip": "192.168.1.1", "port": "not-a-port"}}),
        encoding="utf-8",
    )

    store = ConfigStore(path)
    store.load()

    # The good field in the same section is kept...
    assert store.config.osc.ip == "192.168.1.1"
    # ...the bad one falls back to its default...
    assert store.config.osc.port == OscConfig().port
    # ...and a warning is recorded.
    assert len(store.load_warnings) == 1
    assert "osc.port" in store.load_warnings[0]


def test_invalid_literal_field_falls_back_to_default_with_warning(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps({"gui": {"theme": "not-a-real-theme", "font_scale": 2.0}}),
        encoding="utf-8",
    )

    store = ConfigStore(path)
    store.load()

    assert store.config.gui.theme == GuiConfig().theme
    assert store.config.gui.font_scale == 2.0
    assert len(store.load_warnings) == 1
    assert "gui.theme" in store.load_warnings[0]


@pytest.mark.parametrize("stored", ["light", "system"])
def test_stored_light_or_system_theme_loads_as_dark(tmp_path, stored):
    # Only the dark palette survives; a config written by an older build that
    # stored "light"/"system" must still load, dropping the field back to
    # "dark". It is no longer a valid Literal value, so the field-by-field
    # fallback records a benign gui.theme warning (the good fields are kept).
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps({"gui": {"theme": stored, "font_scale": 1.25}}),
        encoding="utf-8",
    )

    store = ConfigStore(path)
    store.load()

    assert store.config.gui.theme == "dark"
    assert store.config.gui.font_scale == 1.25
    assert len(store.load_warnings) == 1
    assert "gui.theme" in store.load_warnings[0]


def test_section_that_is_not_an_object_falls_back_to_defaults_with_warning(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"osc": "not-an-object"}), encoding="utf-8")

    store = ConfigStore(path)
    store.load()

    assert store.config.osc == OscConfig()
    assert len(store.load_warnings) == 1
    assert "osc" in store.load_warnings[0]


def test_malformed_json_falls_back_to_defaults_with_warning(tmp_path):
    path = tmp_path / "config.json"
    path.write_text("{not valid json", encoding="utf-8")

    store = ConfigStore(path)
    store.load()  # must not raise

    assert store.config == AppConfig()
    assert len(store.load_warnings) == 1


def test_load_resets_warnings_from_previous_call(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"osc": {"port": "bad"}}), encoding="utf-8")

    store = ConfigStore(path)
    store.load()
    assert len(store.load_warnings) == 1

    path.write_text(json.dumps({}), encoding="utf-8")
    store.load()
    assert store.load_warnings == []


def test_save_soon_debounces_repeated_calls_into_a_single_write(tmp_path):
    path = tmp_path / "config.json"
    store = ConfigStore(path, debounce_s=0.05)

    store.save_soon()
    time.sleep(0.02)
    store.save_soon()  # restarts the timer; should not write yet
    time.sleep(0.02)
    store.save_soon()

    assert not path.exists()

    assert _wait_until(path.exists, timeout=2.0)
    mtime_first = path.stat().st_mtime_ns

    # Give it plenty of time past the debounce window; it must not write again.
    time.sleep(0.2)
    assert path.stat().st_mtime_ns == mtime_first


def test_save_soon_is_safe_to_call_from_multiple_threads(tmp_path):
    path = tmp_path / "config.json"
    store = ConfigStore(path, debounce_s=0.05)

    def hammer():
        for _ in range(20):
            store.save_soon()

    threads = [threading.Thread(target=hammer) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert _wait_until(path.exists, timeout=2.0)

    loaded = ConfigStore(path)
    loaded.load()
    assert loaded.config == AppConfig()


def test_save_now_cancels_pending_debounced_write(tmp_path):
    path = tmp_path / "config.json"
    store = ConfigStore(path, debounce_s=5.0)

    store.config.gui.theme = "dark"
    store.save_soon()
    store.save_now()

    assert path.exists()
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["gui"]["theme"] == "dark"


def test_save_now_writes_atomically_leaving_no_tmp_file(tmp_path):
    path = tmp_path / "config.json"
    store = ConfigStore(path)
    store.save_now()

    assert path.exists()
    assert not path.with_suffix(".json.tmp").exists()


def test_saved_json_is_pretty_printed_and_not_ascii_escaped(tmp_path):
    path = tmp_path / "config.json"
    store = ConfigStore(path)
    store.config.stt.initial_prompt = "こんにちは"
    store.save_now()

    text = path.read_text(encoding="utf-8")
    assert "\n  " in text  # indent=2
    assert "こんにちは" in text  # ensure_ascii=False


def test_default_paths_portable_lands_under_given_app_dir(tmp_path):
    paths = default_paths(portable=True, app_dir=tmp_path)

    assert isinstance(paths, Paths)
    assert paths.config_file == tmp_path / "config.json"
    assert paths.models_dir == tmp_path / "models"
    assert paths.logs_dir == tmp_path / "logs"


def test_default_paths_non_portable_uses_platformdirs(monkeypatch, tmp_path):
    fake_config_dir = tmp_path / "cfgdir"
    fake_data_dir = tmp_path / "datadir"

    monkeypatch.setattr(
        "vrcc.core.config.platformdirs.user_config_dir",
        lambda app_name: str(fake_config_dir),
    )
    monkeypatch.setattr(
        "vrcc.core.config.platformdirs.user_data_dir",
        lambda app_name: str(fake_data_dir),
    )

    paths = default_paths(portable=False)

    assert paths.config_file == fake_config_dir / "config.json"
    assert paths.models_dir == fake_data_dir / "models"
    assert paths.logs_dir == fake_data_dir / "logs"


def _legacy_vs_new_resolver(legacy_dir, new_dir):
    """Return a platformdirs-shaped fn: "VRCT2" -> legacy_dir, "VRCC" -> new_dir.

    Mirrors Windows, where user_config_dir and user_data_dir resolve to the
    same base path for a given appname -- both patched callables can share
    this resolver.
    """

    def _resolve(appname):
        return str(legacy_dir if appname == "VRCT2" else new_dir)

    return _resolve


def test_default_paths_migrates_legacy_vrcc_dir_to_vrcc(monkeypatch, tmp_path):
    legacy_dir = tmp_path / "legacy"
    new_dir = tmp_path / "new"
    (legacy_dir / "models").mkdir(parents=True)
    sentinel_bytes = b"\x00\x01not-actually-a-model\xff"
    (legacy_dir / "models" / "keep.bin").write_bytes(sentinel_bytes)

    resolver = _legacy_vs_new_resolver(legacy_dir, new_dir)
    monkeypatch.setattr("vrcc.core.config.platformdirs.user_config_dir", resolver)
    monkeypatch.setattr("vrcc.core.config.platformdirs.user_data_dir", resolver)

    paths = default_paths(portable=False)

    assert new_dir.exists()
    assert not legacy_dir.exists()
    assert (new_dir / "models" / "keep.bin").read_bytes() == sentinel_bytes
    assert paths.config_file == new_dir / "config.json"
    assert paths.models_dir == new_dir / "models"
    assert paths.logs_dir == new_dir / "logs"


def test_default_paths_migration_falls_back_to_legacy_dir_on_move_failure(monkeypatch, tmp_path):
    legacy_dir = tmp_path / "legacy"
    new_dir = tmp_path / "new"
    (legacy_dir / "models").mkdir(parents=True)
    sentinel_bytes = b"\x00\x01not-actually-a-model\xff"
    (legacy_dir / "models" / "keep.bin").write_bytes(sentinel_bytes)

    resolver = _legacy_vs_new_resolver(legacy_dir, new_dir)
    monkeypatch.setattr("vrcc.core.config.platformdirs.user_config_dir", resolver)
    monkeypatch.setattr("vrcc.core.config.platformdirs.user_data_dir", resolver)

    def _raise(*args, **kwargs):
        raise OSError("simulated move failure")

    monkeypatch.setattr("vrcc.core.config.shutil.move", _raise)

    paths = default_paths(portable=False)

    assert not new_dir.exists()
    assert legacy_dir.exists()
    assert (legacy_dir / "models" / "keep.bin").read_bytes() == sentinel_bytes
    assert paths.config_file == legacy_dir / "config.json"
    assert paths.models_dir == legacy_dir / "models"
    assert paths.logs_dir == legacy_dir / "logs"


# -- apply_profile (Latency/Quality kwargs bundles) --------------------------


def test_apply_profile_quality_applies_bundle_and_records_choice():
    from vrcc.core.config import apply_profile

    cfg = AppConfig()
    apply_profile(cfg, "quality")

    assert cfg.stt.beam_size == 5
    assert cfg.stt.temperature == 0.0
    assert cfg.translate.beam_size == 3
    assert cfg.vad.speculative_silence_ms == 450
    assert cfg.vad.finalize_silence_ms == 800
    assert cfg.vad.pre_roll_ms == 200
    assert cfg.gui.profile == "quality"


def test_apply_profile_latency_restores_defaults():
    from vrcc.core.config import apply_profile

    cfg = AppConfig()
    apply_profile(cfg, "quality")
    apply_profile(cfg, "latency")

    assert cfg.stt.beam_size == 1
    assert cfg.translate.beam_size == 1
    assert cfg.vad.speculative_silence_ms == 350
    assert cfg.vad.finalize_silence_ms == 600
    assert cfg.vad.pre_roll_ms == 150
    assert cfg.gui.profile == "latency"


def test_apply_profile_unknown_profile_raises():
    from vrcc.core.config import apply_profile

    with pytest.raises(KeyError):
        apply_profile(AppConfig(), "turbo")


def test_osc_overflow_defaults_to_split():
    assert OscConfig().overflow == "split"


def test_osc_split_delay_s_defaults_to_2_0():
    assert OscConfig().split_delay_s == 2.0


def test_profiles_reference_only_real_config_fields():
    # Guards the bundles against config-field renames drifting silently.
    from vrcc.core.config import PROFILES

    cfg = AppConfig()
    for preset in PROFILES.values():
        for section_name, fields in preset.items():
            section = getattr(cfg, section_name)
            for field_name in fields:
                assert hasattr(section, field_name)
