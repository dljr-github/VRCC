"""Schema migrations applied when an existing config.json is loaded.

A stored file carries every field, so a changed default reaches new installs
only. Anything that has to reach existing users needs a migration here.
"""

import json

import pytest

from vrcc.core.config import ConfigStore, TranslateConfig


def _stored(path, raw: dict) -> ConfigStore:
    path.write_text(json.dumps(raw), encoding="utf-8")
    store = ConfigStore(path)
    store.load()
    return store


def test_translate_beam_defaults_to_four():
    # Greedy MT rewrites content it cannot translate: "Okay 1,2,3,4,5,6,7"
    # came back as "现在,我们要做什么?" at beam 1, and correct from beam 2 up.
    assert TranslateConfig().beam_size == 4


@pytest.mark.parametrize("stored", [1, 3])
def test_schema_1_profile_written_mt_beam_migrates_to_default(tmp_path, stored):
    store = _stored(
        tmp_path / "config.json",
        {"schema_version": 1, "translate": {"beam_size": stored}},
    )

    assert store.config.translate.beam_size == TranslateConfig().beam_size
    assert store.config.schema_version == 2


def test_schema_1_hand_picked_mt_beam_is_kept(tmp_path):
    # Only the two widths the old profile bundles wrote are migrated; one the
    # user set on the Advanced page is theirs to keep.
    store = _stored(
        tmp_path / "config.json",
        {"schema_version": 1, "translate": {"beam_size": 2}},
    )

    assert store.config.translate.beam_size == 2


def test_schema_2_mt_beam_is_never_migrated(tmp_path):
    store = _stored(
        tmp_path / "config.json",
        {"schema_version": 2, "translate": {"beam_size": 1}},
    )

    assert store.config.translate.beam_size == 1


def test_config_with_no_schema_version_is_migrated(tmp_path):
    # Predates the field entirely, so it predates the split too.
    store = _stored(tmp_path / "config.json", {"translate": {"beam_size": 1}})

    assert store.config.translate.beam_size == TranslateConfig().beam_size


def test_migration_leaves_other_stored_translate_fields_alone(tmp_path):
    store = _stored(
        tmp_path / "config.json",
        {
            "schema_version": 1,
            "translate": {
                "beam_size": 1,
                "model": "m2m100-1.2B-int8",
                "targets": ["Korean", "Japanese"],
                "repetition_penalty": 1.4,
            },
        },
    )

    assert store.config.translate.model == "m2m100-1.2B-int8"
    assert store.config.translate.targets == ["Korean", "Japanese"]
    assert store.config.translate.repetition_penalty == 1.4


def test_a_newer_schema_version_is_not_written_backwards(tmp_path):
    # A file from a later build carries fields this one keeps but does not
    # understand. Stamping our own number on it makes that build re-run its
    # migration over data it has already migrated.
    store = _stored(
        tmp_path / "config.json",
        {"schema_version": 5, "translate": {"beam_size": 7}},
    )

    assert store.config.schema_version == 5
    assert store.config.translate.beam_size == 7

    store.save_now()
    written = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
    assert written["schema_version"] == 5
