"""Tests for :func:`vrcc.core.config.default_paths`: portable vs OS-standard
layout and the legacy VRCT2 -> VRCC directory migration. Split out of
test_config.py to stay under the file-size cap.
"""

from vrcc.core.config import Paths, default_paths


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
