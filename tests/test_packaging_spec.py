"""Packaging guard for the PyInstaller spec.

onnx_asr reads its own version from dist metadata at import time
(importlib.metadata.version("onnx-asr") in its __init__), so a frozen build
without the dist-info raises PackageNotFoundError the moment the Parakeet
engine imports it. The spec must copy the metadata, bundle the package's
ONNX data files, and keep onnx_asr in hiddenimports for its lazy import.

The branding wiring (exe icon, inline version resource) is guarded the same
way, by spec text, so these checks run without PyInstaller installed.
"""

import re
from pathlib import Path

from vrcc import __version__

_SPEC = Path(__file__).resolve().parent.parent / "packaging" / "vrcc.spec"

# The exact pattern body the spec uses (there as a raw string literal) to
# read the version out of vrcc/__init__.py without importing the package.
_VERSION_RE = '^__version__ = "([^"]+)"'


def test_spec_copies_onnx_asr_metadata():
    text = _SPEC.read_text(encoding="utf-8")
    assert 'copy_metadata("onnx-asr")' in text, (
        "vrcc.spec must bundle the onnx-asr dist-info; onnx_asr reads its "
        "version from it at import time in the frozen build"
    )


def test_spec_collects_onnx_asr_data_files():
    text = _SPEC.read_text(encoding="utf-8")
    assert 'collect_data_files("onnx_asr")' in text, (
        "vrcc.spec must bundle onnx_asr package data (preprocessor ONNX "
        "graphs the Parakeet engine loads at runtime)"
    )


def test_spec_hides_onnx_asr_import():
    text = _SPEC.read_text(encoding="utf-8")
    assert "hiddenimports = [" in text, "vrcc.spec must define hiddenimports"
    block = text.split("hiddenimports = [", 1)[1].split("]", 1)[0]
    assert '"onnx_asr"' in block, (
        "vrcc.spec must list onnx_asr in hiddenimports; vrcc.stt.onnx_asr "
        "imports it lazily at engine load time"
    )


def test_spec_lands_the_ico_inside_the_frozen_vrcc_package():
    text = _SPEC.read_text(encoding="utf-8")
    assert '(os.path.join(REPO_ROOT, "vrcc", "vrcc.ico"), "vrcc")' in text, (
        "vrcc.spec must copy vrcc/vrcc.ico to vrcc/ in _internal; "
        "vrcc.gui.style resolves the window icon relative to the package"
    )


def test_spec_sets_the_exe_icon():
    text = _SPEC.read_text(encoding="utf-8")
    assert 'icon=os.path.join(REPO_ROOT, "vrcc", "vrcc.ico")' in text, (
        "vrcc.spec must give EXE the repo ICO so the exe carries the icon"
    )


def test_spec_builds_the_version_resource_inline():
    text = _SPEC.read_text(encoding="utf-8")
    assert "VSVersionInfo(" in text, (
        "vrcc.spec must construct the version resource inline; a separate "
        "version file could desync from the package version"
    )
    assert "version=version_info" in text, (
        "vrcc.spec must hand the version resource to EXE"
    )
    assert 'StringStruct("CompanyName", "dljr-github")' in text


def test_spec_version_regex_matches_the_package_init():
    text = _SPEC.read_text(encoding="utf-8")
    assert _VERSION_RE in text, (
        "vrcc.spec must parse the version from vrcc/__init__.py with this "
        "exact regex (importing vrcc from the spec would trigger package "
        "imports during analysis)"
    )
    init_text = (
        _SPEC.parent.parent / "vrcc" / "__init__.py"
    ).read_text(encoding="utf-8")
    match = re.search(_VERSION_RE, init_text, re.MULTILINE)
    assert match, "the spec's version regex no longer matches vrcc/__init__.py"
    assert match.group(1) == __version__


def test_pyproject_version_matches_package_init():
    pyproject = (
        Path(__file__).resolve().parent.parent / "pyproject.toml"
    ).read_text(encoding="utf-8")
    m = re.search(r'^version = "([^"]+)"', pyproject, re.MULTILINE)
    assert m, "pyproject.toml has no version line"
    assert m.group(1) == __version__, (
        f"pyproject.toml version {m.group(1)} != vrcc/__init__.py {__version__}"
    )
