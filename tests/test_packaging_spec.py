"""Packaging guard for the PyInstaller spec.

onnx_asr reads its own version from dist metadata at import time
(importlib.metadata.version("onnx-asr") in its __init__), so a frozen build
without the dist-info raises PackageNotFoundError the moment the Parakeet
engine imports it. The spec must copy the metadata, bundle the package's
ONNX data files, and keep onnx_asr in hiddenimports for its lazy import.
"""

from pathlib import Path

_SPEC = Path(__file__).resolve().parent.parent / "packaging" / "vrcc.spec"


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
