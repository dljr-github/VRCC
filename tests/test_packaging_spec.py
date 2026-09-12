"""Packaging guard for the PyInstaller spec.

onnx_asr reads its own version from dist metadata at import time
(importlib.metadata.version("onnx-asr") in its __init__), so a frozen build
without the dist-info raises PackageNotFoundError the moment the Parakeet
engine imports it. The spec must copy the metadata, bundle the package's
ONNX data files, and keep onnx_asr in hiddenimports for its lazy import.

The branding wiring (exe icon, inline version resource) is guarded the same
way, by spec text, so these checks run without PyInstaller installed.
"""

import importlib.util
import os
import re
import sys
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


def test_spec_ships_soundcard():
    """Captioning what you hear imports soundcard lazily, inside a function on
    the capture thread, which PyInstaller's static analysis cannot see. Without
    both of these the feature is dead in every packaged build, and the error it
    then shows tells the user to reinstall, which would not help."""
    text = _SPEC.read_text(encoding="utf-8")
    block = text.split("hiddenimports = [", 1)[1].split("]", 1)[0]
    assert '"soundcard"' in block, (
        "vrcc.spec must list soundcard in hiddenimports; vrcc.audio.loopback "
        "imports it inside a function"
    )
    assert 'collect_data_files("soundcard")' in text, (
        "vrcc.spec must bundle soundcard's package data; it reads a cffi cdef "
        "header (mediafoundation.py.h) from beside its own source at import"
    )


_SPLASH_PNG = Path(__file__).resolve().parent.parent / "assets" / "splash.png"
_SPLASH_SVG = Path(__file__).resolve().parent.parent / "assets" / "splash.svg"
_MAKE_SPLASH_SOURCE = Path(__file__).resolve().parent.parent / "tools" / "make_splash.py"

# PyInstaller resizes an oversized splash only when Pillow is installed, and
# Pillow is not a dependency of this project. PyInstaller/building/splash.py
# defaults max_img_size to this, and raises on a larger image without Pillow.
_MAX_SPLASH = (760, 480)


def _load_make_splash():
    """tools/ carries no __init__.py, so make_splash.py is loaded from its
    file path rather than imported by module name."""
    spec = importlib.util.spec_from_file_location("make_splash", _MAKE_SPLASH_SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _png_size(blob: bytes) -> tuple[int, int]:
    """Width and height straight out of the IHDR chunk, so this check needs no
    image library of its own."""
    assert blob[:8] == b"\x89PNG\r\n\x1a\n", "not a PNG"
    assert blob[12:16] == b"IHDR", "first chunk is not IHDR"
    return int.from_bytes(blob[16:20], "big"), int.from_bytes(blob[20:24], "big")


def test_splash_sources_exist():
    assert _SPLASH_SVG.exists(), "assets/splash.svg is the drawn source of record"
    assert _SPLASH_PNG.exists(), "assets/splash.png is what PyInstaller reads"


def test_splash_png_is_a_png():
    _png_size(_SPLASH_PNG.read_bytes())


def test_splash_png_fits_without_pillow():
    """An image over max_img_size makes PyInstaller demand Pillow, which this
    project does not depend on, so the release build would fail."""
    width, height = _png_size(_SPLASH_PNG.read_bytes())
    assert width <= _MAX_SPLASH[0], width
    assert height <= _MAX_SPLASH[1], height


def test_splash_png_avoids_the_windows_transparency_key():
    """The bootloader treats pure magenta as transparent on Windows, so the art
    must not contain it or holes appear in the image. PNG pixel data is
    zlib-compressed, so the check has to decode actual pixels; a raw byte scan
    over the compressed file proves nothing about what the image shows."""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtGui import QImage

    image = QImage(str(_SPLASH_PNG))
    assert not image.isNull(), "could not decode splash.png"
    for y in range(image.height()):
        for x in range(image.width()):
            color = image.pixelColor(x, y)
            assert (color.red(), color.green(), color.blue()) != (255, 0, 255), (
                x,
                y,
            )


def test_splash_svg_source_is_xml_text():
    """This does not prove the PNG was generated from this SVG; that guarantee
    comes from the render round trip below. This only rules out an empty or
    non-XML file being checked in as the source of record."""
    assert _SPLASH_SVG.read_text(encoding="utf-8").lstrip().startswith("<")


def test_splash_png_matches_a_fresh_render_of_the_svg():
    """Nothing else regenerates splash.png from splash.svg on every run, so a
    committed PNG that has drifted from its SVG would otherwise pass every
    other check in this file."""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    # QApplication, not QGuiApplication: this suite shares one process-wide Qt
    # singleton across modules, and once a bare QGuiApplication claims it, it
    # can never be upgraded to the widget-capable QApplication other test
    # modules need.
    from PySide6.QtWidgets import QApplication

    QApplication.instance() or QApplication(sys.argv[:1])
    make_splash = _load_make_splash()
    blob, _, _ = make_splash.render_png(make_splash.SVG)
    assert blob == _SPLASH_PNG.read_bytes()


def test_spec_defines_the_splash_target():
    text = _SPEC.read_text(encoding="utf-8")
    assert "splash = Splash(" in text, (
        "vrcc.spec must define a Splash target so the bootloader paints "
        "before the interpreter starts"
    )
    assert 'os.path.join(REPO_ROOT, "assets", "splash.png")' in text


def test_spec_passes_the_splash_to_the_exe():
    """PyInstaller validates EXE's positional arguments against (PYZ, Splash)
    at PyInstaller/building/api.py:543-544, so the object goes in positionally
    rather than as a keyword."""
    text = _SPEC.read_text(encoding="utf-8")
    # Split on the closing paren at the start of a line, not the first one:
    # the EXE call contains os.path.join(...) and a naive split stops inside it.
    exe_call = text.split("exe = EXE(", 1)[1].split("\n)", 1)[0]
    assert "splash," in exe_call, (
        "the Splash object is passed positionally to EXE; without it the "
        "bootloader has nothing to show"
    )


def test_spec_collects_the_splash_binaries():
    text = _SPEC.read_text(encoding="utf-8")
    coll_call = text.split("coll = COLLECT(", 1)[1].split("\n)", 1)[0]
    assert "splash.binaries," in coll_call, (
        "a one-folder build needs the splash's own binaries in COLLECT"
    )


def test_spec_gives_the_splash_no_text():
    """The bootloader's text channel mangles a message by slicing between the
    first open paren and the last close paren, and CJK over it is unverified.
    This app ships in 17 languages, so the splash carries a logo and nothing
    else."""
    text = _SPEC.read_text(encoding="utf-8")
    splash_call = text.split("splash = Splash(", 1)[1].split("\n)", 1)[0]
    assert "text_pos" not in splash_call
    assert "text_size" not in splash_call
    assert "text_color" not in splash_call
