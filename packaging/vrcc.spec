# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller one-folder spec for VRCC.

Build (from the repo root, inside the venv):

    .venv/Scripts/pyinstaller packaging/vrcc.spec --noconfirm --distpath dist

Produces ``dist/VRCC/VRCC.exe`` -- a windowed (no console) one-folder
build. Models are NOT bundled; the app downloads them on first run into the
per-user data directory (or next to the exe with ``--portable``).

Notes:
- ``collect_data_files("faster_whisper")`` bundles the package's assets
  directory, most importantly the Silero VAD ONNX model that
  ``vrcc.audio.vad`` loads at runtime.
- ``collect_dynamic_libs("ctranslate2")`` bundles ctranslate2.dll and the
  oneDNN/OpenMP runtimes next to the extension module.
- ``collect_dynamic_libs`` for the ``nvidia.*`` wheels is best-effort: the
  CUDA wheels (cuBLAS for CTranslate2, cuDNN for onnxruntime's CUDA provider)
  come from an optional extra (``pip install -e .[cuda]``); when absent the
  build is CPU-only and ``vrcc.core.hardware`` falls back gracefully.
"""

import glob
import os
import re

from PyInstaller.utils.hooks import (
    collect_data_files,
    collect_dynamic_libs,
    copy_metadata,
)
from PyInstaller.utils.win32.versioninfo import (
    FixedFileInfo,
    StringFileInfo,
    StringStruct,
    StringTable,
    VarFileInfo,
    VarStruct,
    VSVersionInfo,
)

# The repo root, so the analysis can find the ``vrcc`` package as a plain
# source directory. (An editable ``pip install -e .`` uses a PEP 660 import
# hook that PyInstaller's static analysis cannot follow, and relative
# ``pathex`` entries resolve against the *current* directory, not the spec.)
REPO_ROOT = os.path.abspath(os.path.join(SPECPATH, os.pardir))  # noqa: F821

# The exe version resource is built inline from the version in
# vrcc/__init__.py, so a version bump can never desync the exe metadata.
# Read with a regex, not an import: the spec must not trigger package
# imports (vrcc pulls in Qt and the audio stack at import time).
with open(
    os.path.join(REPO_ROOT, "vrcc", "__init__.py"), encoding="utf-8"
) as f:
    VERSION = re.search(
        r'^__version__ = "([^"]+)"', f.read(), re.MULTILINE
    ).group(1)
# VS_FIXEDFILEINFO wants exactly four numeric fields. Only the leading
# dotted release numbers count: digits from a pre-release suffix (1.2.0-rc1)
# must not land in the fourth field, or the rc's file version would compare
# newer than the final release that supersedes it.
_release = re.match(r"\d+(?:\.\d+)*", VERSION).group(0)
_nums = ([int(n) for n in _release.split(".")] + [0, 0, 0])[:4]
VERSION_4 = ".".join(str(n) for n in _nums)

version_info = VSVersionInfo(
    ffi=FixedFileInfo(filevers=tuple(_nums), prodvers=tuple(_nums)),
    kids=[
        StringFileInfo(
            [
                StringTable(
                    "040904B0",
                    [
                        StringStruct("CompanyName", "dljr-github"),
                        StringStruct(
                            "FileDescription",
                            "Live captions and translation for the "
                            "VRChat chatbox",
                        ),
                        StringStruct("FileVersion", VERSION_4),
                        StringStruct("InternalName", "VRCC"),
                        StringStruct(
                            "LegalCopyright",
                            "(c) 2026 dljr-github. MIT License.",
                        ),
                        StringStruct("OriginalFilename", "VRCC.exe"),
                        StringStruct("ProductName", "VRCC"),
                        StringStruct("ProductVersion", VERSION_4),
                    ],
                )
            ]
        ),
        VarFileInfo([VarStruct("Translation", [1033, 1200])]),
    ],
)

datas = collect_data_files("faster_whisper")
# onnx-asr ships its preprocessor ONNX graphs (nemo128 mel features etc.) as
# package data; the Parakeet engine needs them at runtime.
datas += collect_data_files("onnx_asr")
# onnx_asr reads its own version from dist metadata at import time, so the
# frozen app must ship the dist-info or the Parakeet engine load raises
# PackageNotFoundError.
datas += copy_metadata("onnx-asr")
# UI translation catalogs: vrcc.i18n loads them from the directory of its own
# __file__, which in a frozen build is _internal/vrcc/i18n/ -- exactly where
# this lands them.
datas += [
    (path, os.path.join("vrcc", "i18n"))
    for path in glob.glob(os.path.join(REPO_ROOT, "vrcc", "i18n", "*.json"))
]
# Window icon: vrcc.gui.style resolves it relative to the vrcc package, so
# it must land at _internal/vrcc/vrcc.ico, same shape as the i18n catalogs.
datas += [(os.path.join(REPO_ROOT, "vrcc", "vrcc.ico"), "vrcc")]
binaries = collect_dynamic_libs("ctranslate2")
# onnxruntime's own DLLs -- in the CUDA build this includes the CUDA execution
# provider libraries from the onnxruntime-gpu overlay (see release.yml), which
# Parakeet uses for GPU captions.
binaries += collect_dynamic_libs("onnxruntime")
for cuda_pkg in ("nvidia.cublas", "nvidia.cudnn"):
    try:
        binaries += collect_dynamic_libs(cuda_pkg)
    except Exception:
        pass  # CUDA extra not installed; CPU-only build.

hiddenimports = [
    # Imported lazily (inside functions) by vrcc.osc.*; keep them explicit
    # so a refactor to dynamic imports can never silently drop them.
    "pythonosc.udp_client",
    "pythonosc.dispatcher",
    "pythonosc.osc_server",
    "zeroconf",
    # Imported lazily by vrcc.core.hardware / vrcc.gui.firstrun for VRAM /
    # compute-capability / driver-version queries (nvidia-ml-py).
    "pynvml",
    # Imported lazily by vrcc.stt.onnx_asr at engine load time.
    "onnx_asr",
]

a = Analysis(
    ["launch.py"],
    pathex=[REPO_ROOT],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "tkinter",
        "matplotlib",
        "IPython",
    ],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="VRCC",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,  # windowed app; logs go to the rotating file in logs_dir
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=os.path.join(REPO_ROOT, "vrcc", "vrcc.ico"),
    version=version_info,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="VRCC",
)
