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
- ``collect_dynamic_libs("nvidia.cublas")`` is best-effort: the CUDA wheel
  is an optional extra (``pip install -e .[cuda]``); when it is absent the
  build is CPU-only and ``vrcc.core.hardware`` falls back gracefully.
"""

import os

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs

# The repo root, so the analysis can find the ``vrcc`` package as a plain
# source directory. (An editable ``pip install -e .`` uses a PEP 660 import
# hook that PyInstaller's static analysis cannot follow, and relative
# ``pathex`` entries resolve against the *current* directory, not the spec.)
REPO_ROOT = os.path.abspath(os.path.join(SPECPATH, os.pardir))  # noqa: F821

datas = collect_data_files("faster_whisper")
binaries = collect_dynamic_libs("ctranslate2")
try:
    binaries += collect_dynamic_libs("nvidia.cublas")
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
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="VRCC",
)
