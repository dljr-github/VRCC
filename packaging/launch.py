"""PyInstaller entry point for the frozen VRCC.exe.

A tiny shim (instead of pointing the spec at ``vrcc/cli.py`` directly) so
the frozen app gets ``multiprocessing.freeze_support()`` -- required on
Windows in case any dependency spawns worker processes -- without touching
the real CLI module.
"""

import multiprocessing
import sys

from vrcc.cli import main

if __name__ == "__main__":
    multiprocessing.freeze_support()
    sys.exit(main())
