import argparse
import os
import sys


def _ensure_std_streams() -> None:
    """Give ``sys.stdout``/``sys.stderr`` a real stream when missing.

    A windowed PyInstaller build / ``pythonw.exe`` sets them to ``None``, so
    writes (e.g. huggingface_hub's tqdm) crash. Point them at the null device --
    real diagnostics go to the log file.
    """
    for name in ("stdout", "stderr"):
        if getattr(sys, name, None) is None:
            setattr(sys, name, open(os.devnull, "w", encoding="utf-8"))


def main() -> int:
    _ensure_std_streams()
    parser = argparse.ArgumentParser(prog="vrcc")
    parser.add_argument("--portable", action="store_true", help="store config/models next to the app")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    from vrcc.app import run
    return run(portable=args.portable, verbose=args.verbose)


if __name__ == "__main__":
    sys.exit(main())
