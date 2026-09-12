import argparse
import os
import sys

from vrcc.core.instance import InstanceGuard


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
    """Parse arguments, then decide whether this copy may run.

    Order is a contract, not a preference: the single-instance guard sits
    after parse_args and before the boot import, so a refused launch exits
    without paying for sounddevice, huggingface_hub, faster_whisper and Qt
    itself. Nothing Qt may move above it, or a second copy would flash a
    window and vanish.
    """
    _ensure_std_streams()
    parser = argparse.ArgumentParser(prog="vrcc")
    parser.add_argument("--portable", action="store_true", help="store config/models next to the app")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    guard = InstanceGuard()
    if not guard.acquire():
        # Silent by design: any message here would be raised before a UI
        # language is loaded. The feedback is the running copy showing
        # itself, or its taskbar button flashing.
        guard.ring()
        return 0

    try:
        from vrcc.boot import boot

        return boot(portable=args.portable, verbose=args.verbose, guard=guard)
    finally:
        guard.release()


if __name__ == "__main__":
    sys.exit(main())
