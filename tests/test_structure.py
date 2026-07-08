"""Permanent structure guard: every Python source file stays at or under the
500-line cap (the whole-codebase invariant behind the structure refactors).
"""

from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_DIRS = ("vrcc", "tests", "tools")
_MAX_LINES = 500


def test_no_python_file_exceeds_500_lines():
    offenders = []
    checked = 0
    for top in _DIRS:
        for path in sorted((_ROOT / top).rglob("*.py")):
            checked += 1
            count = len(path.read_text(encoding="utf-8").splitlines())
            if count > _MAX_LINES:
                offenders.append(f"{path.relative_to(_ROOT)}: {count} lines")
    assert checked, f"no Python sources found under {_ROOT}"
    assert not offenders, (
        f"files exceed the {_MAX_LINES}-line cap (split them):\n  "
        + "\n  ".join(offenders)
    )
