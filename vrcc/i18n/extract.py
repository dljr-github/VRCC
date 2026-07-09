"""AST extraction of translatable source strings. Zero Qt, stdlib only.

A source string is the literal first argument of any ``tr(...)`` or
``tr_noop(...)`` call under the ``vrcc`` package. The catalog tests compare
every ``vrcc/i18n/*.json`` against this extraction (fully translated, no
stale keys, placeholders intact), and catalog (re)generation starts from it.
Dynamic first arguments (``tr(friendly)``) are deliberately invisible here:
their literals must be marked ``tr_noop`` where they are defined.
"""

from __future__ import annotations

import ast
import string
from pathlib import Path

_MARKER_FUNCS = {"tr", "tr_noop"}


def extract_from_source(source: str, filename: str = "<string>") -> list[tuple[str, int]]:
    """``(text, lineno)`` for each literal ``tr()``/``tr_noop()`` first
    argument in ``source``, in file order."""
    tree = ast.parse(source, filename=filename)
    found: list[tuple[str, int]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not node.args:
            continue
        func = node.func
        if isinstance(func, ast.Name):
            name = func.id
        elif isinstance(func, ast.Attribute):
            name = func.attr
        else:
            continue
        if name not in _MARKER_FUNCS:
            continue
        first = node.args[0]
        if isinstance(first, ast.Constant) and isinstance(first.value, str):
            found.append((first.value, node.lineno))
    return found


def extract_source_strings(root: Path | None = None) -> dict[str, list[str]]:
    """Map every translatable source string to its ``path:line`` locations,
    scanning each ``*.py`` under ``root`` (default: the ``vrcc`` package)."""
    if root is None:
        root = Path(__file__).resolve().parent.parent
    locations: dict[str, list[str]] = {}
    for path in sorted(root.rglob("*.py")):
        rel = path.relative_to(root.parent)
        for text, lineno in extract_from_source(
            path.read_text(encoding="utf-8"), str(path)
        ):
            locations.setdefault(text, []).append(f"{rel}:{lineno}")
    return locations


def placeholder_names(text: str) -> set[str]:
    """The ``str.format`` field names in ``text`` (``"{pct}%"`` -> ``{"pct"}``).

    Raises ``ValueError`` on malformed format syntax, so a test can flag a
    catalog entry whose braces were mangled in translation.
    """
    return {
        field.split(".")[0].split("[")[0]
        for _, field, _, _ in string.Formatter().parse(text)
        if field is not None
    }
