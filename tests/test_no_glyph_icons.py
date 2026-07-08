"""Guard: no emoji or symbol-glyph-as-icon characters anywhere in ``vrcc``
source (icons are drawn SVGs; state is words + color; user-visible paths use
ASCII ">" / "->", never arrow glyphs). Typography glyphs stay allowed.
"""

from pathlib import Path

_SRC_DIR = Path(__file__).resolve().parent.parent / "vrcc"

# Every emoji plane codepoint, plus the specific symbol glyphs that have been
# used as icons before (the arrow also covers app.py's "Settings > Audio" copy).
_FORBIDDEN = set("●▶▸✓✗✕✖→⋯▾☰⚙✂■⏸⏵⏹")
_EMOJI_FLOOR = 0x1F000


def test_vrcc_sources_contain_no_glyph_icons():
    files = sorted(_SRC_DIR.rglob("*.py"))
    assert files, f"no sources found under {_SRC_DIR}"
    offenders = []
    for path in files:
        for lineno, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), 1
        ):
            for ch in line:
                if ord(ch) >= _EMOJI_FLOOR or ch in _FORBIDDEN:
                    rel = path.relative_to(_SRC_DIR.parent)
                    offenders.append(f"{rel}:{lineno} U+{ord(ch):04X}")
    assert not offenders, "glyph-as-icon characters in source: " + ", ".join(offenders)
