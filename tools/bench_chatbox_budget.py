"""Measure how much of each target language survives the 144-char chatbox in
"truncate" mode, for every ordering of the targets.

Run from the repo root:

    python tools/bench_chatbox_budget.py
    python tools/bench_chatbox_budget.py --against main

``--against`` loads `vrcc/osc/chatbox_format.py` from a git ref and reports
that column beside the working tree's, which is how the before/after numbers
in the chatbox budget commits were produced.

The fixtures are imported from `tests/test_chatbox_budget.py` rather than
copied, so the numbers here always describe the strings the tests pin.
"""

from __future__ import annotations

import argparse
import importlib.util
import itertools
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tests.test_chatbox_budget import DE, ES, JA, ORIGINAL  # noqa: E402
from vrcc.core.config import OscConfig  # noqa: E402

TARGETS = [("ja", JA), ("es", ES), ("de", DE)]


def _load_module_from_ref(ref: str):
    """Import `vrcc/osc/chatbox_format.py` as it exists at `ref`."""
    source = subprocess.run(
        ["git", "show", f"{ref}:vrcc/osc/chatbox_format.py"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    ).stdout
    tmp = Path(tempfile.mkdtemp()) / "chatbox_format_ref.py"
    tmp.write_text(source, encoding="utf-8")
    spec = importlib.util.spec_from_file_location("chatbox_format_ref", tmp)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _delivered(module, ordering) -> dict[str, int]:
    """Characters of each target that reach the chatbox, by target name."""
    cfg = OscConfig(overflow="truncate")
    translations = [(name, text) for name, text in ordering]
    parts = module.fit_message(ORIGINAL, translations, cfg)
    sent = parts[0] if parts else ""
    out = {}
    for name, text in ordering:
        kept = 0
        for size in range(len(text), 0, -1):
            if text[:size] in sent:
                kept = size
                break
        out[name] = kept
    return out


def _report(module, label: str) -> None:
    print(f"\n{label}")
    print(f"  fixture lengths: " + ", ".join(f"{n}={len(t)}" for n, t in TARGETS))
    for ordering in itertools.permutations(TARGETS):
        kept = _delivered(module, ordering)
        cells = " ".join(
            f"{name}={kept[name]:>3}/{len(text):<3}" for name, text in ordering
        )
        lost = [name for name, _ in ordering if kept[name] == 0]
        flag = f"  LOST: {','.join(lost)}" if lost else ""
        print(f"  {' > '.join(n for n, _ in ordering):<12} {cells}{flag}")


def _policies() -> None:
    """What each way of dividing the limit gives the fixture trio."""
    from vrcc.osc.chatbox_format import CHATBOX_LIMIT, _share_limit

    sep = "\n"
    texts = [t for _, t in TARGETS]
    print(f"\nallocation policies (limit {CHATBOX_LIMIT}, separator {sep!r})")

    print("  equal shares, water-filled (what ships)")
    for (name, text), got in zip(TARGETS, _share_limit(texts, sep, CHATBOX_LIMIT)):
        body = got.rstrip("…")
        print(f"    {name} {len(body):>3}/{len(text):<4}{100 * len(body) // len(text):>3}%")

    room = CHATBOX_LIMIT - len(sep) * (len(texts) - 1)
    fraction = room / sum(len(t) for t in texts)
    print("  proportional")
    for name, text in TARGETS:
        kept = int(len(text) * fraction)
        print(f"    {name} {kept:>3}/{len(text):<4}{100 * kept // len(text):>3}%")

    whole, used = [], 0
    for name, text in sorted(TARGETS, key=lambda pair: len(pair[1])):
        cost = len(text) + (len(sep) if whole else 0)
        if used + cost <= CHATBOX_LIMIT:
            whole.append(name)
            used += cost
    dropped = [name for name, _ in TARGETS if name not in whole]
    print(f"  complete-units-first\n    whole={whole} dropped={dropped}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--against",
        metavar="REF",
        help="also measure vrcc/osc/chatbox_format.py as of this git ref",
    )
    parser.add_argument(
        "--policies",
        action="store_true",
        help="compare equal, proportional and complete-units-first division",
    )
    args = parser.parse_args()

    if args.against:
        _report(_load_module_from_ref(args.against), f"as of {args.against}")
    from vrcc.osc import chatbox_format

    _report(chatbox_format, "working tree")
    if args.policies:
        _policies()


if __name__ == "__main__":
    main()
