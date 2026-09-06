"""Measure how much of each target language survives the 144-char chatbox in
"truncate" mode, for every ordering of the targets.

Run from the repo root:

    python tools/bench_chatbox_budget.py
    python tools/bench_chatbox_budget.py --against main

``--against`` loads `vrcc/osc/chatbox_format.py` from a git ref, serving every
`vrcc.osc` module it imports from that same ref, and reports that column
beside the working tree's, which is how the before/after numbers in the
chatbox budget commits were produced.

The fixtures are imported from `tests/test_chatbox_budget.py` rather than
copied, so the numbers here always describe the strings the tests pin.
"""

from __future__ import annotations

import argparse
import atexit
import importlib.abc
import importlib.util
import itertools
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tests.test_chatbox_budget import DE, ES, JA, ORIGINAL  # noqa: E402
from vrcc.core.config import OscConfig  # noqa: E402

TARGETS = [("ja", JA), ("es", ES), ("de", DE)]

# Every module the shaping code reaches that decides how a text is cut.
# `vrcc.core.charclass` is named rather than the whole `vrcc.core` package:
# the harness builds the OscConfig it hands the ref's fit_message, so
# `vrcc.core.config` has to stay the working tree's one type.
_REF_PREFIXES = ("vrcc.osc.", "vrcc.core.charclass")


def _resolve(ref: str) -> str:
    """The commit `ref` names. Raises on a ref git cannot resolve, so a
    mistyped ref fails here rather than loading the working tree under its
    label."""
    return subprocess.run(
        ["git", "rev-parse", "--verify", f"{ref}^{{commit}}"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def _git_show(commit: str, rel_path: str) -> str | None:
    """`git show <commit>:<rel_path>`'s stdout, or `None` when the commit has
    no such path.

    Existence is asked first, so `git show` keeps `check=True`: any other git
    failure (a corrupt object, a lock) raises here rather than reading as a
    missing path, which would let the ref's chatbox_format fall through to
    the working tree's modules and report a hybrid under the ref's label."""
    exists = subprocess.run(
        ["git", "cat-file", "-e", f"{commit}:{rel_path}"],
        cwd=ROOT,
        capture_output=True,
    )
    if exists.returncode != 0:
        return None
    return subprocess.run(
        ["git", "show", f"{commit}:{rel_path}"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    ).stdout


def _is_ref_module(fullname: str) -> bool:
    return any(fullname.startswith(prefix) for prefix in _REF_PREFIXES)


class _RefFinder(importlib.abc.MetaPathFinder):
    """Serve every shaping import from `commit` while a ref's
    chatbox_format.py is being executed, whatever modules it reaches and in
    whatever order: a hand-kept list of them goes stale at the next split."""

    def __init__(self, commit: str, tmp_dir: Path) -> None:
        self._commit = commit
        self._dir = tmp_dir

    def find_spec(self, fullname, path, target=None):
        if not _is_ref_module(fullname):
            return None
        source = _git_show(self._commit, fullname.replace(".", "/") + ".py")
        if source is None:
            return None
        # Named for the full dotted path: two packages can hold a module of
        # the same last segment, and the file backs a traceback's line
        # numbers for the lifetime of the process.
        file = self._dir / f"{fullname.replace('.', '_')}_ref.py"
        file.write_text(source, encoding="utf-8")
        return importlib.util.spec_from_file_location(fullname, file)


def _load_module_from_ref(ref: str):
    """Import `vrcc/osc/chatbox_format.py` as it exists at `ref`.

    The working tree's shaping modules (`_REF_PREFIXES`) are evicted from
    `sys.modules` for the duration and a finder serves the ref's copies in
    their place, so a ref's chatbox_format cannot fall through to today's
    chatbox_slice, linebreak or charclass and report a hybrid under the ref's
    label. Everything is restored on exit, since `main()` imports the real
    modules afterwards in the same process. The extracted sources outlive the
    call, so a traceback inside the ref's code still shows its lines.
    """
    commit = _resolve(ref)
    tmp_dir = Path(tempfile.mkdtemp())
    atexit.register(shutil.rmtree, tmp_dir, ignore_errors=True)
    evicted = {
        name: sys.modules.pop(name)
        for name in list(sys.modules)
        if _is_ref_module(name)
    }
    finder = _RefFinder(commit, tmp_dir)
    sys.meta_path.insert(0, finder)
    try:
        source = _git_show(commit, "vrcc/osc/chatbox_format.py")
        if source is None:
            raise FileNotFoundError(f"{ref} has no vrcc/osc/chatbox_format.py")
        file = tmp_dir / "chatbox_format_ref.py"
        file.write_text(source, encoding="utf-8")
        spec = importlib.util.spec_from_file_location("chatbox_format_ref", file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.meta_path.remove(finder)
        # The parent package keeps an attribute per submodule alongside the
        # sys.modules entry, so both have to be put back or a later
        # `vrcc.osc.chatbox_slice` attribute lookup finds the ref's copy.
        for name in [n for n in sys.modules if _is_ref_module(n)]:
            del sys.modules[name]
            parent, _, leaf = name.rpartition(".")
            if parent in sys.modules:
                sys.modules[parent].__dict__.pop(leaf, None)
        for name, module in evicted.items():
            sys.modules[name] = module
            parent, _, leaf = name.rpartition(".")
            if parent in sys.modules:
                setattr(sys.modules[parent], leaf, module)


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
    print("  fixture lengths: " + ", ".join(f"{n}={len(t)}" for n, t in TARGETS))
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
