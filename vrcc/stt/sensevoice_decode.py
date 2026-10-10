"""Pure helpers for the SenseVoice engine: CTC confidence and vocab loading."""

from __future__ import annotations

from pathlib import Path

import numpy as np

# CTC blank. The export's vocab puts it at 0 ("<unk>"), matching what
# sherpa-onnx's own decoder assumes for this model family.
BLANK_ID = 0


def mean_nonblank_logprob(logits: np.ndarray, ids: np.ndarray) -> float | None:
    """Mean softmax logprob of each frame's chosen token, over frames whose
    argmax is not the CTC blank. Averaging over every frame instead would be
    dominated by blank frames (measured at 72-77% of frames regardless of
    noise, see module docstring), which barely moves between clean and
    degraded audio. ``None`` when every frame decoded blank."""
    max_logit = logits.max(axis=-1)
    logsumexp = np.log(np.exp(logits - max_logit[:, None]).sum(axis=-1)) + max_logit
    token_logprob = max_logit - logsumexp
    non_blank = ids != BLANK_ID
    if not non_blank.any():
        return None
    return float(token_logprob[non_blank].mean())


def read_vocab(path: Path) -> list[str]:
    """Read a sherpa-style ``tokens.txt`` ("<piece> <id>" per line) into a list
    indexed by id. ``rsplit`` on the last space so a piece that contains one
    still parses."""
    tokens: dict[int, str] = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if not line:
                continue
            piece, _, index = line.rpartition(" ")
            tokens[int(index)] = piece
    if not tokens:
        raise RuntimeError(f"{path} contains no tokens")
    return [tokens.get(i, "") for i in range(max(tokens) + 1)]
