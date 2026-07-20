"""Live source-delegation helpers for the pipeline: gain and denoise pushes
to the current audio source. Module functions take the Pipeline instance
``p``; only the per-call delegation logic lives here, mirroring pipeline_jobs.
Import direction: pipeline imports this module (never the reverse at runtime).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vrcc.core.pipeline import Pipeline


def set_source_gain(p: "Pipeline", gain_db: float, auto: bool) -> None:
    """Push a live gain change to the current source. No-op if the source
    has no gain processor."""
    setter = getattr(p._source, "set_gain", None)
    if setter is not None:
        setter(gain_db, auto)


def set_source_denoise(p: "Pipeline", enabled: bool, strength: float) -> None:
    """Push a live denoise change to the current source. No-op if the source
    has no denoise processor."""
    setter = getattr(p._source, "set_denoise", None)
    if setter is not None:
        setter(enabled, strength)
