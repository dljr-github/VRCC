"""Streaming Silero VAD over the ONNX model bundled with faster-whisper.

Runs faster-whisper's ``silero_vad_v6.onnx`` directly, one 512-sample frame at
a time, persisting the recurrent state + 64-sample context across calls (its
own wrapper is batch-only; the streaming pip package needs torch, forbidden
here). Zero Qt.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import onnxruntime

_ASSET_NAME = "silero_vad_v6.onnx"
_CONTEXT = 64
_STATE_SHAPE = (2, 1, 1, 128)  # stacked (h, c), each [1, 1, 128]


def find_silero_onnx() -> Path:
    """Locate the Silero VAD ONNX asset bundled inside faster-whisper.

    Resolves it via the installed package location. Raises `FileNotFoundError`
    with a clear message if faster-whisper is absent or the asset is missing.
    """
    spec = importlib.util.find_spec("faster_whisper")
    if spec is None or not spec.submodule_search_locations:
        raise FileNotFoundError(
            "faster_whisper is not installed; cannot locate the Silero VAD "
            f"asset '{_ASSET_NAME}'"
        )

    for base in spec.submodule_search_locations:
        candidate = Path(base) / "assets" / _ASSET_NAME
        if candidate.is_file():
            return candidate

    searched = ", ".join(
        str(Path(base) / "assets") for base in spec.submodule_search_locations
    )
    raise FileNotFoundError(
        f"Silero VAD asset '{_ASSET_NAME}' not found under faster_whisper "
        f"assets (searched: {searched})"
    )


class StreamingVad:
    """Persistent-state Silero VAD; call `prob()` with one 512-sample float32
    frame at a time. Recurrent state + 64-sample context carry over between
    calls; `reset()` returns to cold start. `threshold` is stored for callers
    but never applied here. One reused single-threaded onnxruntime CPU session.
    """

    FRAME = 512

    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = threshold

        opts = onnxruntime.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        opts.enable_cpu_mem_arena = False
        # Silence onnxruntime's info/warning spam so test output stays clean.
        opts.log_severity_level = 3

        self._session = onnxruntime.InferenceSession(
            str(find_silero_onnx()),
            providers=["CPUExecutionProvider"],
            sess_options=opts,
        )

        # Stack h and c as one array so reset/copy stay trivial; split per call.
        self._state = np.zeros(_STATE_SHAPE, dtype=np.float32)
        self._context = np.zeros((1, _CONTEXT), dtype=np.float32)

    def reset(self) -> None:
        """Zero the recurrent state and audio context (cold-start baseline)."""
        self._state = np.zeros(_STATE_SHAPE, dtype=np.float32)
        self._context = np.zeros((1, _CONTEXT), dtype=np.float32)

    def prob(self, frame: np.ndarray) -> float:
        """Speech probability in [0, 1] for one 512-sample frame.

        `frame` must be float32-convertible with exactly `FRAME` samples (int16
        PCM is accepted); anything else raises `ValueError`. Advances the state.
        """
        try:
            samples = np.asarray(frame, dtype=np.float32)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"frame must be float32-convertible: {exc}"
            ) from exc

        if samples.shape != (self.FRAME,):
            raise ValueError(
                f"frame must be 1-D with {self.FRAME} samples, got shape "
                f"{samples.shape}"
            )

        # Prepend the previous frame's tail (64 samples) -> [1, 576].
        model_input = np.concatenate(
            [self._context, samples.reshape(1, self.FRAME)], axis=1
        )
        h, c = self._state[0], self._state[1]

        speech_probs, hn, cn = self._session.run(
            None,
            {"input": model_input, "h": h, "c": c},
        )

        self._state = np.stack([hn, cn])
        # This frame's last 64 samples become the next frame's context. Must be
        # a copy: np.asarray may return the caller's own array, and capture
        # loops reuse buffers in place -- a view would corrupt the next context.
        self._context = samples[-_CONTEXT:].copy().reshape(1, _CONTEXT)

        return float(speech_probs.reshape(-1)[0])
