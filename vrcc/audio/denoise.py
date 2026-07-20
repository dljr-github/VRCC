"""GTCRN speech denoiser: a gentle, torch-free noise-suppression stage.

Runs the bundled GTCRN streaming ONNX on onnxruntime (CPU), one 256-sample hop
at a time via StreamingSTFT, threading the model's three recurrent caches. The
enhanced signal is blended with a dry copy at `strength`, because full-strength
suppression damages short words. The dry copy passes through its own identical
StreamingSTFT round-trip, so it carries the same delay as the enhanced path and
the blend aligns with no hard-coded latency. Disabled is a zero-cost identity
bypass. configure() runs on the GUI thread (plain attribute stores, GIL-atomic);
process()/reset() run on the audio thread. Single audio-thread caller, same
lock-free rationale as GainProcessor.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from vrcc.audio.stft import HOP, StreamingSTFT

_ASSET = Path(__file__).resolve().parent / "gtcrn.onnx"


class Denoiser:
    def __init__(self, model_path: Path | None = None) -> None:
        self._path = Path(model_path) if model_path is not None else _ASSET
        self._enabled = False
        self._strength = 0.5
        self._session = None
        self._out_names: list[str] = []
        self._cache_names: list[str] = []
        self._cache_shapes: dict[str, list[int]] = {}
        self._caches: dict[str, np.ndarray] = {}
        self._stft = StreamingSTFT()      # enhanced path
        self._dry_stft = StreamingSTFT()  # dry path, identical delay
        self._in_buf = np.zeros(0, dtype=np.float32)   # samples awaiting a full hop
        self._out_buf = np.zeros(0, dtype=np.float32)  # blended samples ready to emit
        self._residual = 0.0

    def configure(self, enabled: bool, strength: float) -> None:
        self._strength = float(np.clip(strength, 0.0, 1.0))
        self._enabled = bool(enabled)

    def reset(self) -> None:
        self._stft.reset()
        self._dry_stft.reset()
        self._in_buf = np.zeros(0, dtype=np.float32)
        self._out_buf = np.zeros(0, dtype=np.float32)
        if self._session is not None:
            self._init_caches()

    def residual_noise_rms(self) -> float:
        return self._residual

    def process(self, frame: np.ndarray) -> np.ndarray:
        frame = np.asarray(frame, dtype=np.float32)
        if not self._enabled or frame.size == 0:
            return frame
        if not np.all(np.isfinite(frame)):
            return frame  # never let a bad frame poison recurrent state
        if self._session is None:
            self._load()
        n = frame.size
        self._in_buf = np.concatenate([self._in_buf, frame])
        while self._in_buf.size >= HOP:
            hop, self._in_buf = self._in_buf[:HOP], self._in_buf[HOP:]
            enh = self._stft.synthesize(self._run(self._stft.analyze(hop)))
            dry = self._dry_stft.synthesize(self._dry_stft.analyze(hop))
            self._residual = float(np.sqrt(np.mean((dry - enh) ** 2)))
            blended = self._strength * enh + (1.0 - self._strength) * dry
            self._out_buf = np.concatenate([self._out_buf, blended])
        if self._out_buf.size < n:
            pad = np.zeros(n - self._out_buf.size, dtype=np.float32)
            out = np.concatenate([pad, self._out_buf])
            self._out_buf = np.zeros(0, dtype=np.float32)
        else:
            out, self._out_buf = self._out_buf[:n], self._out_buf[n:]
        return np.clip(out, -1.0, 1.0).astype(np.float32)

    def _load(self) -> None:
        import onnxruntime as ort

        self._session = ort.InferenceSession(str(self._path), providers=["CPUExecutionProvider"])
        self._out_names = [o.name for o in self._session.get_outputs()]
        self._cache_names = [i.name for i in self._session.get_inputs() if i.name != "mix"]
        self._cache_shapes = {i.name: [int(d) for d in i.shape]
                              for i in self._session.get_inputs() if i.name != "mix"}
        self._init_caches()

    def _init_caches(self) -> None:
        self._caches = {n: np.zeros(self._cache_shapes[n], dtype=np.float32)
                        for n in self._cache_names}

    def _run(self, spec: np.ndarray) -> np.ndarray:
        mix = spec.reshape(1, 257, 1, 2).astype(np.float32)
        res = self._session.run(self._out_names, {"mix": mix, **self._caches})
        for name, val in zip(self._cache_names, res[1:]):
            self._caches[name] = val
        return res[0].reshape(257, 2)
