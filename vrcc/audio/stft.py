"""Pure-numpy streaming STFT/ISTFT for the GTCRN denoiser.

n_fft 512, hop 256, 50 percent overlap. Analysis uses a square-root periodic
Hann window (matching the torch.stft the model was trained with); synthesis
uses a square-root symmetric Hann with librosa-style window-squared
normalization, so the analysis/synthesis pair reconstructs the signal without a
separate gain constant. One synthesize() follows each analyze(); with 50
percent overlap the algorithmic delay is N_FFT - HOP, so output lags input by
one hop (about 16 ms). Zero Qt, no torch, single audio-thread caller.
"""

from __future__ import annotations

import numpy as np

N_FFT = 512
HOP = 256

# sqrt periodic Hann (torch.hann_window(512).pow(0.5)); sqrt symmetric Hann
# (np.hanning(512) ** 0.5) as the repo's librosa.istft uses.
_ANALYSIS = np.sqrt(0.5 - 0.5 * np.cos(2 * np.pi * np.arange(N_FFT) / N_FFT)).astype(np.float32)
_SYNTH = np.sqrt(np.hanning(N_FFT)).astype(np.float32)


class StreamingSTFT:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._in = np.zeros(N_FFT, dtype=np.float32)
        self._ola = np.zeros(N_FFT, dtype=np.float32)
        self._wsum = np.zeros(N_FFT, dtype=np.float32)

    def analyze(self, hop: np.ndarray) -> np.ndarray:
        self._in = np.concatenate([self._in[HOP:], np.asarray(hop, dtype=np.float32)])
        spec = np.fft.rfft(self._in * _ANALYSIS)
        return np.stack([spec.real, spec.imag], axis=-1).astype(np.float32)

    def synthesize(self, spec: np.ndarray) -> np.ndarray:
        comp = spec[..., 0] + 1j * spec[..., 1]
        frame = np.fft.irfft(comp, N_FFT).astype(np.float32)
        self._ola += frame * _SYNTH
        self._wsum += _SYNTH * _SYNTH
        sig = self._ola[:HOP].copy()
        wsum = self._wsum[:HOP].copy()
        self._ola = np.concatenate([self._ola[HOP:], np.zeros(HOP, dtype=np.float32)])
        self._wsum = np.concatenate([self._wsum[HOP:], np.zeros(HOP, dtype=np.float32)])
        out = np.where(wsum > 1e-8, sig / wsum, 0.0)
        return out.astype(np.float32)
