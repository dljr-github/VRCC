"""Kaldi-compatible log-mel filterbank plus the FunASR LFR/CMVN stack, in numpy.

The sherpa-onnx SenseVoice export ships only the acoustic graph: unlike the
onnx-asr NeMo exports (which carry their mel front-end as a bundled ONNX
graph), it expects features already computed the way Kaldi's ``compute-fbank``
and FunASR's ``WavFrontend`` produce them. This module is that front-end,
written against ``torchaudio.compliance.kaldi.fbank``'s exact semantics so the
numbers match the reference implementation bit-for-bit enough to reproduce
sherpa-onnx's own published transcripts (pinned in tests/test_stt_fbank.py).

Pure numpy on purpose: VRCC is torch-free and this has to run inside the
PyInstaller build. Zero Qt.
"""

from __future__ import annotations

import numpy as np

# torch.finfo(torch.float).eps -- the floor Kaldi's log() clamps to, so silent
# frames land on the same value the reference implementation produces rather
# than -inf.
_LOG_FLOOR = np.float32(1.1920928955078125e-07)

_PREEMPHASIS = 0.97
_LOW_FREQ = 20.0
_FRAME_LENGTH_MS = 25
_FRAME_SHIFT_MS = 10


def _mel_banks(num_bins: int, n_fft: int, sample_rate: int) -> np.ndarray:
    """Kaldi's triangular mel filterbank, shape ``(num_bins, n_fft // 2 + 1)``.

    Kaldi builds the triangles over ``n_fft // 2`` bins and leaves the Nyquist
    bin out; the trailing zero column restores the width the power spectrum
    actually has, which is what torchaudio does too.
    """
    nyquist = 0.5 * sample_rate
    fft_bin_width = sample_rate / n_fft

    def mel(freq):
        return 1127.0 * np.log(1.0 + freq / 700.0)

    mel_low, mel_high = mel(_LOW_FREQ), mel(nyquist)
    delta = (mel_high - mel_low) / (num_bins + 1)

    bins = np.arange(num_bins, dtype=np.float64)[:, None]
    left = mel_low + bins * delta
    center = mel_low + (bins + 1) * delta
    right = mel_low + (bins + 2) * delta

    freqs = mel(fft_bin_width * np.arange(n_fft // 2, dtype=np.float64))[None, :]
    up = (freqs - left) / (center - left)
    down = (right - freqs) / (right - center)
    banks = np.maximum(0.0, np.minimum(up, down))
    return np.pad(banks, ((0, 0), (0, 1)))


def fbank(
    samples: np.ndarray, sample_rate: int = 16000, num_mel_bins: int = 80
) -> np.ndarray:
    """Log-mel filterbank energies for ``samples``, shape ``(frames, num_mel_bins)``.

    ``samples`` must already be in the amplitude scale the model expects (see
    :func:`vrcc.stt.sensevoice.scale_samples`); the frame geometry is Kaldi's
    ``snip_edges=True``, so audio shorter than one 25 ms window yields zero
    frames rather than a padded partial one.
    """
    window_size = int(sample_rate * _FRAME_LENGTH_MS / 1000)
    hop = int(sample_rate * _FRAME_SHIFT_MS / 1000)
    if samples.shape[0] < window_size:
        return np.zeros((0, num_mel_bins), dtype=np.float32)

    n_frames = 1 + (samples.shape[0] - window_size) // hop
    offsets = np.arange(window_size)[None, :] + hop * np.arange(n_frames)[:, None]
    frames = samples[offsets].astype(np.float64)

    frames -= frames.mean(axis=1, keepdims=True)  # remove_dc_offset
    # Kaldi's preemphasis replicates the first sample rather than padding zero.
    previous = np.concatenate([frames[:, :1], frames[:, :-1]], axis=1)
    frames -= _PREEMPHASIS * previous

    index = np.arange(window_size)
    frames *= 0.54 - 0.46 * np.cos(2.0 * np.pi * index / (window_size - 1))

    n_fft = 1
    while n_fft < window_size:  # round_to_power_of_two -> 512 at 16 kHz
        n_fft *= 2
    padded = np.zeros((n_frames, n_fft), dtype=np.float64)
    padded[:, :window_size] = frames

    power = np.abs(np.fft.rfft(padded, n=n_fft)) ** 2
    energies = power @ _mel_banks(num_mel_bins, n_fft, sample_rate).T
    return np.log(np.maximum(energies, _LOG_FLOOR)).astype(np.float32)


def apply_lfr(feats: np.ndarray, window: int, shift: int) -> np.ndarray:
    """FunASR low frame rate stacking: ``window`` frames per step, hopping
    ``shift``, giving ``(ceil(frames / shift), num_mel_bins * window)``.

    The leading ``(window - 1) // 2`` frames repeat the first frame and a short
    final chunk repeats the last, matching FunASR's ``apply_lfr`` so the frame
    count the model sees is the one it was trained against.
    """
    if feats.shape[0] == 0:
        return np.zeros((0, feats.shape[1] * window), dtype=np.float32)

    padded = np.vstack([np.tile(feats[0], ((window - 1) // 2, 1)), feats])
    steps = int(np.ceil(padded.shape[0] / shift))
    stacked = []
    for i in range(steps):
        chunk = padded[i * shift: i * shift + window]
        if chunk.shape[0] < window:
            chunk = np.vstack([chunk, np.tile(padded[-1], (window - chunk.shape[0], 1))])
        stacked.append(chunk.reshape(-1))
    return np.stack(stacked).astype(np.float32)


def apply_cmvn(feats: np.ndarray, neg_mean: np.ndarray, inv_stddev: np.ndarray):
    """Global CMVN as FunASR stores it: add the (already negated) mean, then
    scale by the inverted standard deviation."""
    return (feats + neg_mean) * inv_stddev
