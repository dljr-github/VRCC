import numpy as np
from vrcc.audio.stft import StreamingSTFT, N_FFT, HOP


def _process(sig):
    st = StreamingSTFT()
    out = []
    for i in range(0, len(sig) - HOP + 1, HOP):
        spec = st.analyze(sig[i:i + HOP].astype(np.float32))
        out.append(st.synthesize(spec))
    return np.concatenate(out) if out else np.zeros(0, np.float32)


def test_identity_reconstruction_in_steady_state():
    rng = np.random.default_rng(0)
    sig = rng.standard_normal(16000).astype(np.float32) * 0.1
    y = _process(sig)
    # Steady-state region (skip the first/last window warmup) reconstructs the
    # input delayed by one hop. With 50 percent overlap the algorithmic delay
    # of an analyze+synthesize pair is N_FFT - HOP, which equals HOP here.
    lag = HOP
    a = sig[HOP:HOP + 4000]
    b = y[HOP + lag:HOP + lag + 4000]
    n = min(len(a), len(b))
    err = np.max(np.abs(a[:n] - b[:n]))
    assert err < 1e-3, f"round-trip error too high: {err}"


def test_shapes_and_reset():
    st = StreamingSTFT()
    spec = st.analyze(np.zeros(HOP, np.float32))
    assert spec.shape == (257, 2) and spec.dtype == np.float32
    out = st.synthesize(spec)
    assert out.shape == (HOP,) and out.dtype == np.float32
    st.reset()  # must not raise
